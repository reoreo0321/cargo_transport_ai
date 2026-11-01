#!/usr/bin/env python3
"""Local-only Ollama extraction; no API key or cloud fallback."""
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

from mission_dialogue import normalize, station_candidates
from mission_language_backend import response_schema, BackendError, RuleBackend


PRIMARY_FIELDS = ["start", "goal", "cargo_description", "handling", "shape"]
OPTIONAL_FIELDS = [
    "dimensions_m",
    "mass_kg",
    "fragility",
    "slippery",
    "secured",
    "box_approximation",
]
LOCAL_PROMPT = """運搬指示から5項目を抽出する。JSONだけを返す。
ユーザーの最後のメッセージが今回の発言。current_slotsは既存の回答。今回言われていない項目はnullにして既存の回答を保持する。
質問への短い回答ではquestionsの項目だけを更新する。形状の回答は荷物名の変更ではない。明示的な訂正は質問対象外の項目も更新できる。
start/goalはstationsの別名をA/B/Cに変換する。「倉庫にある物を運ぶ」は出発地A。「作業台」だけはBとCが曖昧。
cargo_descriptionは自由な荷物名。shapeはbox=箱、tall_box=背の高い/縦長、flat_box=平たい、cylinder=円筒、other=複雑な形、unknown=形状不明という明示。
工具箱の「箱」はbox。顕微鏡などの品名だけから形を推測せず、形が言われていなければshape=null。
handlingはcareful=慎重/丁寧/ゆっくり、standard=標準/普通に/いつもどおり、time_priority=時間優先/急ぎ、unknown=おまかせ。
「慎重じゃなく時間優先」はtime_priority。「慎重、でも急いで」で優先順位が未定ならhandling=null、ambiguous_fields=["handling"]。
evidenceは各値の根拠となる今回のutteranceの原文の部分文字列。地点は原文の地点名だけ。検品するところを勝手に検査場と言い換えない。更新しない項目のevidenceはnull。
questionsは発言前の不足項目。今回解決した項目をambiguous_fieldsに入れない。通常ambiguous_fields=[]。複数の解釈が残る項目だけ入れる。
短い「不明」「おまかせ」はquestionsの対象だけをunknownへ更新（地点を除く）。荷物が不明なら「不明な荷物」。他の既存回答は消さない。
寸法・質量などの物理値は別処理なので出力不要。品名から物理特性を推測しない。コードや任意の速度は出力しない。
例「倉庫から組立場へ工具箱を丁寧に運んで」：
{"slots":{"start":"A","goal":"B","cargo_description":"工具箱","handling":"careful","shape":"box"},"evidence":{"start":"倉庫","goal":"組立場","cargo_description":"工具箱","handling":"丁寧","shape":"工具箱"},"ambiguous_fields":[]}
例「検査するとこまで工具箱を運んで」ならgoal=C,cargo_description=工具箱,shape=box。他はnull、ambiguous_fields=[]。不足条件の質問はプログラムが行う。"""


def local_schema(stations):
    schema = response_schema()
    for group in ["slots", "evidence"]:
        schema["properties"][group]["properties"] = {
            k: v
            for k, v in schema["properties"][group]["properties"].items()
            if k in PRIMARY_FIELDS
        }
        schema["properties"][group]["required"] = PRIMARY_FIELDS
    for field in ["start", "goal"]:
        schema["properties"]["slots"]["properties"][field]["enum"] = list(stations) + [
            None
        ]
    schema["properties"]["ambiguous_fields"]["items"]["enum"] = PRIMARY_FIELDS
    return schema


def ground_primary(patch, text, context, literal):
    """Project a full-state answer to updates and ground quotes in this input.

    The model still chooses the location IDs and free cargo name. This adapter
    links selected IDs to literal map aliases, without choosing a route itself.
    """
    text = normalize(text)
    changes = []
    ambiguous = patch["ambiguous_fields"]

    def ungrounded(field):
        declared = literal["slots"].get(field)
        prior = context["slots"].get(field)
        patch["slots"][field] = None
        patch["evidence"][field] = None
        if (
            prior is not None
            and declared is not None
            and declared != prior
            and field not in literal["ambiguous_fields"]
            and not re.search(r"じゃなく|ではなく|でなく|ではない|じゃない", text)
        ):
            patch["evidence"][field] = literal["evidence"].get(field)
            if field not in patch["ambiguous_fields"]:
                patch["ambiguous_fields"].append(field)
            changes.append(field + ":unresolved_current_correction")
        else:
            if field in patch["ambiguous_fields"]:
                patch["ambiguous_fields"].remove(field)
            changes.append(field + ":ungrounded_proposal_ignored")

    unknown_words = r"不明|わから|分から|わかりません|分かりません|わかんない|知らない|不詳|おまかせ|お任せ|任せ|未確認|unknown"
    for field in ["handling", "shape"]:
        value = patch["slots"].get(field)
        quote = patch["evidence"].get(field)
        if value == "unknown" and not (
            isinstance(quote, str)
            and normalize(quote) in text
            and re.search(unknown_words, normalize(quote), re.I)
        ):
            patch["slots"][field] = None
            if field not in ambiguous:
                patch["evidence"][field] = None
            changes.append(field + ":unreported_unknown_ignored")
    # A literal shape declaration resolves an "unknown" model guess; do not
    # use names such as 顕微鏡 to infer any shape or physical property.
    shape = literal["slots"]["shape"]
    if (
        patch["slots"].get("shape") is None
        and shape not in [None, "unknown"]
        and "shape" not in ambiguous
        and not re.search(r"じゃなく|ではなく|でなく|ではない|じゃない", text)
    ):
        patch["slots"]["shape"] = shape
        patch["evidence"]["shape"] = literal["evidence"]["shape"]
        changes.append("shape:explicit_literal")
    specific_shapes = [
        r"背の高い|背が高い|縦長",
        r"平たい|平べったい|薄い箱|平らな箱",
        r"円筒|筒状",
        r"複雑な形|その他の形",
    ]
    if (
        shape is not None
        and patch["slots"].get("shape") == shape
        and "shape" in ambiguous
        and sum(bool(re.search(pattern, text)) for pattern in specific_shapes) <= 1
        and not re.search(
            r"か(?!ら|ぎり)|または|どちら|どっち|それとも|じゃなく|ではなく|でなく",
            text,
        )
    ):
        ambiguous.remove("shape")
        changes.append("shape:explicit_declaration_resolves_ambiguity")
    handling = literal["slots"]["handling"]
    if (
        handling not in [None, "unknown"]
        and patch["slots"].get("handling") == handling
        and "handling" in ambiguous
        and "handling" not in literal["ambiguous_fields"]
        and not re.search(
            r"か(?!ら|ぎり)|または|どちら|どっち|それとも|じゃなく|ではなく|でなく",
            text,
        )
    ):
        ambiguous.remove("handling")
        changes.append("handling:explicit_declaration_resolves_ambiguity")
    correction = re.search(
        r"(?:じゃなく|ではなく|でなく)\s*(時間優先|速さ優先|スピード優先|慎重|丁寧|標準)",
        text,
    )
    if correction and not re.match(
        r"ではない|じゃない|ではなく|じゃなく", text[correction.end() :]
    ):
        value = {
            "時間優先": "time_priority",
            "速さ優先": "time_priority",
            "スピード優先": "time_priority",
            "慎重": "careful",
            "丁寧": "careful",
            "標準": "standard",
        }[correction[1]]
        patch["slots"]["handling"] = value
        patch["evidence"]["handling"] = correction.group()
        patch["ambiguous_fields"] = [f for f in ambiguous if f != "handling"]
        ambiguous = patch["ambiguous_fields"]
        changes.append("handling:explicit_negation_correction")
    for field, value in list(patch["slots"].items()):
        if value is None:
            continue
        if value == context["slots"].get(field) and field not in ambiguous:
            declared = literal["slots"].get(field)
            if (
                declared is not None
                and declared != value
                and field not in literal["ambiguous_fields"]
                and not re.search(r"じゃなく|ではなく|でなく|ではない|じゃない", text)
            ):
                # A stale full-state answer must not silently ignore a newly
                # declared correction. Ask for this field instead of executing.
                patch["slots"][field] = None
                patch["evidence"][field] = literal["evidence"].get(field)
                patch["ambiguous_fields"].append(field)
                changes.append(field + ":unresolved_current_correction")
                continue
            patch["slots"][field] = None
            patch["evidence"][field] = None
            changes.append(field + ":unchanged_state_retained")
            continue
        quote = patch["evidence"].get(field)
        if field in ["start", "goal"]:
            grounded = False
            aliases = sorted(
                {a for s in context["stations"].values() for a in s["aliases"]},
                key=len,
                reverse=True,
            )
            for alias in aliases:
                pattern = re.escape(normalize(alias))
                if re.fullmatch("[A-Za-z]", alias):
                    pattern = r"(?<![A-Za-z0-9])" + pattern + r"(?![A-Za-z0-9])"
                found = re.search(pattern, text, re.I)
                if found and station_candidates(
                    found.group(), dict(stations=context["stations"])
                ) == [value]:
                    if quote != found.group():
                        changes.append(field + ":literal_map_alias")
                    patch["evidence"][field] = found.group()
                    grounded = True
                    break
            if not grounded:
                ungrounded(field)
        elif (
            field == "cargo_description"
            and isinstance(value, str)
            and normalize(value) in text
        ):
            patch["evidence"][field] = normalize(value)
        elif field == "cargo_description" and (
            not isinstance(quote, str)
            or not normalize(quote)
            or normalize(quote) not in text
        ):
            ungrounded(field)
        elif field in ["handling", "shape"] and (
            not isinstance(quote, str)
            or not normalize(quote)
            or normalize(quote) not in text
        ):
            if literal["slots"].get(field) == value:
                patch["evidence"][field] = literal["evidence"][field]
                changes.append(field + ":explicit_literal_evidence")
            else:
                ungrounded(field)
    return changes


class OllamaBackend:
    name = "ollama"

    def __init__(self, model=None, base_url=None, transport=None):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL", "http://cargo-ollama:11434")
        ).rstrip("/")
        parsed = urllib.parse.urlsplit(self.base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname
            not in ["cargo-ollama", "localhost", "127.0.0.1", "host.docker.internal"]
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path
        ):
            raise BackendError(
                "Ollamaはローカル接続先を指定してください。例：http://cargo-ollama:11434"
            )
        if "cloud" in self.model.casefold():
            raise BackendError(
                "ローカル実行するモデルを指定してください。cloudモデルは使用しません。"
            )
        # Ignore external HTTP proxies for the local model endpoint.
        self.transport = (
            transport
            or urllib.request.build_opener(urllib.request.ProxyHandler({})).open
        )
        self.last_metadata = {}

    def extract(self, text, context):
        known = {
            f: context["slots"][f]
            for f in PRIMARY_FIELDS
            if context["slots"][f] is not None
        }
        background = dict(
            current_slots=known,
            questions=(
                [
                    f
                    for q in context["questions"]
                    for f in q["fields"]
                    if f in PRIMARY_FIELDS
                ]
                if known
                else []
            ),
            stations=context["stations"],
        )
        payload = dict(
            model=self.model,
            messages=[
                dict(
                    role="system",
                    content=LOCAL_PROMPT
                    + "\n地図と会話の状態:\n"
                    + json.dumps(background, ensure_ascii=False),
                ),
                dict(role="user", content=normalize(text)),
            ],
            format=local_schema(context["stations"]),
            stream=False,
            keep_alive="5m",
            options=dict(
                temperature=0, seed=42, num_ctx=4096, num_predict=1024, use_mmap=True
            ),
        )
        if self.model.startswith("qwen3"):
            payload["think"] = False
        request = urllib.request.Request(
            self.base_url + "/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.transport(request, timeout=300) as response:
                raw = response.read(1_000_001)
            if len(raw) > 1_000_000:
                raise BackendError(
                    "ローカルLLMの応答が大きすぎます。条件は更新していません。"
                )
            result = json.loads(raw)
        except urllib.error.HTTPError as exc:
            raise BackendError(
                f"Ollama接続エラー（HTTP {exc.code}）。サーバーの起動とモデル取得を確認してください。"
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise BackendError(
                "ローカルLLMに接続できないか応答がタイムアウトしました。cargo_ollamaの起動を確認してください。"
            ) from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise BackendError(
                "Ollama応答を読み取れませんでした。条件は更新していません。"
            ) from None
        if (
            not isinstance(result, dict)
            or not result.get("done")
            or result.get("done_reason") != "stop"
        ):
            raise BackendError(
                "ローカルLLMの応答が完了していません。条件は更新していません。"
            )
        message = result.get("message", {})
        content = message.get("content") if isinstance(message, dict) else None
        try:
            patch = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            raise BackendError(
                "ローカルLLMから条件のJSONを受け取れませんでした。"
            ) from None
        if not isinstance(patch, dict):
            raise BackendError("ローカルLLMの条件形式が正しくありません。")
        if (
            not isinstance(patch.get("slots"), dict)
            or not isinstance(patch.get("evidence"), dict)
            or set(patch["slots"]) - set(PRIMARY_FIELDS)
            or set(patch["evidence"]) - set(PRIMARY_FIELDS)
        ):
            raise BackendError(
                "ローカルLLMが担当外の項目を返しました。条件は更新していません。"
            )
        ambiguous = patch.get("ambiguous_fields", [])
        if not isinstance(ambiguous, list) or any(
            f not in PRIMARY_FIELDS for f in ambiguous
        ):
            raise BackendError(
                "ローカルLLMの曖昧項目が正しくありません。条件は更新していません。"
            )
        raw_primary = json.loads(json.dumps(patch))
        literal = RuleBackend().extract(text, context)
        patch["ambiguous_fields"] = ambiguous
        normalizations = ground_primary(patch, text, context, literal)
        for field in OPTIONAL_FIELDS:
            if literal["slots"][field] is not None:
                patch["slots"][field] = literal["slots"][field]
                patch["evidence"][field] = literal["evidence"][field]
        patch.setdefault("ambiguous_fields", []).extend(
            f for f in literal["ambiguous_fields"] if f in OPTIONAL_FIELDS
        )
        self.last_metadata = {
            k: result[k]
            for k in [
                "model",
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            ]
            if k in result
        }
        self.last_metadata["optional_fields_from_rules"] = [
            f for f in OPTIONAL_FIELDS if literal["slots"][f] is not None
        ]
        self.last_metadata["raw_primary_patch"] = raw_primary
        self.last_metadata["normalizations"] = normalizations
        return patch

    def release(self):
        """Free model memory before Gazebo runs. Failure cannot alter a mission."""
        request = urllib.request.Request(
            self.base_url + "/api/generate",
            data=json.dumps(dict(model=self.model, keep_alive=0)).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.transport(request, timeout=10) as response:
                response.read(10000)
        except (urllib.error.URLError, TimeoutError, OSError):
            return False
        return True
