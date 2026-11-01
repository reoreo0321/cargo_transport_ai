#!/usr/bin/env python3
"""Mission slots, clarification and independent validation; no ROS or API needed."""
import copy
import json
import math
import re
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/mission_language.json"
FIELDS = [
    "start",
    "goal",
    "cargo_description",
    "handling",
    "shape",
    "dimensions_m",
    "mass_kg",
    "fragility",
    "slippery",
    "secured",
    "box_approximation",
]
REQUIRED = ["start", "goal", "cargo_description", "handling", "shape"]
ENUMS = {
    "handling": ["careful", "standard", "time_priority", "unknown"],
    "shape": ["box", "tall_box", "flat_box", "cylinder", "other", "unknown"],
    "fragility": ["fragile", "robust", "unknown"],
    "slippery": ["yes", "no", "unknown"],
    "secured": ["yes", "no", "unknown"],
    "box_approximation": ["yes", "no", "unknown"],
}


def normalize(text):
    return unicodedata.normalize("NFKC", text).strip()


def load_config(path=CONFIG):
    c = json.loads(path.read_text(encoding="utf-8"))
    if set(c["stations"]) != {"A", "B", "C"}:
        raise ValueError("地図の地点A/B/Cを設定してください")
    for p in c["profiles"].values():
        if (
            not 0.1 <= p["speed_mps"] <= 0.65
            or not 0.05 <= p["acceleration_mps2"] <= 0.30
        ):
            raise ValueError("走行設定が対応範囲外です")
    return c


def station_candidates(text, c):
    token = normalize(text).casefold()
    direct = [
        k
        for k, s in c["stations"].items()
        if token in [normalize(a).casefold() for a in s["aliases"]]
    ]
    return direct


def validate_value(field, value, c):
    if field in ["start", "goal"]:
        if not isinstance(value, str) or value not in c["stations"]:
            raise ValueError(
                "地点はA（部品置き場）・B（組立場）・C（検査場）から指定してください"
            )
    elif field in ENUMS:
        if value not in ENUMS[field]:
            raise ValueError("対応する選択肢から指定してください")
    elif field == "cargo_description":
        if not isinstance(value, str) or not 1 <= len(value.strip()) <= 120:
            raise ValueError("運ぶ物を120文字以内で指定してください")
    elif field == "dimensions_m":
        if isinstance(value, str):
            if value not in ["standard", "unknown"]:
                raise ValueError("寸法、標準サイズ、または不明を指定してください")
        elif (
            not isinstance(value, list)
            or len(value) != 3
            or any(type(v) not in [int, float] or not math.isfinite(v) for v in value)
        ):
            raise ValueError(
                "寸法は前後×左右×高さの3つで指定してください。例：18×16×18cm"
            )
        elif not all(0.06 <= v <= limit for v, limit in zip(value, [0.45, 0.30, 0.35])):
            raise ValueError(
                "この荷台の対応寸法は前後6〜45cm・左右6〜30cm・高さ6〜35cmです"
            )
    elif field == "mass_kg":
        if value != "unknown" and (
            type(value) not in [int, float]
            or not math.isfinite(value)
            or not 0.1 <= value <= 2.0
        ):
            raise ValueError("質量は0.1〜2kg、または不明を指定してください")
    else:
        raise ValueError("未対応の項目です")


def validate_measurement_evidence(field, value, quote):
    """Numeric physical values must be traceable to explicit measured units."""
    if field == "mass_kg" and type(value) in [int, float]:
        matches = re.findall(
            r"([-+]?\d+(?:\.\d+)?)\s*(kg|キログラム|g|グラム)(?![A-Za-z])", quote, re.I
        )
        values = [
            float(n) * (0.001 if unit.lower() in ["g", "グラム"] else 1.0)
            for n, unit in matches
        ]
        if not any(abs(v - value) < 1e-8 for v in values):
            raise ValueError("質量の単位と数値を確認してください。例：800g、0.8kg")
    if field == "dimensions_m" and isinstance(value, list):
        scale = {
            "cm": 0.01,
            "mm": 0.001,
            "m": 1.0,
            "センチメートル": 0.01,
            "センチ": 0.01,
            "ミリメートル": 0.001,
            "ミリ": 0.001,
            "メートル": 1.0,
        }
        unit = "|".join(sorted(scale, key=len, reverse=True))
        tokens = re.findall(r"([-+]?\d+(?:\.\d+)?)\s*(" + unit + r")?", quote, re.I)
        if len(tokens) != 3 or not any(u for _, u in tokens):
            raise ValueError("寸法の3つの数値と単位を確認してください。例：18×16×18cm")
        common = next(u.lower() for _, u in reversed(tokens) if u)
        measured = sorted(float(n) * scale[(u or common).lower()] for n, u in tokens)
        if any(abs(a - b) > 1e-8 for a, b in zip(sorted(value), measured)):
            raise ValueError(
                "入力された寸法と解釈した数値が一致しません。前後×左右×高さと単位を指定してください"
            )


class Dialogue:
    def __init__(self, config=None):
        self.config = config or load_config()
        self.slots = {k: None for k in FIELDS}
        self.evidence = {}
        self.issues = {}
        self.turns = []
        self.last_questions = []

    def apply(self, text, patch, backend):
        text = normalize(text)
        if not isinstance(patch, dict) or not isinstance(patch.get("slots"), dict):
            raise ValueError("解釈結果の形式が正しくありません")
        if set(patch) - {"slots", "evidence", "ambiguous_fields"}:
            raise ValueError("解釈結果に未対応の項目があります")
        if set(patch["slots"]) - set(FIELDS):
            raise ValueError("許可していない運搬項目が含まれています")
        ambiguous = patch.get("ambiguous_fields", [])
        if not isinstance(ambiguous, list) or any(f not in FIELDS for f in ambiguous):
            raise ValueError("曖昧項目の形式が正しくありません")
        quotes = patch.get("evidence", {})
        if not isinstance(quotes, dict):
            raise ValueError("根拠の形式が正しくありません")
        changed = {}
        for field, value in patch["slots"].items():
            if value is None:
                continue
            raw_quote = quotes.get(field)
            quote = normalize(raw_quote) if isinstance(raw_quote, str) else ""
            if not quote or quote not in text:
                self.slots[field] = None
                self.issues[field] = (
                    "明示された情報として確認できませんでした。もう一度指定してください"
                )
                continue
            try:
                validate_value(field, value, self.config)
                validate_measurement_evidence(field, value, quote)
                if field in ["start", "goal"]:
                    candidates = station_candidates(quote, self.config)
                    if len(candidates) > 1:
                        raise ValueError(
                            "その呼び方は複数地点に対応します。B（組立場）かC（検査場）か指定してください"
                        )
                    # A route phrase may be quoted in full. Extractor must quote
                    # the actual station name for independent alias validation.
                    if candidates != [value]:
                        raise ValueError(
                            "地点名を地図と対応づけられませんでした。A/B/Cで指定してください"
                        )
            except ValueError as exc:
                self.slots[field] = None
                self.issues[field] = str(exc)
                continue
            self.slots[field] = copy.deepcopy(value)
            self.evidence[field] = quote
            self.issues.pop(field, None)
            changed[field] = copy.deepcopy(value)
        for field in patch.get("ambiguous_fields", []):
            if field not in FIELDS:
                raise ValueError("未対応の曖昧項目です")
            quote = quotes.get(field)
            if field not in REQUIRED and not (
                isinstance(quote, str) and normalize(quote) and normalize(quote) in text
            ):
                # Optional physics must not create questions merely because the
                # model does not know it. Only explicitly supplied conflicts do.
                continue
            self.slots[field] = None
            self.issues[field] = "複数の解釈があるため、希望を確認してください"
        self.turns.append(dict(user=text, backend=backend, updates=changed))
        self.last_questions = self.questions()
        return self.last_questions

    def problems(self):
        problems = dict(self.issues)
        s = self.slots
        if s["start"] is not None and s["start"] == s["goal"]:
            problems["goal"] = "出発地と目的地は異なる地点を指定してください"
        if s["shape"] in ["cylinder", "other"] and s["box_approximation"] != "yes":
            problems["box_approximation"] = (
                "現在は箱モデルです。箱の外接形状として近似して試しますか？「箱として試す」または形状の変更を指定してください"
            )
        if s["secured"] == "yes":
            problems["secured"] = (
                "この版は未固定の荷物を扱います。未固定として試す場合は「固定なしで試す」、それ以外は中止してください"
            )
        dims = s["dimensions_m"]
        if isinstance(dims, list):
            if s["shape"] == "tall_box" and dims[2] < 0.8 * max(dims[:2]):
                problems["shape"] = (
                    "背が高い形状と寸法が食い違っています。形状または寸法を訂正してください"
                )
            if s["shape"] == "flat_box" and dims[2] > 1.2 * max(dims[:2]):
                problems["shape"] = (
                    "平たい形状と寸法が食い違っています。形状または寸法を訂正してください"
                )
        return problems

    def questions(self):
        s = self.slots
        problems = self.problems()
        missing = {f for f in REQUIRED if s[f] is None} | set(problems)
        result = []

        def add(fields, text):
            needed = [f for f in fields if f in missing]
            if needed:
                result.append(
                    dict(
                        fields=needed,
                        text=" ".join(problems[f] for f in needed if f in problems)
                        + " "
                        + text,
                    )
                )

        route_missing = [f for f in ["start", "goal"] if f in missing]
        if route_missing:
            label = (
                "出発地と目的地"
                if len(route_missing) == 2
                else ("出発地" if route_missing == ["start"] else "目的地")
            )
            add(
                route_missing,
                label + "を教えてください。A＝部品置き場、B＝組立場、C＝検査場です。",
            )
        add(
            ["cargo_description"],
            "何を運びますか？ 例：工具箱、精密機器。不明でも大丈夫です。",
        )
        add(
            ["handling"],
            "運搬方針は「慎重」「標準」「時間優先」のどれですか？「おまかせ」は慎重にします。",
        )
        add(
            ["shape"],
            "形状を教えてください。例：普通の箱、背の高い箱、平たい箱、円筒形、不明。",
        )
        add(
            ["dimensions_m"],
            "前後×左右×高さはどのくらいですか？ 例：18×16×18cm。「標準サイズ」「不明」も使えます。",
        )
        detail = [f for f in ["fragility", "slippery", "secured"] if f in missing]
        if detail:
            labels = {
                "fragility": "壊れやすさ（壊れやすい／丈夫／不明）",
                "slippery": "滑りやすさ（滑りやすい／滑りにくい／不明）",
                "secured": "荷台への固定（固定なし／固定あり／不明）",
            }
            add(
                detail,
                "、".join(labels[f] for f in detail)
                + "を教えてください。分からない項目は「不明」で大丈夫です。",
            )
        add(["box_approximation"], "円筒形・複雑な形は箱としての近似に対応しています。")
        add(["mass_kg"], "質量を0.1〜2kg、または不明で指定してください。")
        if not result:
            return []
        fields = list(dict.fromkeys(f for q in result for f in q["fields"]))
        # Ask all remaining essential items together after the initial sentence.
        # Absent optional measurements/properties never enter this list.
        return [
            dict(
                fields=fields,
                text="不足・確認が必要な項目だけ教えてください。 "
                + " ".join(q["text"].strip() for q in result),
            )
        ]

    @property
    def ready(self):
        return not self.questions()

    def context(self):
        return dict(
            slots=copy.deepcopy(self.slots),
            questions=self.questions(),
            stations=self.config["stations"],
        )

    def compile(self, scenario="standard"):
        if not self.ready:
            raise ValueError("確認が必要な項目が残っています")
        for field, value in self.slots.items():
            if value is not None:
                validate_value(field, value, self.config)
        if scenario not in self.config["simulation_scenarios"]:
            raise ValueError("未対応のシミュレーション条件")
        s = copy.deepcopy(self.slots)
        notes = []
        requested = s["handling"]
        effective = requested
        unreported = [
            f
            for f in ["dimensions_m", "mass_kg", "fragility", "slippery", "secured"]
            if s[f] is None
        ]
        # Unreported values remain distinguishable from user-declared unknowns.
        # They are not claimed as physical measurements and cause no questions.
        for field in unreported:
            s[field] = "standard" if field == "dimensions_m" else "unknown"
        if unreported:
            notes.append(
                "未指定の物理情報は追加質問せず未知のまま扱い、必要なシミュレーション値には仮設定を使用"
            )
        if requested == "unknown":
            effective = "careful"
            notes.append("運搬方針を任されたため慎重設定")
        if self.slots["fragility"] in ["fragile", "unknown"]:
            effective = "careful"
            notes.append("壊れやすい、または壊れやすさが不明なため慎重設定")
        if (
            s["shape"] in ["tall_box", "unknown"]
            or self.slots["dimensions_m"] == "unknown"
        ):
            effective = "careful"
            notes.append("背の高い形状、または形状・寸法が不明なため慎重設定")
        if s["slippery"] == "yes":
            effective = "careful"
            notes.append("滑りやすいという申告があるため慎重な速度・加速度上限を使用")
        shape = s["shape"] if s["shape"] in self.config["shape_defaults_m"] else "box"
        dims = s["dimensions_m"]
        if not isinstance(dims, list):
            dims = copy.deepcopy(self.config["shape_defaults_m"][shape])
            notes.append("寸法はシミュレーション用の仮設定を使用（実測値ではない）")
        physical = copy.deepcopy(self.config["simulation_scenarios"][scenario])
        mass = s["mass_kg"]
        if type(mass) in [int, float]:
            physical["mass_kg"] = mass
        else:
            notes.append("質量・重心はシミュレーション用の仮設定を使用")
        if (
            not all(
                lo <= v <= hi
                for v, lo, hi in zip(dims, [0.15, 0.15, 0.10], [0.22, 0.18, 0.24])
            )
            or not 0.45 <= physical["mass_kg"] <= 1.1
        ):
            effective = "careful"
            notes.append("代表的な比較条件から外れる寸法・質量のため慎重設定")
        physical.update(id="mission_cargo", size_m=dims)
        if s["shape"] in ["cylinder", "other"]:
            notes.append("実際の形状を箱の外接形状で近似")
        if s["secured"] == "unknown":
            notes.append("固定状態は不明。シミュレーションは未固定で試行")
        notes.append(
            "摩擦は独立したシミュレーション条件。荷物名やLLMから推定した値ではない"
        )
        return dict(
            schema=1,
            route=[s["start"], s["goal"]],
            request=s,
            requested_profile=requested,
            effective_profile=effective,
            controller_profile=copy.deepcopy(self.config["profiles"][effective]),
            initial_cautious=s["slippery"] == "yes",
            physical_simulation=physical,
            scenario=scenario,
            notes=notes,
            unreported_fields=unreported,
        )

    def save(self, path):
        path.write_text(
            json.dumps(
                dict(
                    schema=1,
                    slots=self.slots,
                    evidence=self.evidence,
                    issues=self.issues,
                    turns=self.turns,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def restore(cls, path, config=None):
        data = json.loads(path.read_text(encoding="utf-8"))
        d = cls(config)
        if data.get("schema") != 1 or set(data.get("slots", {})) != set(FIELDS):
            raise ValueError("会話ファイルの形式が違います")
        for field, value in data["slots"].items():
            if value is not None:
                validate_value(field, value, d.config)
        d.slots = data["slots"]
        d.evidence = data.get("evidence", {})
        d.issues = data.get("issues", {})
        d.turns = data.get("turns", [])
        if set(d.issues) - set(FIELDS):
            raise ValueError("未対応の確認項目")
        d.last_questions = d.questions()
        return d
