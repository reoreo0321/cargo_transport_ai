import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mission_dialogue import Dialogue
from mission_language_backend import (
    empty_patch,
    BackendError,
    response_schema,
    RuleBackend,
)
from mission_ollama_backend import OllamaBackend, local_schema, PRIMARY_FIELDS
from run_language_mission import make_backend


class OllamaBackendTests(unittest.TestCase):
    def response(self, data):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self, *args):
                return json.dumps(data).encode()

        return Response()

    def test_local_schema_transport_never_sends_openai_key(self):
        received = []
        p = dict(
            slots=dict(handling="careful"),
            evidence=dict(handling="慎重"),
            ambiguous_fields=[],
        )

        def transport(request, timeout):
            received.append(request)
            return self.response(
                dict(
                    done=True,
                    done_reason="stop",
                    model="qwen3:4b",
                    eval_count=12,
                    message=dict(content=json.dumps(p)),
                )
            )

        with patch.dict(
            "os.environ", {"OPENAI_API_KEY": "secret-test-key"}, clear=True
        ):
            b = OllamaBackend(model="qwen3:4b", transport=transport)
            result = b.extract("慎重", Dialogue().context())
        self.assertEqual(result, p)
        payload = json.loads(received[0].data)
        self.assertEqual(payload["format"], local_schema(Dialogue().config["stations"]))
        self.assertFalse(payload["think"])
        self.assertFalse(payload["stream"])
        self.assertTrue(payload["options"]["use_mmap"])
        self.assertEqual(
            payload["format"]["properties"]["slots"]["properties"]["goal"]["enum"],
            ["A", "B", "C", None],
        )
        self.assertEqual(
            payload["format"]["properties"]["slots"]["required"], PRIMARY_FIELDS
        )
        self.assertNotIn(
            "slippery", payload["format"]["properties"]["slots"]["properties"]
        )
        self.assertEqual(received[0].full_url, "http://cargo-ollama:11434/api/chat")
        self.assertNotIn("secret-test-key", received[0].data.decode())
        self.assertIsNone(received[0].get_header("Authorization"))
        self.assertEqual(b.last_metadata["eval_count"], 12)

    def test_local_backend_needs_no_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsInstance(make_backend("ollama"), OllamaBackend)

    def test_remote_and_cloud_endpoints_are_rejected(self):
        for url in [
            "https://api.example.com",
            "http://example.com:11434",
            "http://user:secret@localhost:11434",
            "http://localhost:11434/path",
        ]:
            with self.assertRaises(BackendError):
                OllamaBackend(base_url=url)
        with self.assertRaises(BackendError):
            OllamaBackend(model="some-model:cloud")

    def test_truncated_and_non_json_outputs_are_not_accepted(self):
        for data in [
            dict(done=False),
            dict(done=True, done_reason="length", message=dict(content="{}")),
            dict(done=True, done_reason="stop", message=dict(content="not json")),
            dict(done=True, done_reason="stop", message=dict(content="[]")),
        ]:
            b = OllamaBackend(transport=lambda *a, **kw: self.response(data))
            with self.assertRaises(BackendError):
                b.extract("箱", Dialogue().context())

    def test_http_error_is_sanitized(self):
        def fail(*args, **kwargs):
            raise urllib.error.HTTPError(
                "http://localhost", 404, "secret server message", {}, None
            )

        with self.assertRaises(BackendError) as exc:
            OllamaBackend(transport=fail).extract("箱", Dialogue().context())
        self.assertIn("404", str(exc.exception))
        self.assertNotIn("secret server message", str(exc.exception))

    def test_unload_request_releases_model_before_motion(self):
        received = []

        def transport(request, timeout):
            received.append(request)
            return self.response(dict(done=True))

        self.assertTrue(OllamaBackend(transport=transport).release())
        self.assertEqual(json.loads(received[0].data)["keep_alive"], 0)
        self.assertTrue(received[0].full_url.endswith("/api/generate"))

    def test_explicit_measurements_use_existing_literal_parser(self):
        p = dict(
            slots={f: None for f in PRIMARY_FIELDS},
            evidence={f: None for f in PRIMARY_FIELDS},
            ambiguous_fields=[],
        )
        b = OllamaBackend(
            transport=lambda *a, **kw: self.response(
                dict(done=True, done_reason="stop", message=dict(content=json.dumps(p)))
            )
        )
        result = b.extract("18×16×18cm、重さ800g", Dialogue().context())
        self.assertEqual(result["slots"]["dimensions_m"], [0.18, 0.16, 0.18])
        self.assertEqual(result["slots"]["mass_kg"], 0.8)
        self.assertNotIn("slippery", result["slots"])
        self.assertNotIn("fragility", result["slots"])

    def test_llm_cannot_invent_physical_properties_outside_its_schema(self):
        p = dict(
            slots=dict(mass_kg=1.0), evidence=dict(mass_kg="箱"), ambiguous_fields=[]
        )
        b = OllamaBackend(
            transport=lambda *a, **kw: self.response(
                dict(done=True, done_reason="stop", message=dict(content=json.dumps(p)))
            )
        )
        with self.assertRaises(BackendError):
            b.extract("箱", Dialogue().context())

    def test_full_state_answer_retains_previously_grounded_fields(self):
        d = Dialogue()
        text = "AからCへ工具箱を標準で運んで"
        d.apply(text, RuleBackend().extract(text, d.context()), "rules")
        p = dict(
            slots=dict(
                start="A",
                goal="C",
                cargo_description="工具箱",
                handling="careful",
                shape="box",
            ),
            evidence=dict(
                start="倉庫",
                goal="検査場",
                cargo_description="工具箱",
                handling="慎重",
                shape="工具箱",
            ),
            ambiguous_fields=[],
        )
        b = OllamaBackend(
            transport=lambda *a, **kw: self.response(
                dict(done=True, done_reason="stop", message=dict(content=json.dumps(p)))
            )
        )
        text = "方針は慎重に変えて"
        result = b.extract(text, d.context())
        d.apply(text, result, "ollama")
        self.assertTrue(d.ready)
        self.assertEqual(d.slots["goal"], "C")
        self.assertEqual(d.slots["handling"], "careful")
        self.assertIn(
            "goal:unchanged_state_retained", b.last_metadata["normalizations"]
        )

    def test_paraphrased_evidence_is_grounded_in_actual_input(self):
        text = "倉庫にある試験管ラックを検品するところへ届けてほしい。背が高い形で、丁寧にお願い。"
        p = dict(
            slots=dict(
                start="A",
                goal="C",
                cargo_description="試験管ラック",
                handling="careful",
                shape="tall_box",
            ),
            evidence=dict(
                start="倉庫",
                goal="検査場",
                cargo_description="試験管ラック",
                handling="丁寧",
                shape="背の高い",
            ),
            ambiguous_fields=[],
        )
        d = Dialogue()
        b = OllamaBackend(
            transport=lambda *a, **kw: self.response(
                dict(done=True, done_reason="stop", message=dict(content=json.dumps(p)))
            )
        )
        result = b.extract(text, d.context())
        d.apply(text, result, "ollama")
        self.assertTrue(d.ready)
        self.assertEqual(result["evidence"]["goal"], "検品するところ")

    def test_unreported_unknown_shape_still_requires_question(self):
        text = "AからCへ顕微鏡を慎重に届けて"
        p = dict(
            slots=dict(
                start="A",
                goal="C",
                cargo_description="顕微鏡",
                handling="careful",
                shape="unknown",
            ),
            evidence=dict(
                start="A",
                goal="C",
                cargo_description="顕微鏡",
                handling="慎重",
                shape="顕微鏡",
            ),
            ambiguous_fields=[],
        )
        d = Dialogue()
        b = OllamaBackend(
            transport=lambda *a, **kw: self.response(
                dict(done=True, done_reason="stop", message=dict(content=json.dumps(p)))
            )
        )
        result = b.extract(text, d.context())
        d.apply(text, result, "ollama")
        self.assertEqual(d.questions()[0]["fields"], ["shape"])

    def test_stale_full_state_cannot_ignore_explicit_destination_change(self):
        d = Dialogue()
        text = "AからCへ工具箱を標準で運んで"
        d.apply(text, RuleBackend().extract(text, d.context()), "rules")
        p = dict(
            slots=dict(
                start="A",
                goal="C",
                cargo_description="工具箱",
                handling="standard",
                shape="box",
            ),
            evidence=dict(
                start="A",
                goal="C",
                cargo_description="工具箱",
                handling="標準",
                shape="工具箱",
            ),
            ambiguous_fields=[],
        )
        b = OllamaBackend(
            transport=lambda *a, **kw: self.response(
                dict(done=True, done_reason="stop", message=dict(content=json.dumps(p)))
            )
        )
        text = "目的地はBに変更"
        result = b.extract(text, d.context())
        d.apply(text, result, "ollama")
        self.assertFalse(d.ready)
        self.assertEqual(d.questions()[0]["fields"], ["goal"])
        self.assertEqual(d.slots["start"], "A")

    def test_model_cannot_default_missing_origin_or_handling(self):
        text = "検査するとこまで工具箱を運んで"
        p = dict(
            slots=dict(
                start="A",
                goal="C",
                cargo_description="工具箱",
                handling="standard",
                shape=None,
            ),
            evidence=dict(
                start="倉庫",
                goal="検査場",
                cargo_description="工具箱",
                handling="標準",
                shape=None,
            ),
            ambiguous_fields=[],
        )
        d = Dialogue()
        b = OllamaBackend(
            transport=lambda *a, **kw: self.response(
                dict(done=True, done_reason="stop", message=dict(content=json.dumps(p)))
            )
        )
        d.apply(text, b.extract(text, d.context()), "ollama")
        self.assertEqual(d.questions()[0]["fields"], ["start", "handling"])

    def test_shape_reply_does_not_invent_cargo_or_replace_known_route(self):
        d = Dialogue()
        text = "AからCまで顕微鏡を慎重に届けて"
        d.apply(text, RuleBackend().extract(text, d.context()), "rules")
        p = dict(
            slots=dict(
                start="A",
                goal="C",
                cargo_description="不明な荷物",
                handling="careful",
                shape="tall_box",
            ),
            evidence=dict(
                start="倉庫",
                goal="検査場",
                cargo_description="不明な荷物",
                handling="丁寧",
                shape="背の高い",
            ),
            ambiguous_fields=["shape"],
        )
        b = OllamaBackend(
            transport=lambda *a, **kw: self.response(
                dict(done=True, done_reason="stop", message=dict(content=json.dumps(p)))
            )
        )
        text = "背の高い箱として扱って"
        d.apply(text, b.extract(text, d.context()), "ollama")
        self.assertTrue(d.ready)
        self.assertEqual(d.slots["cargo_description"], "顕微鏡")
        self.assertEqual(d.slots["shape"], "tall_box")

    def test_reported_careful_request_is_not_ambiguous(self):
        text = "部ヒ部品置き場から検査するところまで背の高い精密機器をできるだけ慎重に"
        p = dict(
            slots=dict(
                start="A",
                goal="C",
                cargo_description="精密機器",
                handling="careful",
                shape="tall_box",
            ),
            evidence=dict(
                start="部品置き場",
                goal="検査するところ",
                cargo_description="精密機器",
                handling="慎重",
                shape="背の高い",
            ),
            ambiguous_fields=["handling"],
        )
        d = Dialogue()
        b = OllamaBackend(
            transport=lambda *a, **kw: self.response(
                dict(done=True, done_reason="stop", message=dict(content=json.dumps(p)))
            )
        )
        d.apply(text, b.extract(text, d.context()), "ollama")
        self.assertTrue(d.ready)

    def test_competing_priorities_still_require_clarification(self):
        text = "AからCへ箱を慎重に、でも急いで運んで"
        p = dict(
            slots=dict(
                start="A",
                goal="C",
                cargo_description="箱",
                handling="careful",
                shape="box",
            ),
            evidence=dict(
                start="A", goal="C", cargo_description="箱", handling="慎重", shape="箱"
            ),
            ambiguous_fields=["handling"],
        )
        d = Dialogue()
        b = OllamaBackend(
            transport=lambda *a, **kw: self.response(
                dict(done=True, done_reason="stop", message=dict(content=json.dumps(p)))
            )
        )
        d.apply(text, b.extract(text, d.context()), "ollama")
        self.assertEqual(d.questions()[0]["fields"], ["handling"])

    def test_recorded_eight_conversations_still_pass_postprocessing(self):
        from check_live_language import evaluate_case

        root = Path(__file__).resolve().parent.parent
        cases = json.loads(
            (root / "examples/missions/live_language_cases.json").read_text(
                encoding="utf-8"
            )
        )
        recorded = json.loads(
            (root / "results/local_llm_dialogue_v1/validation.json").read_text(
                encoding="utf-8"
            )
        )
        for case, original in zip(cases, recorded["cases"]):
            with self.subTest(case=case["id"]):
                responses = iter(
                    turn["inference_metadata"]["raw_primary_patch"]
                    for turn in original["turns"]
                )

                def transport(*a, **kw):
                    return self.response(
                        dict(
                            done=True,
                            done_reason="stop",
                            message=dict(content=json.dumps(next(responses))),
                        )
                    )

                result = evaluate_case(case, OllamaBackend(transport=transport))
                self.assertTrue(result["passed"], result["failures"])


if __name__ == "__main__":
    unittest.main()
