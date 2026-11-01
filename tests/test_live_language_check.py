import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from check_live_language import evaluate_case
from mission_language_backend import RuleBackend, BackendError


class LiveLanguageCheckTests(unittest.TestCase):
    def cases(self):
        return json.loads(
            (
                Path(__file__).resolve().parent.parent
                / "examples/missions/live_language_cases.json"
            ).read_text(encoding="utf-8")
        )

    def test_evaluator_accepts_complete_sentence_with_test_backend(self):
        result = evaluate_case(self.cases()[0], RuleBackend())
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["turns"][0]["questions"], [])

    def test_evaluator_detects_wrong_destination(self):
        case = copy.deepcopy(self.cases()[0])
        case["turns"][0]["slots"]["goal"] = "B"
        result = evaluate_case(case, RuleBackend())
        self.assertFalse(result["passed"])
        self.assertTrue(any("goal" in failure for failure in result["failures"]))

    def test_evaluator_checks_followup_and_retained_answers(self):
        case = next(c for c in self.cases() if c["id"] == "missing_only_and_retention")
        result = evaluate_case(case, RuleBackend())
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(
            result["turns"][0]["questions"][0]["fields"], ["start", "handling"]
        )
        self.assertEqual(result["turns"][1]["slots"]["goal"], "C")

    def test_backend_failure_is_not_reported_as_success(self):
        class Broken:
            name = "test_backend"

            def extract(self, *args):
                raise BackendError("Connection failed")

        with self.assertRaises(BackendError):
            evaluate_case(self.cases()[0], Broken())


if __name__ == "__main__":
    unittest.main()
