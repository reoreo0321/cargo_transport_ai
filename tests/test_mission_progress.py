import contextlib
import io
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mission_dialogue import Dialogue
from mission_language_backend import RuleBackend, BackendError
from run_language_mission import extract_with_progress, run_dialogue


class ProgressTests(unittest.TestCase):
    def test_waiting_is_visible_and_result_preserved(self):
        expected = {"slots": {}}

        class Slow:
            def extract(self, *args):
                time.sleep(0.04)
                return expected

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = extract_with_progress(Slow(), "箱", {}, interval=0.005)
        self.assertIs(result, expected)
        self.assertIn("秒経過", output.getvalue())
        self.assertIn("解釈完了", output.getvalue())

    def test_backend_error_is_not_reported_as_completion(self):
        class Broken:
            def extract(self, *args):
                raise BackendError("test connection error")

        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(BackendError):
            extract_with_progress(Broken(), "箱", {}, interval=0.005)
        self.assertNotIn("解釈完了", output.getvalue())

    def test_cancel_during_inference_saves_session(self):
        class Local:
            name = "ollama"

        with tempfile.TemporaryDirectory() as folder:
            session = Path(folder) / "session.json"
            with contextlib.redirect_stdout(io.StringIO()), patch(
                "run_language_mission.extract_with_progress",
                side_effect=KeyboardInterrupt,
            ):
                state = run_dialogue(
                    Dialogue(), Local(), ["箱を運んで"], session, interactive=True
                )
            self.assertEqual(state, "cancelled")
            self.assertTrue(session.exists())

    def test_failed_correction_does_not_execute_old_ready_mission(self):
        class Broken:
            name = "test_backend"

            def extract(self, *args):
                raise BackendError("test connection error")

        d = Dialogue()
        text = "AからCへ工具箱を標準で運んで"
        d.apply(text, RuleBackend().extract(text, d.context()), "rules")
        self.assertTrue(d.ready)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(
            io.StringIO()
        ), patch("builtins.input", return_value="/quit"):
            state = run_dialogue(
                d,
                Broken(),
                ["目的地をBへ変更"],
                Path(folder) / "session.json",
                interactive=True,
            )
        self.assertEqual(state, "cancelled")
        self.assertEqual(d.slots["goal"], "C")


if __name__ == "__main__":
    unittest.main()
