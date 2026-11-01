import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cargo_transport import field
from cargo_transport.controller import COMMON, ShortHistoryController
from cargo_transport.paths import ROOT
from mission_dialogue import Dialogue
from mission_language_backend import RuleBackend
from mission_execution import MissionFollower


class Zero:
    def predict(self, x):
        return 0.0

    def margin(self, x):
        return 0.1


def feed(f, t, slip=0.0, relative=0.0):
    f.observe(
        dict(
            base_stamp_s=t,
            cargo_stamp_s=t,
            base_v_forward_mps=f.v,
            yaw_rate_rps=f.w,
            previous_cmd_v_mps=f.v,
            previous_cmd_yaw_rps=f.w,
            cargo_dx_m=slip,
            cargo_dy_m=0.0,
            cargo_displacement_m=slip,
            cargo_vx_rel_mps=relative,
            cargo_vy_rel_mps=0.0,
            cargo_roll_rel_rad=0.0,
            cargo_pitch_rel_rad=0.0,
        )
    )
    return f.step(0.0, 0.0, 0.0, 0.05)


def settle(f):
    feed(f, 0.6, relative=0.0015)
    for i in range(16):
        result = feed(f, 0.65 + i * 0.05)
    return result


class GuardTests(unittest.TestCase):
    def make(self, model=None, careful=False):
        c = field.load_config(ROOT / "configs/factory_field.json")["controller"]
        c.update(COMMON, cruise_speed_mps=0.65, max_acceleration_mps2=0.30)
        route = dict(points=[[0.0, 0.0], [10.0, 0.0]], length_m=10.0)
        if careful:
            d = Dialogue()
            text = "AからBへ工具箱を慎重に運んで"
            d.apply(text, RuleBackend().extract(text, d.context()), "rules")
            f = MissionFollower(route, c, d.compile(), model or Zero())
        else:
            f = ShortHistoryController(route, c, "short_model", model or Zero())
        for i in range(12):
            feed(f, i * 0.05)
        f.v = 0.3
        f.motion_start = -2.0
        return f

    def test_prediction_rejects_risky_strong_restart(self):
        class Risk(Zero):
            def predict(self, x):
                return 6.0 if x[42] > 0.15 else 0.0

        f = self.make(Risk())
        _, _, d = settle(f)
        self.assertTrue(d["model_used"])
        self.assertEqual(d["action_accel_mps2"], 0.15)

    def test_careful_profile_caps_all_restart_candidates(self):
        f = self.make(careful=True)
        _, _, d = settle(f)
        self.assertTrue(d["model_used"])
        for item in json.loads(d["candidate_scores_json"]):
            self.assertLessEqual(item["action"][0], 0.45)
            self.assertLessEqual(item["action"][2], 0.15)

    def test_no_fit_holds_and_does_not_claim_safety(self):
        class Risk(Zero):
            def predict(self, x):
                return 6.0

        f = self.make(Risk())
        _, _, d = settle(f)
        self.assertTrue(d["no_feasible_candidate"])
        self.assertEqual(d["restart_phase"], "PREDICTED_HOLD")

    def test_measured_slip_and_budget_stop_override_prediction(self):
        f = self.make()
        settle(f)
        previous = f.v
        v, _, d = feed(f, 1.45, relative=0.0015)
        self.assertLessEqual(v, previous)
        self.assertFalse(d["model_used"])
        _, _, d = feed(f, 1.5, slip=0.006)
        self.assertEqual(d["restart_phase"], "STOP_LATCHED")
        self.assertFalse(d["model_used"])


if __name__ == "__main__":
    unittest.main()
