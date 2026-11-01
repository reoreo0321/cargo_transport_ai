"""Frozen real-trajectory replay plus meaningful control guard regressions."""

import gzip
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cargo_transport import field
from cargo_transport.controller import ShortHistoryController
from cargo_transport.predictor import ShortHistoryPredictor, feature_vector


class ControllerTests(unittest.TestCase):
    def test_all_six_frozen_cargo_route_traces_keep_identical_commands(self):
        model = ShortHistoryPredictor()
        for path in sorted((Path(__file__).parent / "fixtures").glob("*.json.gz")):
            with self.subTest(trace=path.name), gzip.open(path, "rt") as f:
                data = json.load(f)
                route = field.plan(data["config"], *data["route"])
                controller = ShortHistoryController(
                    route, data["config"]["controller"], "short_model", model
                )
                for row in data["observations"]:
                    controller.observe(row)
                    v, w, d = controller.step(
                        row["x_m"], row["y_m"], row["yaw_rad"], 0.05
                    )
                    self.assertAlmostEqual(v, row["cmd_v_mps"], places=8)
                    self.assertAlmostEqual(w, row["cmd_yaw_rps"], places=8)
                    self.assertEqual(d["restart_phase"], row["restart_phase"])
                    self.assertEqual(d["model_used"], row["model_used"] == "True")

    def test_history_never_uses_future_or_invents_missing_observations(self):
        row = dict(
            base_v_forward_mps=0.2,
            yaw_rate_rps=0,
            previous_cmd_v_mps=0.2,
            previous_cmd_yaw_rps=0,
            cargo_dx_m=0.001,
            cargo_dy_m=0,
            cargo_vx_rel_mps=0,
            cargo_vy_rel_mps=0,
            cargo_roll_rel_rad=0,
            cargo_pitch_rel_rad=0,
        )
        self.assertIsNone(feature_vector([row], [0], [0.3, 0, 0.15, 2.5]))
        vector = feature_vector(
            [row] * 11, [i * 0.05 for i in range(11)], [0.3, 0, 0.15, 2.5]
        )
        self.assertEqual(len(vector), 44)
        self.assertEqual(vector[4], 1.0)
        self.assertEqual(vector[-4:], [0.3, 0, 0.15, 2.5])

    def test_true_physics_and_cargo_id_do_not_enter_features(self):
        row = dict(
            base_v_forward_mps=0.2,
            yaw_rate_rps=0,
            previous_cmd_v_mps=0.2,
            previous_cmd_yaw_rps=0,
            cargo_dx_m=0,
            cargo_dy_m=0,
            cargo_vx_rel_mps=0,
            cargo_vy_rel_mps=0,
            cargo_roll_rel_rad=0,
            cargo_pitch_rel_rad=0,
        )
        a = feature_vector(
            [row] * 11, [i * 0.05 for i in range(11)], [0.3, 0, 0.15, 2.5]
        )
        row.update(mu=0.0001, mass_kg=100, cargo_id="arbitrary", future_slip=999)
        b = feature_vector(
            [row] * 11, [i * 0.05 for i in range(11)], [0.3, 0, 0.15, 2.5]
        )
        self.assertEqual(a, b)

    def test_invalid_model_features_are_rejected(self):
        model = ShortHistoryPredictor()
        for x in [[0.0] * 43, [float("nan")] * 44]:
            with self.assertRaises(ValueError):
                model.predict(x)


if __name__ == "__main__":
    unittest.main()
