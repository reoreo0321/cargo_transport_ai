import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_controllers import FixedController
from cargo_transport import field
from cargo_transport.controller import COMMON
from cargo_transport.paths import ROOT


def make():
    c = field.load_config(ROOT / "configs/factory_field.json")["controller"]
    c.update(COMMON, cruise_speed_mps=0.65, max_acceleration_mps2=0.30)
    return FixedController(dict(points=[[0.0, 0.0], [20.0, 0.0]], length_m=20.0), c)


def step(controller, time, offset=0.0, sliding=0.0):
    controller.observe(
        dict(
            base_stamp_s=time,
            cargo_stamp_s=time,
            base_v_forward_mps=controller.v,
            yaw_rate_rps=controller.w,
            previous_cmd_v_mps=controller.v,
            previous_cmd_yaw_rps=controller.w,
            cargo_dx_m=offset,
            cargo_dy_m=0.0,
            cargo_displacement_m=offset,
            cargo_vx_rel_mps=sliding,
            cargo_vy_rel_mps=0.0,
            cargo_roll_rel_rad=0.0,
            cargo_pitch_rel_rad=0.0,
        )
    )
    return controller.step(0.0, 0.0, 0.0, 0.05)


class Tests(unittest.TestCase):
    def test_fixed_commands_ignore_slip_magnitude_below_shared_budget(self):
        a, b = make(), make()
        for i in range(140):
            x = step(a, i * 0.05, offset=0.0001, sliding=0.0)
            y = step(b, i * 0.05, offset=0.004, sliding=0.01)
            self.assertEqual(x[:2], y[:2])
            self.assertFalse(y[2]["model_used"])
        self.assertEqual(x[2]["acceleration_cap_mps2"], 0.30)

    def test_common_measured_budget_stop_is_retained(self):
        controller = make()
        for i in range(80):
            step(controller, i * 0.05)
        old = controller.v
        v, _, d = step(controller, 4.0, offset=0.006)
        self.assertLess(v, old)
        self.assertEqual(d["stop_reason"], "measured_budget_exceeded")
        self.assertTrue(controller.experiment_done)


if __name__ == "__main__":
    unittest.main()
