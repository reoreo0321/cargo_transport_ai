"""44 causal features and portable inference for the selected short variant."""

import bisect
import json
import math
import struct
from .paths import MODEL

STATE_KEYS = [
    "base_v_forward_mps",
    "yaw_rate_rps",
    "previous_cmd_v_mps",
    "previous_cmd_yaw_rps",
    "cargo_dx_m",
    "cargo_dy_m",
    "cargo_vx_rel_mps",
    "cargo_vy_rel_mps",
    "cargo_roll_rel_rad",
    "cargo_pitch_rel_rad",
]
LAGS = [0.0, 0.15, 0.30, 0.45]


def feature_vector(rows, times, action):
    values = []
    for lag in LAGS:
        j = bisect.bisect_right(times, times[-1] - lag + 1e-9) - 1
        if j < 0 or times[-1] - lag - times[j] > 0.065:
            return None
        try:
            values.extend(observed(rows[j]))
        except (KeyError, ValueError, TypeError):
            return None
    values.extend(action)
    return values if all(math.isfinite(v) for v in values) else None


def action_group(x):
    end = x[0] + max(-x[42] * 0.25, min(x[42] * 0.25, x[40] - x[0]))
    if abs(end - x[0]) > 0.015:
        return "ramp"
    if max(abs(x[1]), abs(x[41])) > 0.05:
        return "turn"
    return "cruise"


class ShortHistoryPredictor:
    def __init__(self, path=MODEL):
        self.model = json.loads(path.read_text())
        self.variant = self.model["variants"]["short"]
        if self.model["history_s"] != 0.45 or self.model["horizon_s"] != 0.25:
            raise ValueError("Unsupported prediction horizon")
        if self.variant["columns"] != list(range(40)) + list(range(60, 64)):
            raise ValueError("Unsupported feature schema")

    def predict(self, x):
        if len(x) != 44 or not all(math.isfinite(v) for v in x):
            raise ValueError("Invalid short-history features")
        return max(0.0, forest_predict(self.variant["trees"], x))

    def margin(self, x):
        return self.variant["calibration"]["groups"][action_group(x)]["margin_mm"]


def observed(row):
    return [
        float(row["base_v_forward_mps"]),
        float(row["yaw_rate_rps"]),
        float(row["previous_cmd_v_mps"]),
        float(row["previous_cmd_yaw_rps"]),
        1000 * float(row["cargo_dx_m"]),
        1000 * float(row["cargo_dy_m"]),
        float(row["cargo_vx_rel_mps"]),
        float(row["cargo_vy_rel_mps"]),
        float(row["cargo_roll_rel_rad"]),
        float(row["cargo_pitch_rel_rad"]),
    ]


def forest_predict(trees, x):
    x = [struct.unpack("f", struct.pack("f", v))[0] for v in x]
    result = []
    for tree in trees:
        i = 0
        while tree["left"][i] != -1:
            i = (
                tree["left"][i]
                if x[tree["feature"][i]] <= tree["threshold"][i]
                else tree["right"][i]
            )
        result.append(tree["value"][i])
    return sum(result) / len(result)
