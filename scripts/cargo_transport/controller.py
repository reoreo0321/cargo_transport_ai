"""Observed slip guards and short-history selection of restart actions.

PathFollower handles path geometry. ObservationController owns the causal buffer
and startup/budget guards. HoldController responds to measured slip. The public
ShortHistoryController selects hold or restart after settling. No physical truth
or cargo identity enters the controller. Constants match the frozen candidate.
"""

import bisect
import copy
import json
import math
from . import field
from .predictor import feature_vector

COMMON = dict(
    max_deceleration_mps2=0.15,
    max_yaw_rate_rps=1.0,
    max_yaw_acceleration_rps2=2.5,
    max_lateral_acceleration_mps2=0.3,
)


def clip_change(current, target, limit, dt):
    return current + max(-limit * dt, min(limit * dt, target - current))


def stopping_speed(distance, deceleration=0.15, hold_s=0.25, reserve_m=0.05):
    available = max(0.0, distance - reserve_m)
    return (
        math.sqrt((deceleration * hold_s) ** 2 + 2 * deceleration * available)
        - deceleration * hold_s
    )


class PathFollower:

    def __init__(self, route, settings):
        self.points = route["points"]
        self.lengths = field.arc_lengths(self.points)
        self.settings = settings
        self.progress = 0.0
        self.v = 0.0
        self.w = 0.0
        self.goal_latched = False

    def nearest(self, x, y):
        i0 = max(0, bisect.bisect_right(self.lengths, max(0, self.progress - 0.2)) - 1)
        i1 = min(
            len(self.points) - 1, bisect.bisect_right(self.lengths, self.progress + 1.2)
        )
        best = (float("inf"), self.progress)
        for i in range(i0, i1):
            (a, b) = self.points[i : i + 2]
            (vx, vy) = (b[0] - a[0], b[1] - a[1])
            denom = vx * vx + vy * vy
            t = (
                max(0, min(1, ((x - a[0]) * vx + (y - a[1]) * vy) / denom))
                if denom > 1e-12
                else 0
            )
            (px, py) = (a[0] + t * vx, a[1] + t * vy)
            distance = math.hypot(x - px, y - py)
            s = self.lengths[i] + t * math.sqrt(denom)
            if distance < best[0]:
                best = (distance, s)
        self.progress = max(self.progress, best[1])
        return best[0]

    def step(self, x, y, yaw, dt):
        c = self.settings
        error = self.nearest(x, y)
        remaining = max(0, self.lengths[-1] - self.progress)
        goal_distance = math.dist((x, y), self.points[-1])
        if goal_distance <= c["goal_tolerance_m"] and remaining < 0.25:
            self.goal_latched = True
        lookahead = c["lookahead_m"] + 0.2 * self.v
        target = field.interpolate(self.points, self.lengths, self.progress + lookahead)
        (dx, dy) = (target[0] - x, target[1] - y)
        lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
        heading_error = math.atan2(lateral, math.cos(yaw) * dx + math.sin(yaw) * dy)
        curvature = 2 * lateral / max(dx * dx + dy * dy, 0.02)
        indices = range(
            max(1, bisect.bisect_left(self.lengths, self.progress)),
            min(
                len(self.points) - 1,
                bisect.bisect_right(self.lengths, self.progress + lookahead),
            ),
        )
        future_curve = max(
            [abs(curvature)] + [abs(field.curvature(self.points, i)) for i in indices]
        )
        desired = min(
            c["cruise_speed_mps"],
            math.sqrt(c["max_lateral_acceleration_mps2"] / max(future_curve, 0.001)),
            c["max_yaw_rate_rps"] / max(future_curve, 0.001),
            0.8 * goal_distance,
        )
        if abs(heading_error) > 1.1:
            desired = min(desired, 0.035)
        if self.goal_latched:
            desired = 0.0
        limit = (
            c["max_acceleration_mps2"]
            if desired > self.v
            else c["max_deceleration_mps2"]
        )
        self.v = clip_change(self.v, desired, limit, dt)
        desired_w = max(
            -c["max_yaw_rate_rps"], min(c["max_yaw_rate_rps"], self.v * curvature)
        )
        self.w = clip_change(self.w, desired_w, c["max_yaw_acceleration_rps2"], dt)
        if self.goal_latched and self.v < 0.001:
            (self.v, self.w) = (0.0, 0.0)
        return (
            self.v,
            self.w,
            {
                "path_error_m": error,
                "progress_m": self.progress,
                "remaining_m": remaining,
                "goal_distance_m": goal_distance,
                "curvature_command_per_m": curvature,
                "goal_latched": self.goal_latched,
            },
        )


class ObservationController(PathFollower):

    def __init__(self, route, settings):
        super().__init__(route, settings)
        self.rows = []
        self.times = []
        self.next_decision = -1
        self.action = (0.0, 0.0, 0.15, 2.5)
        self.diagnostic = {}
        self.stop_reason = ""
        self.experiment_done = False
        self.motion_start = None

    def vector(self, action):
        return feature_vector(self.rows, self.times, action)

    def observe(self, row):
        stamp = max(row["base_stamp_s"], row["cargo_stamp_s"])
        if not self.times or stamp > self.times[-1]:
            self.rows.append(row)
            self.times.append(stamp)
        while self.times and stamp - self.times[0] > 1.8:
            self.times.pop(0)
            self.rows.pop(0)

    def acceleration_cap(self, now):
        if self.motion_start is None or now - self.motion_start < 2.0 - 1e-09:
            return 0.15
        window = [r for (r, t) in zip(self.rows, self.times) if now - t <= 0.5 + 1e-09]
        quiet = (
            len(window) >= 6
            and all(
                (
                    isinstance(r.get(k), (int, float)) and math.isfinite(r[k])
                    for r in window
                    for k in ["cargo_vx_rel_mps", "cargo_vy_rel_mps"]
                )
            )
            and (
                max((r["cargo_displacement_m"] for r in window))
                - min((r["cargo_displacement_m"] for r in window))
                <= 0.00025
            )
            and (
                max(
                    (
                        math.hypot(r["cargo_vx_rel_mps"], r["cargo_vy_rel_mps"])
                        for r in window
                    )
                )
                <= 0.001
            )
        )
        return 0.3 if quiet else 0.15

    def step(self, x, y, yaw, dt):
        guide = copy.copy(self)
        (_, _, d) = PathFollower.step(guide, x, y, yaw, dt)
        (self.progress, self.goal_latched) = (guide.progress, guide.goal_latched)
        c = self.settings
        curve = d["curvature_command_per_m"]
        now = self.times[-1]
        desired = min(
            c["cruise_speed_mps"],
            math.sqrt(c["max_lateral_acceleration_mps2"] / max(abs(curve), 0.001)),
            c["max_yaw_rate_rps"] / max(abs(curve), 0.001),
            0.8 * d["goal_distance_m"],
            stopping_speed(min(d["goal_distance_m"], d["remaining_m"])),
        )
        if self.goal_latched:
            desired = 0.0
        yaw_for = lambda v: max(
            -c["max_yaw_rate_rps"], min(c["max_yaw_rate_rps"], curve * v)
        )
        row = self.rows[-1]
        current = 1000 * row["cargo_displacement_m"]
        rv = [row.get("cargo_vx_rel_mps"), row.get("cargo_vy_rel_mps")]
        rel = (
            math.hypot(*rv)
            if all((isinstance(v, (int, float)) and math.isfinite(v) for v in rv))
            else None
        )
        hold = (min(self.v, desired), yaw_for(min(self.v, desired)), 0.15, 2.5)
        ready = self.vector(hold) is not None and rel is not None
        if current > 5.0 and (not self.stop_reason):
            self.stop_reason = "measured_budget_exceeded"
        if now >= self.next_decision - 1e-09 or self.goal_latched:
            if ready and self.motion_start is None:
                self.motion_start = now
            cap = self.acceleration_cap(now)
            if not ready:
                self.action = (0.0, 0.0, 0.15, 2.5)
            else:
                limit = 0.65
                acc = 0.3
                caution = rel >= 0.001 or current >= 2.5
                limit = 0.45 if caution else 0.65
                acc = 0.15 if caution else cap
                target = min(desired, limit)
                self.action = (
                    target,
                    yaw_for(target),
                    acc if target > self.v else 0.15,
                    2.5,
                )
            self.diagnostic = dict(
                model_used=False,
                observation_ready=ready,
                decision_stamp_s=now,
                decision_current_displacement_mm=current,
                acceleration_cap_mps2=cap,
            )
            self.next_decision = now + 0.25
        action = self.action
        if self.stop_reason:
            action = (0.0, 0.0, 0.15, 0.6)
            self.experiment_done = True
        (target, w, acc, yacc) = action
        self.v = clip_change(self.v, target, acc, dt)
        self.w = clip_change(self.w, w, yacc, dt)
        state = (
            "STOP_LATCHED"
            if self.stop_reason
            else (
                "WAIT_OBSERVATION"
                if not self.diagnostic.get("observation_ready")
                else "RUNNING"
            )
        )
        return (
            self.v,
            self.w,
            {
                **d,
                **self.diagnostic,
                "target_v_mps": target,
                "target_yaw_rps": w,
                "action_accel_mps2": acc,
                "action_yaw_accel_rps2": yacc,
                "relative_sliding_speed_mps": rel,
                "risk_state": state,
                "stop_reason": self.stop_reason,
                "budget_mm": 5.0,
            },
        )


class HoldController(ObservationController):

    def __init__(self, route, settings):
        super().__init__(route, settings)
        self.policy = "reactive_hold"
        self.hold_cap = None
        self.first_hold = None
        self.sliding_since = None
        self.quiet_since = None
        self.warning_previous = False

    def step(self, x, y, yaw, dt):
        now = self.times[-1]
        row = self.rows[-1]
        rel = math.hypot(row["cargo_vx_rel_mps"], row["cargo_vy_rel_mps"])
        finite = math.isfinite(rel)
        warning = finite and rel >= 0.001
        if warning and (not self.warning_previous):
            self.next_decision = min(self.next_decision, now)
        self.warning_previous = warning
        self.sliding_since = (
            (now if self.sliding_since is None else self.sliding_since)
            if warning
            else None
        )
        self.quiet_since = (
            (now if self.quiet_since is None else self.quiet_since)
            if finite and rel <= 0.0005
            else None
        )
        (old_v, old_w) = (self.v, self.w)
        (v, w, d) = super().step(x, y, yaw, dt)
        ready = d.get("observation_ready", False)
        if self.hold_cap is None and ready and warning:
            self.hold_cap = old_v
            self.first_hold = now
        phase = "NORMAL"
        if ready and (not self.stop_reason):
            if abs(d["decision_stamp_s"] - now) < 1e-06 or self.goal_latched:
                c = self.settings
                curve = d["curvature_command_per_m"]
                desired = min(
                    c["cruise_speed_mps"],
                    math.sqrt(
                        c["max_lateral_acceleration_mps2"] / max(abs(curve), 0.001)
                    ),
                    c["max_yaw_rate_rps"] / max(abs(curve), 0.001),
                    0.8 * d["goal_distance_m"],
                    stopping_speed(min(d["goal_distance_m"], d["remaining_m"])),
                )
                target = (
                    min(desired, self.hold_cap)
                    if self.hold_cap is not None
                    else desired
                )
                quiet = (
                    self.quiet_since is not None
                    and now - self.quiet_since >= 0.25 - 1e-09
                )
                if self.hold_cap is not None and (not quiet):
                    target = min(target, old_v)
                sustained = (
                    self.sliding_since is not None
                    and now - self.sliding_since >= 0.5 - 1e-09
                )
                if sustained:
                    target = min(target, max(0.0, old_v - 0.15 * 0.25))
                    self.hold_cap = min(self.hold_cap, target)
                if self.goal_latched:
                    target = 0.0
                yaw_target = max(
                    -c["max_yaw_rate_rps"], min(c["max_yaw_rate_rps"], curve * target)
                )
                acc = (
                    0.15
                    if self.hold_cap is not None or target <= old_v
                    else d["acceleration_cap_mps2"]
                )
                self.action = (target, yaw_target, acc, 2.5)
            self.v = clip_change(old_v, self.action[0], self.action[2], dt)
            self.w = clip_change(old_w, self.action[1], self.action[3], dt)
            (v, w) = (self.v, self.w)
            phase = (
                (
                    "PERSISTENT_SLIP_BRAKE"
                    if self.sliding_since is not None
                    and now - self.sliding_since >= 0.5 - 1e-09
                    else "HOLD_SPEED"
                )
                if self.hold_cap is not None
                else "NORMAL"
            )
        if self.stop_reason:
            phase = "STOP_LATCHED"
        elif not ready:
            phase = "WAIT_OBSERVATION"
        d.update(
            target_v_mps=0.0 if self.stop_reason else self.action[0],
            target_yaw_rps=0.0 if self.stop_reason else self.action[1],
            action_accel_mps2=0.15 if self.stop_reason else self.action[2],
            action_yaw_accel_rps2=0.6 if self.stop_reason else self.action[3],
            hold_policy=self.policy,
            hold_phase=phase,
            hold_speed_cap_mps=self.hold_cap,
            first_hold_stamp_s=self.first_hold,
            sliding_since_s=self.sliding_since,
            model_used=False,
        )
        return (v, w, d)


class ShortHistoryController(HoldController):

    def __init__(self, route, settings, method, predictor=None):
        if method not in ["short_model", "zero", "gentle"]:
            raise ValueError(method)
        super().__init__(route, settings)
        self.control_method = method
        self.predictor_bundle = predictor
        self.info = dict(
            model_used=False,
            restart_phase="NO_RESTART",
            candidate_scores_json="",
            no_feasible_candidate=False,
        )

    def settled(self, now):
        if self.quiet_since is None or now - self.quiet_since < 0.5 - 1e-09:
            return False
        rr = [r for (r, t) in zip(self.rows, self.times) if now - t <= 0.5 + 1e-09]
        return (
            len(rr) >= 6
            and max((r["cargo_displacement_m"] for r in rr))
            - min((r["cargo_displacement_m"] for r in rr))
            <= 0.0002
        )

    def select(self, candidates, current, old_v):
        scores = []
        for action in candidates:
            x = self.vector(action)
            modeled = self.control_method == "short_model"
            pred = self.predictor_bundle.predict(x) if modeled else 0.0
            margin = (
                self.predictor_bundle.margin(x)
                if modeled or self.control_method == "zero"
                else 0.0
            )
            permitted = self.control_method != "gentle" or action[2] <= 0.15 + 1e-09
            end = clip_change(old_v, action[0], action[2], 0.25)
            scores.append(
                dict(
                    action=action,
                    predicted_increment_mm=pred,
                    margin_mm=margin,
                    permitted=permitted,
                    feasible=permitted and current + pred + margin <= 5.0,
                    utility=end - 0.13 * pred,
                )
            )
        fits = [s for s in scores if s["feasible"]]
        return (
            max(fits, key=lambda s: s["utility"]) if fits else scores[0],
            scores,
            not fits,
        )

    def step(self, x, y, yaw, dt):
        now = self.times[-1]
        (old_v, old_w) = (self.v, self.w)
        if self.hold_cap is not None:
            self.hold_cap = old_v
        (v, w, d) = super().step(x, y, yaw, dt)
        if abs(d.get("decision_stamp_s", -1) - now) < 1e-06:
            self.info = dict(
                model_used=False,
                restart_phase="NO_RESTART",
                candidate_scores_json="",
                no_feasible_candidate=False,
            )
            current = 1000 * self.rows[-1]["cargo_displacement_m"]
            if (
                self.first_hold is not None
                and d["observation_ready"]
                and self.settled(now)
                and (current <= 4.0)
                and (not self.stop_reason)
                and (not self.goal_latched)
            ):
                c = self.settings
                curve = d["curvature_command_per_m"]
                desired = min(
                    c["cruise_speed_mps"],
                    math.sqrt(
                        c["max_lateral_acceleration_mps2"] / max(abs(curve), 0.001)
                    ),
                    c["max_yaw_rate_rps"] / max(abs(curve), 0.001),
                    0.8 * d["goal_distance_m"],
                    stopping_speed(min(d["goal_distance_m"], d["remaining_m"])),
                )
                if desired > old_v + 1e-09:
                    yaw_for = lambda speed: max(
                        -c["max_yaw_rate_rps"],
                        min(c["max_yaw_rate_rps"], curve * speed),
                    )
                    candidates = [(old_v, yaw_for(old_v), 0.15, 2.5)] + [
                        (desired, yaw_for(desired), a, 2.5)
                        for a in [0.15, 0.3]
                        if a <= d["acceleration_cap_mps2"] + 1e-09
                    ]
                    (chosen, scores, no_fit) = self.select(candidates, current, old_v)
                    self.action = tuple(chosen["action"])
                    self.v = clip_change(old_v, self.action[0], self.action[2], dt)
                    self.w = clip_change(old_w, self.action[1], self.action[3], dt)
                    (v, w) = (self.v, self.w)
                    self.info = dict(
                        model_used=self.control_method == "short_model",
                        restart_phase=(
                            "RESTART"
                            if self.action[0] > old_v + 1e-09
                            else "PREDICTED_HOLD"
                        ),
                        candidate_scores_json=json.dumps(scores),
                        no_feasible_candidate=no_fit,
                        predicted_increment_mm=chosen["predicted_increment_mm"],
                        prediction_margin_mm=chosen["margin_mm"],
                    )
        if self.stop_reason:
            self.info.update(model_used=False, restart_phase="STOP_LATCHED")
        d.update(
            self.info,
            method=self.control_method,
            target_v_mps=0.0 if self.stop_reason else self.action[0],
            target_yaw_rps=0.0 if self.stop_reason else self.action[1],
            action_accel_mps2=0.15 if self.stop_reason else self.action[2],
            action_yaw_accel_rps2=0.6 if self.stop_reason else self.action[3],
        )
        return (v, w, d)
