#!/usr/bin/env python3
"""Frozen three-arm comparison: fixed motion, measured reaction, prediction plus reaction."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from cargo_transport import field, physics, simulation
from cargo_transport.controller import (
    COMMON,
    ObservationController,
    PathFollower,
    ShortHistoryController,
    clip_change,
    stopping_speed,
)
from cargo_transport.io import dependencies, ensure_stepper, numeric_rows, sha
from cargo_transport.paths import ROOT, MODEL
from cargo_transport.predictor import ShortHistoryPredictor

METHODS = ["fixed", "reactive", "learned"]
SEEDS = [22301, 22401, 22501]


class FixedController(ObservationController):
    """No cargo-dependent cruise/ramp adjustment; common measured budget stop remains."""

    def step(self, x, y, yaw, dt):
        guide = copy.copy(self)
        _, _, d = PathFollower.step(guide, x, y, yaw, dt)
        self.progress, self.goal_latched = guide.progress, guide.goal_latched
        c = self.settings
        now = self.times[-1]
        row = self.rows[-1]
        curve = d["curvature_command_per_m"]
        desired = min(
            c["cruise_speed_mps"],
            math.sqrt(c["max_lateral_acceleration_mps2"] / max(abs(curve), 0.001)),
            c["max_yaw_rate_rps"] / max(abs(curve), 0.001),
            0.8 * d["goal_distance_m"],
            stopping_speed(min(d["goal_distance_m"], d["remaining_m"])),
        )
        if self.goal_latched:
            desired = 0.0
        current = 1000 * row["cargo_displacement_m"]
        ready = self.vector((self.v, 0.0, 0.15, 2.5)) is not None
        if current > 5.0 and not self.stop_reason:
            self.stop_reason = "measured_budget_exceeded"
        if now >= self.next_decision - 1e-9 or self.goal_latched:
            if ready and self.motion_start is None:
                self.motion_start = now
            cap = min(
                c["max_acceleration_mps2"],
                (
                    0.15
                    if self.motion_start is None or now - self.motion_start < 2.0 - 1e-9
                    else 0.30
                ),
            )
            target = desired if ready else 0.0
            yaw_target = max(
                -c["max_yaw_rate_rps"], min(c["max_yaw_rate_rps"], curve * target)
            )
            self.action = (target, yaw_target, cap if target > self.v else 0.15, 2.5)
            self.diagnostic = dict(
                model_used=False,
                observation_ready=ready,
                decision_stamp_s=now,
                acceleration_cap_mps2=cap,
                restart_phase="FIXED_MOTION",
            )
            self.next_decision = now + 0.25
        action = self.action
        if self.stop_reason:
            action = (0.0, 0.0, 0.15, 0.6)
            self.experiment_done = True
        self.v = clip_change(self.v, action[0], action[2], dt)
        self.w = clip_change(self.w, action[1], action[3], dt)
        return (
            self.v,
            self.w,
            {
                **d,
                **self.diagnostic,
                "target_v_mps": action[0],
                "target_yaw_rps": action[1],
                "action_accel_mps2": action[2],
                "action_yaw_accel_rps2": action[3],
                "stop_reason": self.stop_reason,
                "budget_mm": 5.0,
                "comparison_method": "fixed",
                "risk_state": (
                    "STOP_LATCHED"
                    if self.stop_reason
                    else ("RUNNING" if ready else "WAIT_OBSERVATION")
                ),
            },
        )


def make(route, settings, method, model):
    if method == "fixed":
        return FixedController(route, settings)
    return ShortHistoryController(
        route, settings, "short_model" if method == "learned" else "zero", model
    )


def sources():
    return dependencies() | {"scripts/compare_controllers.py": sha(Path(__file__))}


def prepare(output):
    ensure_stepper()
    source = ROOT / "results/short_reproducibility_v1/same_seed_1/summary.json"
    old = json.loads(source.read_text())["protocol"]
    conditions = old["conditions"]
    model = json.loads(MODEL.read_text())
    used = {
        tuple(s)
        for s in model["train_physical_signatures"]
        + model["calibration_physical_signatures"]
    }
    assert all(
        (c["mu"], c["mass_kg"], *c["size_m"], c["com_z_fraction"]) not in used
        for c in conditions
    )
    protocol = dict(
        conditions=conditions,
        routes=[["A", "B"], ["A", "C"]],
        seeds=SEEDS,
        methods=METHODS,
        common_settings=dict(COMMON, cruise_speed_mps=0.65, max_acceleration_mps2=0.30),
        observation_s=0.05,
        scheduled_decision_s=0.25,
        max_sim_s=120.0,
        budget_mm=5.0,
        model_sha256=sha(MODEL),
        stepper_sha256=sha(ROOT / "logs/runtime/gazebo_stepper"),
        sources=sources(),
        fixed="Same .45 s observation warmup; open-loop .15 acceleration for first 2 s then .30. No cargo-magnitude-based cruise/acceleration or warning-edge redecision. Common route geometry, goal braking and measured >5 mm budget stop retained.",
        reactive="Existing zero-mean policy: same measured guards, settling/restart candidates, short-variant calibration and ranking with predicted mean zero. No learned mean used.",
        learned="Existing fixed short-model policy with measured guards; weights, calibration and selection unchanged.",
        design="54 fresh acquisitions: three methods x three previously evaluated cargo combinations x two routes x three paired seeds. Protocol frozen before acquisition. Not a new holdout-cargo test; no tuning from outcomes.",
        limitation="Safety/budget stop is common even to fixed motion. This is not literal absence of all feedback: path tracking is present in every method. Time is stepped simulation time. Three seeds are not independent real-world trials.",
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    for name in protocol["sources"]:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    print("Frozen 54 new paired acquisitions.", flush=True)


def validate(output):
    p = json.loads((output / "protocol.json").read_text())
    assert p["sources"] == sources() and p["model_sha256"] == sha(MODEL)
    assert p["stepper_sha256"] == sha(ROOT / "logs/runtime/gazebo_stepper")
    return p


def jobs(p):
    return [
        dict(
            id=f"{c['id']}_{''.join(route)}_{method}_S{seed}",
            condition=c,
            route=route,
            method=method,
            seed=seed,
        )
        for c in p["conditions"]
        for route in p["routes"]
        for seed in p["seeds"]
        for method in p["methods"]
    ]


def collect(args):
    p = validate(args.output)
    model = ShortHistoryPredictor()
    for i, job in enumerate(jobs(p)):
        if (i // 3 + i % 3) % 3 != args.shard:
            continue
        folder = args.output / job["id"]
        if folder.exists():
            raise FileExistsError(folder)
        folder.mkdir()
        c = field.load_config(ROOT / "configs/factory_field.json")
        c["controller"].update(p["common_settings"])
        route = field.plan(c, *job["route"])
        field.write_world(
            ROOT / "worlds/cargo_contact_fixed.world", folder / "field.world", c, route
        )
        physics.apply_cargo(folder / "field.world", job["condition"])
        (folder / "manifest.json").write_text(
            json.dumps(
                dict(
                    job=job,
                    config=c,
                    protocol_sha256=sha(args.output / "protocol.json"),
                    world_sha256=sha(folder / "field.world"),
                ),
                indent=2,
            )
        )
        run = SimpleNamespace(
            route=job["route"],
            seed=job["seed"],
            gui=False,
            domain_id=94 + args.shard,
            max_sim_s=p["max_sim_s"],
            abort_slip_mm=40.0,
            abort_tilt_deg=15,
        )
        print("START " + job["id"], flush=True)
        simulation.execute(
            run,
            c,
            route,
            folder,
            follower_factory=lambda route, settings: make(
                route, settings, job["method"], model
            ),
            controller_label=job["method"],
        )
        summary = json.loads((folder / "summary.json").read_text())
        rows = numeric_rows(folder)
        if summary["status"] == "profile_completed" and rows[-1].get("stop_reason"):
            summary["status"] = "risk_stopped"
        summary["stop_reason"] = rows[-1].get("stop_reason", "")
        (folder / "summary.json").write_text(json.dumps(summary, indent=2))
        assert summary["status"] not in [
            "error",
            "invalid_physics",
            "interrupted",
        ], folder


def stats(values):
    return (
        dict(
            n=len(values),
            mean=sum(values) / len(values),
            min=min(values),
            max=max(values),
        )
        if values
        else None
    )


def analyze(args):
    p = validate(args.output)
    model = ShortHistoryPredictor()
    records = []
    paired = {}
    commands = 0
    for job in jobs(p):
        folder = args.output / job["id"]
        m = json.loads((folder / "manifest.json").read_text())
        s = json.loads((folder / "summary.json").read_text())
        assert (
            m["job"] == job
            and m["protocol_sha256"] == sha(args.output / "protocol.json")
            and m["world_sha256"] == sha(folder / "field.world")
        )
        key = (job["condition"]["id"], tuple(job["route"]), job["seed"])
        signature = (m["world_sha256"], json.dumps(m["config"], sort_keys=True))
        assert key not in paired or paired[key] == signature
        paired[key] = signature
        assert s["status"] not in ["error", "invalid_physics", "interrupted"]
        c = m["config"]
        route = field.plan(c, *job["route"])
        f = make(route, c["controller"], job["method"], model)
        calls = changes = 0
        previous = None
        for row in numeric_rows(folder):
            assert abs(row["cmd_v_mps"]) <= 0.65 + 1e-6
            if previous is not None:
                dt = row["base_stamp_s"] - previous["base_stamp_s"]
                dv = row["cmd_v_mps"] - previous["cmd_v_mps"]
                assert (
                    abs(dt - 0.05) < 1e-6
                    and -0.15 * dt - 1e-6 <= dv <= 0.30 * dt + 1e-6
                )
            previous = row
            if not isinstance(row.get("cargo_displacement_m"), float):
                continue
            f.observe(row)
            v, w, d = f.step(row["x_m"], row["y_m"], row["yaw_rad"], 0.05)
            assert (
                abs(v - row["cmd_v_mps"]) < 1e-7 and abs(w - row["cmd_yaw_rps"]) < 1e-7
            )
            commands += 1
            if (
                not d.get("model_used")
                or abs(row["base_stamp_s"] - d["decision_stamp_s"]) > 1e-6
            ):
                continue
            calls += 1
            scores = json.loads(d["candidate_scores_json"])
            current = 1000 * row["cargo_displacement_m"]
            zero = [
                q for q in scores if q["permitted"] and current + q["margin_mm"] <= 5.0
            ]
            counter = (
                max(
                    zero,
                    key=lambda q: clip_change(
                        row["previous_cmd_v_mps"], q["action"][0], q["action"][2], 0.25
                    ),
                )
                if zero
                else scores[0]
            )
            chosen = [
                d[k]
                for k in [
                    "target_v_mps",
                    "target_yaw_rps",
                    "action_accel_mps2",
                    "action_yaw_accel_rps2",
                ]
            ]
            changes += any(abs(a - b) > 1e-8 for a, b in zip(chosen, counter["action"]))
        records.append(
            dict(
                trial=job["id"],
                condition=job["condition"]["id"],
                route="".join(job["route"]),
                seed=job["seed"],
                method=job["method"],
                status=s["status"],
                arrival_s=(
                    s["sim_duration_s"] if s["status"] == "goal_reached" else None
                ),
                stop_time_s=(
                    s["sim_duration_s"] if s["status"] != "goal_reached" else None
                ),
                max_slip_mm=s["max_cargo_displacement_mm"],
                exceeded_5mm=s["max_cargo_displacement_mm"] > 5.0,
                model_decisions=calls,
                mean_changed_action_vs_zero=changes,
                contacts=len(s["obstacle_contact_pairs"]),
                lcp_errors=s["lcp_error_count"],
                samples_sha256=sha(folder / "samples.csv"),
            )
        )
    successes = {
        m: {
            (r["condition"], r["route"], r["seed"])
            for r in records
            if r["method"] == m and r["arrival_s"] is not None
        }
        for m in METHODS
    }
    common = set.intersection(*successes.values())
    groups = []
    cases = []
    pairs = []
    for method in METHODS:
        rr = [r for r in records if r["method"] == method]
        groups.append(
            dict(
                method=method,
                runs=len(rr),
                safe_arrivals=sum(
                    r["arrival_s"] is not None and not r["exceeded_5mm"] for r in rr
                ),
                arrived=sum(r["arrival_s"] is not None for r in rr),
                exceeded_5mm=sum(r["exceeded_5mm"] for r in rr),
                arrival_s_success_only=stats(
                    [r["arrival_s"] for r in rr if r["arrival_s"] is not None]
                ),
                arrival_s_common_successes=stats(
                    [
                        r["arrival_s"]
                        for r in rr
                        if (r["condition"], r["route"], r["seed"]) in common
                    ]
                ),
                max_slip_mm=stats([r["max_slip_mm"] for r in rr]),
                model_decisions=sum(r["model_decisions"] for r in rr),
                mean_changed_action_vs_zero=sum(
                    r["mean_changed_action_vs_zero"] for r in rr
                ),
            )
        )
    for cargo in p["conditions"]:
        for route in p["routes"]:
            for method in METHODS:
                rr = [
                    r
                    for r in records
                    if r["condition"] == cargo["id"]
                    and r["route"] == "".join(route)
                    and r["method"] == method
                ]
                cases.append(
                    dict(
                        condition=cargo["id"],
                        route="".join(route),
                        method=method,
                        runs=len(rr),
                        safe_arrivals=sum(
                            r["arrival_s"] is not None and not r["exceeded_5mm"]
                            for r in rr
                        ),
                        arrival_s_success_only=stats(
                            [r["arrival_s"] for r in rr if r["arrival_s"] is not None]
                        ),
                        max_slip_mm=stats([r["max_slip_mm"] for r in rr]),
                    )
                )
    for key in paired:
        rr = {
            r["method"]: r
            for r in records
            if (r["condition"], tuple(r["route"]), r["seed"])
            == (key[0], tuple(key[1]), key[2])
        }
        for reference in ["fixed", "reactive"]:
            a, b = rr["learned"], rr[reference]
            pairs.append(
                dict(
                    condition=key[0],
                    route="".join(key[1]),
                    seed=key[2],
                    reference=reference,
                    learned_status=a["status"],
                    reference_status=b["status"],
                    time_saved_s=(
                        b["arrival_s"] - a["arrival_s"]
                        if a["arrival_s"] is not None and b["arrival_s"] is not None
                        else None
                    ),
                    max_slip_reduction_mm=b["max_slip_mm"] - a["max_slip_mm"],
                )
            )
    result = dict(
        protocol=p,
        trials=records,
        groups=groups,
        cases=cases,
        pairs=pairs,
        audit=dict(
            new_runs=len(records),
            commands_replayed=commands,
            paired_world_config_seed=True,
            common_success_pairs=len(common),
            contacts=sum(r["contacts"] for r in records),
            lcp_errors=sum(r["lcp_errors"] for r in records),
        ),
        caveat="Means over successful arrivals exclude stopped runs. Compare paired/common-success times and success counts together; do not treat stop time as delivery time.",
    )
    args.report.mkdir(parents=True, exist_ok=False)
    (args.report / "summary.json").write_text(json.dumps(result, indent=2))
    for name, rr in [("trials", records), ("pairs", pairs)]:
        with (args.report / (name + ".csv")).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rr[0]))
            writer.writeheader()
            writer.writerows(rr)
    print(json.dumps(dict(groups=groups, audit=result["audit"]), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "logs/three_way_v1")
    ap.add_argument("--report", type=Path, default=ROOT / "results/three_way_v1")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--shard", type=int, choices=range(3), default=0)
    a = ap.parse_args()
    if a.prepare and a.analyze:
        ap.error("Choose one operation")
    if a.prepare:
        prepare(a.output)
    elif a.analyze:
        analyze(a)
    else:
        collect(a)
