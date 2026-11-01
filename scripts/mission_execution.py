#!/usr/bin/env python3
"""Validated language mission -> map planning -> frozen short-history controller."""
import csv
import json
from types import SimpleNamespace
from cargo_transport import field, physics, simulation
from cargo_transport.controller import ShortHistoryController, COMMON
from cargo_transport.predictor import ShortHistoryPredictor
from cargo_transport.paths import ROOT, MODEL
from cargo_transport.io import sha, ensure_stepper, dependencies as runtime_dependencies

CONTROLLER = "short_model"
MODEL_VARIANT = "short"
SEED = 22301


def dependencies():
    return runtime_dependencies()


def prepare(mission, folder, method=CONTROLLER):
    if method not in [CONTROLLER, "zero", "gentle"]:
        raise ValueError("Unsupported control method")
    folder.mkdir(parents=True, exist_ok=False)
    c = field.load_config(ROOT / "configs/factory_field.json")
    profile = mission["controller_profile"]
    c["controller"].update(
        COMMON,
        cruise_speed_mps=profile["speed_mps"],
        max_acceleration_mps2=profile["acceleration_mps2"],
    )
    route = field.plan(c, *mission["route"])
    field.write_world(
        ROOT / "worlds/cargo_contact_fixed.world", folder / "field.world", c, route
    )
    physics.apply_cargo(folder / "field.world", mission["physical_simulation"])
    field.write_svg(folder / "planned_route.svg", c, [route])
    (folder / "mission.json").write_text(
        json.dumps(mission, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (folder / "path.json").write_text(json.dumps(route, indent=2))
    (folder / "manifest.json").write_text(
        json.dumps(
            dict(
                mission=mission,
                config=c,
                seed=SEED,
                controller=method,
                model_variant=MODEL_VARIANT,
                history_s=0.45,
                prediction_horizon_s=0.25,
                budget_mm=5.0,
                prediction_scope="Restart after observed slip settles; initial launch and final braking remain reactive/geometric.",
                world_sha256=sha(folder / "field.world"),
                dependencies=dependencies(),
                model_path=str(MODEL.relative_to(ROOT)),
                model_sha256=sha(MODEL),
                calibration_source="short variant embedded in model",
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return c, route


class MissionFollower(ShortHistoryController):
    def __init__(
        self, route, settings, mission, predictor, display=False, method=CONTROLLER
    ):
        profile = mission["controller_profile"]
        settings = dict(
            settings,
            cruise_speed_mps=min(settings["cruise_speed_mps"], profile["speed_mps"]),
        )
        super().__init__(route, settings, method, predictor)
        self.profile_acc = profile["acceleration_mps2"]
        self.profile = mission["effective_profile"]
        self.language_caution = mission["initial_cautious"]
        self.display = display
        self.last_print = -100.0
        self.last_mode = None

    def acceleration_cap(self, now):
        return min(self.profile_acc, super().acceleration_cap(now))

    def step(self, *state):
        v, w, d = super().step(*state)
        now = self.times[-1]
        mode = (
            d["restart_phase"]
            if d["restart_phase"] != "NO_RESTART"
            else d["hold_phase"]
        )
        d.update(
            control_mode=mode,
            language_profile=self.profile,
            language_caution=self.language_caution,
            model_variant=MODEL_VARIANT,
        )
        if self.display and (now - self.last_print >= 1.0 or mode != self.last_mode):
            pred = d.get("predicted_increment_mm") if d["model_used"] else None
            prediction = f"{pred:.3f}" if isinstance(pred, (int, float)) else "--"
            print(
                f'CTRL t={now:.2f}s | {mode} | v={v:.3f} target={d["target_v_mps"]:.3f} | '
                f'slip={1000*self.rows[-1]["cargo_displacement_m"]:.3f}mm forecast={prediction}mm | model={d["model_used"]}',
                flush=True,
            )
            self.last_print = now
            self.last_mode = mode
        return v, w, d


def execute(mission, folder, c, route, gui=False, domain_id=95, display=True):
    manifest = json.loads((folder / "manifest.json").read_text())
    if (
        manifest["controller"] not in [CONTROLLER, "zero", "gentle"]
        or manifest["model_variant"] != MODEL_VARIANT
        or manifest["model_sha256"] != sha(MODEL)
    ):
        raise ValueError(
            "準備時と短履歴モデルが異なります。運搬条件を作り直してください。"
        )
    ensure_stepper()
    model = ShortHistoryPredictor()
    factory = lambda route, settings: MissionFollower(
        route, settings, mission, model, display=display, method=manifest["controller"]
    )
    args = SimpleNamespace(
        route=mission["route"],
        seed=manifest["seed"],
        gui=gui,
        domain_id=domain_id,
        max_sim_s=120.0,
        abort_slip_mm=40.0,
        abort_tilt_deg=15,
    )
    print("制御: " + manifest["controller"] + "（0.45秒履歴・0.25秒予測）", flush=True)
    code = simulation.execute(
        args,
        c,
        route,
        folder,
        follower_factory=factory,
        controller_label="language_mission/" + manifest["controller"],
    )
    summary = json.loads((folder / "summary.json").read_text())
    rows = []
    if (folder / "samples.csv").exists():
        with (folder / "samples.csv").open() as f:
            rows = list(csv.DictReader(f))
    summary["learned_adaptation"] = any(r.get("model_used") == "True" for r in rows)
    summary["stop_reason"] = rows[-1].get("stop_reason", "") if rows else ""
    if summary["status"] == "profile_completed" and summary["stop_reason"]:
        summary["status"] = "risk_stopped"
    summary.update(
        mission_route=mission["route"],
        mission_profile=mission["effective_profile"],
        model_variant=MODEL_VARIANT,
        model_sha256=manifest["model_sha256"],
        model_decisions=sum(
            r.get("model_used") == "True"
            and abs(float(r["base_stamp_s"]) - float(r["decision_stamp_s"])) < 1e-6
            for r in rows
        ),
    )
    (folder / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("配送結果: " + summary["status"], flush=True)
    return code
