#!/usr/bin/env python3
"""Run and audit the retained CLI through planning, transport and saved results."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import mission_execution as execution
from cargo_transport.io import numeric_rows, sha
from cargo_transport.paths import ROOT, MODEL
from cargo_transport.predictor import ShortHistoryPredictor

CASES = {
    "standard_ab": dict(
        request="AからBへ工具箱を標準で運んで",
        answers=[],
        route=["A", "B"],
        profile="standard",
        scenario="standard",
    ),
    "clarification_ac": dict(
        request="検査するところまで工具箱を運んで",
        answers=["部品置き場から、慎重"],
        route=["A", "C"],
        profile="careful",
        scenario="standard",
    ),
    "predictive_ab": dict(
        request="部品置き場から組立場まで工具箱を時間優先で運んで",
        answers=[],
        route=["A", "B"],
        profile="time_priority",
        scenario="slippery",
    ),
    "live_ac": dict(
        request="倉庫から検査するところまで工具箱を急いで運んで",
        answers=[],
        route=["A", "C"],
        profile="time_priority",
        scenario="slippery",
    ),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["rules", "ollama"], default="rules")
    ap.add_argument("--case", choices=list(CASES), action="append", dest="cases")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--domain-id", type=int, default=94)
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.domain_id <= 100:
        ap.error("domain-id must be 1..100")
    cases = args.cases or (
        ["live_ac"]
        if args.backend == "ollama"
        else ["standard_ab", "clarification_ac", "predictive_ab"]
    )
    output = args.output or ROOT / "logs/verification" / datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(
        cases=cases,
        backend=args.backend,
        gui=args.gui,
        sources=execution.dependencies(),
        model_sha256=sha(MODEL),
    )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    records = []
    for name in cases:
        case = CASES[name]
        answers = output / (name + "_answers.json")
        answers.write_text(
            json.dumps(case["answers"], ensure_ascii=False), encoding="utf-8"
        )
        folder = output / name
        command = [
            sys.executable,
            str(ROOT / "scripts/run_language_mission.py"),
            "--backend",
            args.backend,
            "--request",
            case["request"],
            "--answers",
            str(answers),
            "--output",
            str(folder),
            "--sim-cargo",
            case["scenario"],
            "--domain-id",
            str(args.domain_id),
        ]
        if args.backend == "ollama":
            command += ["--model", "qwen2.5:3b"]
        if args.gui:
            command.append("--gui")
        print("START " + name, flush=True)
        with (output / (name + ".console.txt")).open("w") as console:
            result = subprocess.run(
                command, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT, timeout=900
            )
        run = folder / "run"
        summary = json.loads((run / "summary.json").read_text())
        manifest = json.loads((run / "manifest.json").read_text())
        mission = manifest["mission"]
        dialogue = json.loads((folder / "session.json").read_text())
        assert (
            mission["route"] == case["route"]
            and mission["effective_profile"] == case["profile"]
        )
        assert (
            manifest["controller"] == "short_model"
            and manifest["model_sha256"] == protocol["model_sha256"]
        )
        assert protocol["sources"] == execution.dependencies()
        route = json.loads((run / "path.json").read_text())
        replay = execution.MissionFollower(
            route, manifest["config"]["controller"], mission, ShortHistoryPredictor()
        )
        previous = None
        commands = 0
        for row in numeric_rows(run):
            profile = mission["controller_profile"]
            assert abs(row["cmd_v_mps"]) <= profile["speed_mps"] + 1e-6
            if previous:
                dt = row["base_stamp_s"] - previous["base_stamp_s"]
                dv = row["cmd_v_mps"] - previous["cmd_v_mps"]
                assert (
                    abs(dt - 0.05) < 1e-6
                    and -0.15 * dt - 1e-6
                    <= dv
                    <= profile["acceleration_mps2"] * dt + 1e-6
                )
            previous = row
            if not isinstance(row.get("cargo_displacement_m"), float):
                continue
            replay.observe(row)
            v, w, d = replay.step(row["x_m"], row["y_m"], row["yaw_rad"], 0.05)
            assert (
                abs(v - row["cmd_v_mps"]) < 1e-7 and abs(w - row["cmd_yaw_rps"]) < 1e-7
            )
            assert d["model_used"] == (row["model_used"] == "True")
            commands += 1
        records.append(
            dict(
                case=name,
                backend=args.backend,
                status=summary["status"],
                returncode=result.returncode,
                safe_arrival=summary["status"] == "goal_reached"
                and summary["max_cargo_displacement_mm"] <= 5.0,
                time_s=summary["sim_duration_s"],
                max_slip_mm=summary["max_cargo_displacement_mm"],
                model_decisions=summary["model_decisions"],
                commands_replayed=commands,
                turns=len(dialogue["turns"]),
                contacts=len(summary["obstacle_contact_pairs"]),
                lcp_errors=summary["lcp_error_count"],
                samples_sha256=sha(run / "samples.csv"),
            )
        )
        (output / "report.json").write_text(
            json.dumps(dict(protocol=protocol, trials=records), indent=2)
        )
        print(json.dumps(records[-1]), flush=True)
    return (
        0
        if all(
            r["safe_arrival"] and not r["contacts"] and not r["lcp_errors"]
            for r in records
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
