#!/usr/bin/env python3
"""Small real-API dialogue evaluation. Never starts ROS or Gazebo."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from mission_dialogue import Dialogue, ROOT
from mission_language_backend import OpenAIBackend, BackendError
from run_language_mission import load_environment, make_backend


def evaluate_case(case, backend):
    dialogue = Dialogue()
    turns = []
    failures = []
    for index, step in enumerate(case["turns"], 1):
        started = time.monotonic()
        patch = backend.extract(step["text"], dialogue.context())
        dialogue.apply(step["text"], patch, backend.name)
        asked = sorted({f for q in dialogue.questions() for f in q["fields"]})
        checks = []
        for field, expected in step.get("slots", {}).items():
            if dialogue.slots[field] != expected:
                checks.append(
                    f"{field}: expected {expected!r}, got {dialogue.slots[field]!r}"
                )
        if asked != sorted(step["questions"]):
            checks.append(
                f'questions: expected {sorted(step["questions"])!r}, got {asked!r}'
            )
        for field in step.get("must_be_unset", []):
            if dialogue.slots[field] is not None:
                checks.append(f"{field}: unreported property was filled")
        if "cargo_contains" in step and step["cargo_contains"] not in (
            dialogue.slots["cargo_description"] or ""
        ):
            checks.append("cargo_description: requested cargo was not retained")
        if dialogue.ready:
            mission = dialogue.compile()
            if "profile" in step and mission["effective_profile"] != step["profile"]:
                checks.append("effective_profile: " + mission["effective_profile"])
        turns.append(
            dict(
                text=step["text"],
                elapsed_s=round(time.monotonic() - started, 3),
                extracted=patch,
                slots=dict(dialogue.slots),
                questions=dialogue.questions(),
                failures=checks,
                inference_metadata=dict(getattr(backend, "last_metadata", {})),
            )
        )
        failures.extend(f"turn {index}: {item}" for item in checks)
    return dict(id=case["id"], passed=not failures, failures=failures, turns=turns)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "examples/missions/live_language_cases.json",
    )
    ap.add_argument("--env-file", type=Path, default=ROOT / ".env")
    ap.add_argument("--model")
    ap.add_argument(
        "--case", action="append", dest="case_ids", help="Run selected case IDs only"
    )
    ap.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--ollama-url")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    load_environment(args.env_file)
    try:
        backend = make_backend(args.backend, args.model, args.ollama_url)
    except BackendError as exc:
        print(str(exc))
        return 2
    cases = json.loads(args.cases.read_text(encoding="utf-8-sig"))
    if args.case_ids:
        ids = {case["id"] for case in cases}
        if set(args.case_ids) - ids:
            ap.error("Unknown case ID")
        cases = [case for case in cases if case["id"] in args.case_ids]
    folder = args.output or ROOT / "logs/live_language" / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    folder.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        ROOT / "scripts/mission_dialogue.py",
        ROOT / "scripts/mission_language_backend.py",
        ROOT / "scripts/mission_ollama_backend.py",
        ROOT / "scripts/run_language_mission.py",
        ROOT / "configs/mission_language.json",
        args.cases,
    ]
    report = dict(
        status="running",
        backend=args.backend,
        model=backend.model,
        started_at=datetime.now(timezone.utc).isoformat(),
        expected_cases=len(cases),
        planned_api_requests=sum(len(case["turns"]) for case in cases),
        cases=[],
        source_sha256={
            str(
                p.relative_to(ROOT) if p.is_relative_to(ROOT) else p.name
            ): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    )

    def save():
        report["passed_cases"] = sum(c["passed"] for c in report["cases"])
        (folder / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    save()
    print(
        f'Real {args.backend} evaluation / {backend.model} / {report["planned_api_requests"]} requests',
        flush=True,
    )
    print("Report: " + str(folder / "report.json"), flush=True)
    for case in cases:
        print("Checking " + case["id"] + " ...", flush=True)
        try:
            result = evaluate_case(case, backend)
        except (BackendError, ValueError) as exc:
            # Stop on transport/schema failures; never silently fall back to rules.
            report.update(status="interrupted", failed_case=case["id"], error=str(exc))
            save()
            print("Evaluation stopped: " + str(exc), flush=True)
            return 2
        report["cases"].append(result)
        save()
        print(("PASS " if result["passed"] else "FAIL ") + case["id"], flush=True)
        for failure in result["failures"]:
            print("  " + failure, flush=True)
    report["status"] = "completed"
    save()
    print(f'{report["passed_cases"]}/{len(cases)} cases passed', flush=True)
    return 0 if all(c["passed"] for c in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
