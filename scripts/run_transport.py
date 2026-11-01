#!/usr/bin/env python3
"""One transport demonstration, with optional matched comparison policies."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from mission_dialogue import Dialogue, ROOT
import mission_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", nargs=2, choices=["A", "B", "C"], default=["A", "C"])
    parser.add_argument(
        "--profile",
        choices=["careful", "standard", "time_priority"],
        default="time_priority",
    )
    parser.add_argument(
        "--sim-cargo", choices=["standard", "slippery", "stable"], default="slippery"
    )
    parser.add_argument(
        "--method", choices=["short_model", "zero", "gentle"], default="short_model"
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--domain-id", type=int, default=93)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.route[0] == args.route[1] or not 1 <= args.domain_id <= 100:
        parser.error("Invalid route/domain")
    dialogue = Dialogue()
    dialogue.slots.update(
        start=args.route[0],
        goal=args.route[1],
        cargo_description="工具箱",
        shape="box",
        handling=args.profile,
    )
    mission = dialogue.compile(args.sim_cargo)
    folder = args.output or ROOT / "logs/transport" / datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")
    config, route = mission_execution.prepare(mission, folder, args.method)
    print("記録: " + str(folder), flush=True)
    if args.plan_only:
        return 0
    return mission_execution.execute(
        mission, folder, config, route, args.gui, args.domain_id
    )


if __name__ == "__main__":
    raise SystemExit(main())
