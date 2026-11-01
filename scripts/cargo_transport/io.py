"""File fingerprints, CSV observations and the fixed-step helper build."""

import csv
import hashlib
from pathlib import Path
import shlex
import subprocess
from .paths import ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_rows(folder):
    with (folder / "samples.csv").open() as f:
        raw = list(csv.DictReader(f))
    rows = []
    for row in raw:
        converted = {}
        for k, v in row.items():
            try:
                converted[k] = float(v)
            except (ValueError, TypeError):
                converted[k] = v
        rows.append(converted)
    return rows


def ensure_stepper():
    binary = ROOT / "logs/runtime/gazebo_stepper"
    if not binary.is_file():
        binary.parent.mkdir(parents=True, exist_ok=True)
        flags = subprocess.check_output(
            ["pkg-config", "--cflags", "--libs", "gazebo"], text=True
        )
        print("Building fixed-step Gazebo helper...", flush=True)
        subprocess.run(
            [
                "g++",
                "-O2",
                str(ROOT / "tools/gazebo_stepper.cc"),
                "-o",
                str(binary),
                *shlex.split(flags),
            ],
            check=True,
        )
    return binary


def dependencies():
    paths = list((ROOT / "scripts/cargo_transport").glob("*.py"))
    paths += [
        ROOT / n
        for n in [
            "configs/factory_field.json",
            "configs/mission_language.json",
            "worlds/cargo_contact_fixed.world",
            "tools/gazebo_stepper.cc",
        ]
    ]
    paths += list((ROOT / "scripts").glob("mission*.py"))
    paths += [ROOT / "scripts/run_language_mission.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}
