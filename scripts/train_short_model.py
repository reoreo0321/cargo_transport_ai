#!/usr/bin/env python3
"""Reproduce the selected short-history learner from the published windows."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
from cargo_transport.paths import ROOT, MODEL
from cargo_transport.io import sha
from cargo_transport.predictor import ShortHistoryPredictor


def weights(rows):
    groups = defaultdict(set)
    counts = Counter((tuple(r["signature"]), r["episode"]) for r in rows)
    for r in rows:
        groups[tuple(r["signature"])].add(r["episode"])
    return [
        1
        / (
            len(groups)
            * len(groups[tuple(r["signature"])])
            * counts[(tuple(r["signature"]), r["episode"])]
        )
        for r in rows
    ]


def quantile(values):
    if not values:
        raise ValueError("Calibration residuals are empty")
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil((len(values) + 1) * 0.95) - 1)]


def forest_json(model):
    result = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        result.append(
            dict(
                left=tree.children_left.tolist(),
                right=tree.children_right.tolist(),
                feature=tree.feature.tolist(),
                threshold=tree.threshold.tolist(),
                value=tree.value[:, 0, 0].tolist(),
            )
        )
    return result


def main():
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data", type=Path, default=ROOT / "data/training/short_history.jsonl.gz"
    )
    ap.add_argument("--output", type=Path)
    ap.add_argument(
        "--cross-validate",
        action="store_true",
        help="Leave one physical condition out; no hyperparameter search",
    )
    args = ap.parse_args()
    output = args.output or ROOT / "logs/retraining" / datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")
    if output.exists():
        raise FileExistsError(output)
    metadata = json.loads((ROOT / "data/training/manifest.json").read_text())
    if sha(args.data) != metadata["data_sha256"]:
        raise ValueError("Training data fingerprint mismatch")
    with gzip.open(args.data, "rt") as f:
        rows = [json.loads(line) for line in f]
    train = [r for r in rows if r["split"] == "train"]
    cal = [r for r in rows if r["split"] == "calibration"]
    signatures = {tuple(r["signature"]) for r in train}
    cal_signatures = {tuple(r["signature"]) for r in cal}
    assert (
        signatures.isdisjoint(cal_signatures) and len(train) == 6096 and len(cal) == 246
    )
    assert all(
        len(r["x"]) == 44 and all(math.isfinite(v) for v in r["x"]) for r in rows
    )

    def fit(part):
        model = ExtraTreesRegressor(
            n_estimators=64, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=2
        )
        model.fit(
            np.asarray([r["x"] for r in part]),
            np.asarray([r["y"] for r in part]),
            sample_weight=weights(part),
        )
        return model

    folds = []
    if args.cross_validate:
        for held in sorted(signatures):
            tr = [r for r in train if tuple(r["signature"]) != held]
            val = [r for r in train if tuple(r["signature"]) == held]
            prediction = fit(tr).predict(np.asarray([r["x"] for r in val]))
            folds.append(
                dict(
                    held_signature=list(held),
                    windows=len(val),
                    mae_mm=float(np.mean(np.abs(prediction - [r["y"] for r in val]))),
                    zero_mae_mm=float(np.mean([r["y"] for r in val])),
                )
            )
    model = fit(train)
    pred = model.predict(np.asarray([r["x"] for r in cal]))
    residuals = [max(0.0, r["y"] - float(p)) for r, p in zip(cal, pred)]
    global_margin = quantile(residuals)
    groups = {}
    for group in ["ramp", "turn", "cruise"]:
        idx = [i for i, r in enumerate(cal) if r["group"] == group]
        enough = len(idx) >= 40 and len({tuple(cal[i]["signature"]) for i in idx}) >= 2
        groups[group] = dict(
            windows=len(idx),
            fallback_global=not enough,
            margin_mm=(
                quantile([residuals[i] for i in idx]) if enough else global_margin
            ),
        )
    variant = dict(
        columns=list(range(40)) + list(range(60, 64)),
        trees=forest_json(model),
        calibration=dict(global_margin_mm=global_margin, groups=groups),
    )
    reference = json.loads(MODEL.read_text())
    exported = dict(
        schema=1,
        history_s=0.45,
        horizon_s=0.25,
        feature_count=44,
        variants=dict(short=variant),
        dataset_sha256=metadata["original_dataset_sha256"],
        published_data_sha256=sha(args.data),
        train_physical_signatures=[list(s) for s in sorted(signatures)],
        calibration_physical_signatures=[list(s) for s in sorted(cal_signatures)],
    )
    output.mkdir(parents=True)
    path = output / "model.json"
    path.write_text(json.dumps(exported, separators=(",", ":")))
    portable = ShortHistoryPredictor(path)
    parity = max(abs(portable.predict(r["x"]) - float(p)) for r, p in zip(cal, pred))
    reference_variant = reference["variants"]["short"]
    reference_equal = variant == reference_variant
    trees_equal = variant["trees"] == reference_variant["trees"]
    calibration_error = max(
        abs(
            groups[g]["margin_mm"]
            - reference_variant["calibration"]["groups"][g]["margin_mm"]
        )
        for g in groups
    )
    calibration_error = max(
        calibration_error,
        abs(global_margin - reference_variant["calibration"]["global_margin_mm"]),
    )
    report = dict(
        train_windows=len(train),
        calibration_windows=len(cal),
        train_conditions=len(signatures),
        calibration_conditions=len(cal_signatures),
        feature_count=44,
        portable_max_error=parity,
        reference_short_variant_identical=reference_equal,
        reference_trees_identical=trees_equal,
        max_reference_calibration_difference=calibration_error,
        reference_equivalent=trees_equal and calibration_error <= 1e-12,
        calibration_mae_mm=float(np.mean(np.abs(pred - [r["y"] for r in cal]))),
        cross_validation=folds,
        model_sha256=sha(path),
        data_sha256=sha(args.data),
        source_sha256=sha(Path(__file__)),
        limitation="Evaluation cargos are excluded. No online weight updates, parameter search, real-time or safety guarantee.",
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not trees_equal or calibration_error > 1e-12 or parity > 1e-8:
        raise RuntimeError(
            "Retraining differs from the fixed reference; preserve report and inspect runtime versions"
        )


if __name__ == "__main__":
    main()
