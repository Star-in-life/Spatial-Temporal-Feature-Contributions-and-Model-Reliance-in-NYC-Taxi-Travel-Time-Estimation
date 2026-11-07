"""Recompute Table VI or evaluate one supplied checkpoint on its own TEST tier."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from metrics import compute_metrics_s
from run_diagnostics_90k import BaselineModel

MODELS = ["LightGBM", "CatBoost", "RealMLP", "TabM"]
SEEDS = [42, 43, 44]


class CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Supplied checkpoints refer to AttributeDict at an older Lightning module path.
        if (module, name) == ("lightning_fabric.utilities.data", "AttributeDict"):
            from pytorch_lightning.utilities.parsing import AttributeDict
            return AttributeDict
        return super().find_class(module, name)


def evaluate_checkpoint(path, test_path=None):
    path = Path(path).resolve()
    manifest = json.loads((ROOT / "checkpoints/manifest.json").read_text(encoding="utf-8"))
    expected = manifest["files"].get(path.name) if path.parent == ROOT / "checkpoints" else None
    if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {path.name}")
    with path.open("rb") as stream:
        checkpoint = CheckpointUnpickler(stream).load()
    predictor = checkpoint.get("predictor", checkpoint.get("pipeline"))
    if predictor is None or not hasattr(predictor, "predict"):
        raise ValueError(f"Checkpoint has no predictor or pipeline: {path.name}")
    model = predictor.model_name
    tier = checkpoint.get("tier", "90k")
    if tier.startswith("F"):
        tier = "90k"
    seed_match = re.search(r"(?:seed_)?(42|43|44)\.pkl$", path.name)
    seed = checkpoint.get("seed")
    if seed is None:
        if seed_match is None:
            raise ValueError("Checkpoint must contain a seed or use a supplied checkpoint filename")
        seed = int(seed_match.group(1))
    test_path = Path(test_path) if test_path else ROOT / "scaling_subsets" / f"test_{tier}.parquet"
    frame = pd.read_parquet(test_path)
    if frame.empty or not np.isfinite(frame["dur_min"]).all():
        raise ValueError("TEST must contain finite dur_min values and at least one trip")
    prediction = np.asarray(predictor.predict(frame), dtype=np.float64).reshape(-1)
    if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid predictions from {path.name}")
    result = {"model": model, "tier": tier, "seed": seed, "rows": len(frame)}
    result.update(compute_metrics_s(frame["dur_min"].to_numpy(), prediction))
    del predictor, checkpoint
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, help="Evaluate one checkpoint instead of the full Table VI")
    parser.add_argument("--model", choices=[name.lower() for name in MODELS], help="Expected model family for --ckpt")
    parser.add_argument("--test-path", type=Path, help="Override TEST input for --ckpt")
    args = parser.parse_args()
    if (args.model or args.test_path) and args.ckpt is None:
        parser.error("--model and --test-path require --ckpt")
    output_dir = ROOT / "results/evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.ckpt:
        path = args.ckpt if args.ckpt.is_absolute() else ROOT / args.ckpt
        result = evaluate_checkpoint(path, args.test_path)
        if args.model and result["model"].lower() != args.model:
            parser.error("--model does not match the checkpoint")
        print(json.dumps(result, indent=2))
        pd.DataFrame([result]).to_csv(output_dir / "checkpoint_metrics.csv", index=False)
        return
    records = []
    for model in MODELS:
        for seed in SEEDS:
            path = ROOT / "checkpoints" / f"refit_F3__{model}__{seed}.pkl"
            result = evaluate_checkpoint(path)
            records.append(result)
            print(f"{model} seed {seed}: MAE={result['MAE']:.6f} s", flush=True)
    train = pd.read_parquet(ROOT / "scaling_subsets/train_90k.parquet")
    test = pd.read_parquet(ROOT / "scaling_subsets/test_90k.parquet")
    for name in ["OSRM", "TEMP"]:
        baseline = BaselineModel(name).fit(train)
        result = {"model": name, "tier": "90k", "seed": 42, "rows": len(test)}
        result.update(compute_metrics_s(test["dur_min"].to_numpy(), baseline.predict(test)))
        records.append(result)
    per_seed = pd.DataFrame(records)
    metrics = ["MAE", "RMSE", "MedAE", "WAPE", "RMSLE", "R2"]
    summary = per_seed.groupby("model", sort=False)[metrics].agg(["mean", "std"]).fillna(0.0)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    per_seed.to_csv(output_dir / "table_vi_per_seed.csv", index=False)
    summary.to_csv(output_dir / "table_vi_recomputed.csv")
    print("\nRecomputed Table VI. Duration errors are seconds. RMSLE uses minutes.")
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"\nSaved evaluation results to {output_dir}")


if __name__ == "__main__":
    main()
