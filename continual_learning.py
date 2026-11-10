#!/usr/bin/env python3
"""Reproduce the continual-learning experiment in Table 1.

This is a narrow entry point around the tested implementation in
``cl_benchmark_v3.py``. It exposes only the models, methods, and hyperparameters
reported in the final paper and writes one tidy record per run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import utilz.cl_benchmark_v3 as core  # noqa: E402


PAPER_ARCHITECTURES = ("alexnet", "alexnet-l", "lenet", "resnet18")
PAPER_METHODS = ("naive", "gpm", "fgpm", "joint")


def result_record(dataset: str, label: str, result: core.Results,
                  runtime_s: float, seed: int) -> dict:
    return {
        "dataset": dataset,
        "architecture": result.arch,
        "method": result.method,
        "label": label,
        "seed": seed,
        "num_tasks": int(result.acc_matrix.shape[0]),
        "average_final_accuracy": 100.0 * result.avg_final_acc,
        "forgetting": 100.0 * result.forgetting,
        "runtime_seconds": runtime_s,
        "accuracy_matrix": result.acc_matrix.tolist(),
        "basis_ranks_per_task": result.basis_ranks_per_task,
    }


def run_one(tasks, classes_per_task: int, architecture: str, method: str,
            threshold: float | None, args) -> dict:
    cfg = core.Config(
        method=method,
        arch=architecture,
        use_bn=True,
        epochs_per_task=args.epochs,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0,
        energy_gpm=threshold or 0.97,
        energy_fgpm=threshold or 0.97,
        threshold_increment=0.003,
        max_basis_samples=1500,
        seed=args.seed,
        device=args.device,
    )
    cfg.label = method if threshold is None else f"{method} e={threshold}"
    start = time.perf_counter()
    if method == "joint":
        result = core.run_joint(tasks, classes_per_task, cfg)
    else:
        result = core.run_continual(tasks, classes_per_task, cfg)
    elapsed = time.perf_counter() - start
    return result_record(args.current_dataset, cfg.label, result, elapsed,
                         args.seed)


def select_table_rows(records: list[dict]) -> list[dict]:
    """Apply the threshold-selection rule stated in Appendix F."""
    rows = []
    keys = sorted({(r["dataset"], r["architecture"]) for r in records})
    for dataset, architecture in keys:
        group = [r for r in records if r["dataset"] == dataset and
                 r["architecture"] == architecture]
        for method in PAPER_METHODS:
            candidates = [r for r in group if r["method"] == method]
            if not candidates:
                continue
            best = max(candidates, key=lambda r: r["average_final_accuracy"])
            rows.append({
                "dataset": dataset,
                "architecture": architecture,
                "method": method,
                "selected_threshold": (
                    best["label"].split("=")[-1]
                    if method in {"gpm", "fgpm"} else ""
                ),
                "average_final_accuracy": best["average_final_accuracy"],
                "forgetting": best["forgetting"],
                "runtime_seconds": best["runtime_seconds"],
            })
    return rows


def write_outputs(records: list[dict], output_dir: Path, config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"config": config, "runs": records}
    (output_dir / "all_runs.json").write_text(json.dumps(payload, indent=2))

    rows = select_table_rows(records)
    if rows:
        with (output_dir / "table1.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "final_results" / "continual_learning")
    parser.add_argument("--benchmarks", nargs="+",
                        choices=("cifar10", "cifar100"),
                        default=["cifar10", "cifar100"])
    parser.add_argument("--architectures", nargs="+",
                        choices=PAPER_ARCHITECTURES,
                        default=list(PAPER_ARCHITECTURES))
    parser.add_argument("--methods", nargs="+", choices=PAPER_METHODS,
                        default=list(PAPER_METHODS))
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.90, 0.95, 0.97])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--limit-tasks", type=int, default=None,
        help="Optional code-path check; omit for all paper results.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: list[dict] = []
    for dataset in args.benchmarks:
        args.current_dataset = dataset
        tasks, classes_per_task = core.build_split_cifar(
            dataset, str(args.data_root), args.batch_size,
            args.num_workers, args.seed,
        )
        expected_tasks = 5 if dataset == "cifar10" else 20
        if len(tasks) != expected_tasks:
            raise RuntimeError(
                f"{dataset} produced {len(tasks)} tasks; expected {expected_tasks}"
            )
        if args.limit_tasks is not None:
            tasks = tasks[:args.limit_tasks]

        for architecture in args.architectures:
            for method in args.methods:
                thresholds = args.thresholds if method in {"gpm", "fgpm"} else [None]
                for threshold in thresholds:
                    print(f"\n{dataset} | {architecture} | {method} | {threshold}")
                    records.append(run_one(
                        tasks, classes_per_task, architecture, method,
                        threshold, args,
                    ))
                    write_outputs(records, args.output_dir, vars(args) | {
                        "data_root": str(args.data_root),
                        "output_dir": str(args.output_dir),
                    })


if __name__ == "__main__":
    main()
