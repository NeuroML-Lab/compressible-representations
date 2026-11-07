#!/usr/bin/env python3
"""Figure 5A--B: early/late task-sensitive versus null-space CKA."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import tempfile
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets
from utilz.vision_common import (
        DISPLAY_NAMES, IMAGENET_TRANSFORM, MODEL_NAMES, ViT,
        centered_linear_cka, eigenspaces, make_loader, project, set_seed,
    )


PAPER_RANKS = (25, 50, 100, 200, 300)


def split_imagenet(path: Path, seed: int, n_calibration: int, n_test: int):
    dataset = datasets.ImageFolder(path, transform=IMAGENET_TRANSFORM)
    required = n_calibration + n_test
    if len(dataset) < required:
        raise ValueError(f"Need {required} ImageNet images; found {len(dataset)}")
    indices = np.random.default_rng(seed).permutation(len(dataset))[:required]
    calibration = Subset(dataset, indices[:n_calibration].tolist())
    test = Subset(dataset, indices[n_calibration:].tolist())
    return calibration, test, indices


def estimate_model(model: ViT, calibration_loader, test_loader,
                   max_rank: int) -> dict:
    """Estimate all layer matrices in one backward pass per image batch."""
    layers = model.layer_indices_for_depth_experiment()
    fisher = {
        layer: torch.zeros(model.width, model.width, dtype=torch.float64)
        for layer in layers
    }
    n = 0

    for images, _labels in calibration_loader:
        images = images.to(model.device, non_blocking=True).requires_grad_(True)
        with model.capture_blocks(layers) as captured:
            logits = model.model(images)
            selected = logits.gather(1, logits.argmax(1, keepdim=True)).sum()
            layer_outputs = [captured[layer] for layer in layers]
            gradients = torch.autograd.grad(selected, layer_outputs)

        for layer, gradient in zip(layers, gradients, strict=True):
            cls_gradient = gradient[:, 0].detach().double().cpu()
            fisher[layer] += cls_gradient.T @ cls_gradient
        n += images.shape[0]

    features = {layer: [] for layer in layers}
    with torch.no_grad():
        for images, _labels in test_loader:
            images = images.to(model.device, non_blocking=True)
            with model.capture_blocks(layers) as captured:
                model.model(images)
            for layer in layers:
                features[layer].append(captured[layer][:, 0].float().cpu())

    output = {
        "model": model.key,
        "display_name": model.display_name,
        "width": model.width,
        "num_blocks": model.num_blocks,
        "n_calibration": n,
        "layers": {},
    }
    for layer in layers:
        values, top, bottom = eigenspaces(fisher[layer] / n, max_rank)
        output["layers"][layer] = {
            "features": torch.cat(features[layer]),
            "top": top,
            "bottom": bottom,
            "eigenvalues": values.float(),
        }
    return output


def matched_layers(left: dict, right: dict) -> list[tuple[int, int]]:
    left_n, right_n = left["num_blocks"], right["num_blocks"]
    if left_n == right_n == 12:
        return list(zip(range(12), range(12), strict=True))
    count = 8
    left_layers = np.linspace(0, left_n - 1, count, dtype=int).tolist()
    right_layers = np.linspace(0, right_n - 1, count, dtype=int).tolist()
    return list(zip(left_layers, right_layers, strict=True))


def compare_models(left: dict, right: dict, ranks: tuple[int, ...]) -> list[dict]:
    rows = []
    for left_layer, right_layer in matched_layers(left, right):
        a = left["layers"][left_layer]
        b = right["layers"][right_layer]
        depth = 0.5 * (
            left_layer / (left["num_blocks"] - 1) +
            right_layer / (right["num_blocks"] - 1)
        )
        for rank in ranks:
            left_top, right_top = a["top"][:, :rank], b["top"][:, :rank]
            left_bottom = a["bottom"][:, -rank:]
            right_bottom = b["bottom"][:, -rank:]
            rows.append({
                "model_a": left["display_name"],
                "model_b": right["display_name"],
                "layer_a": left_layer,
                "layer_b": right_layer,
                "relative_depth": depth,
                "rank": rank,
                "task_cka": centered_linear_cka(
                    project(a["features"], left_top),
                    project(b["features"], right_top),
                ),
                "null_cka": centered_linear_cka(
                    project(a["features"], left_bottom),
                    project(b["features"], right_bottom),
                ),
            })
    return rows


def early_late_summary(rows: list[dict], ranks: tuple[int, ...],
                       early_depth: float, late_depth: float) -> list[dict]:
    pairs = sorted({(r["model_a"], r["model_b"]) for r in rows})
    summary = []
    for label, target in (("early", early_depth), ("late", late_depth)):
        for rank in ranks:
            task_values, null_values = [], []
            for pair in pairs:
                candidates = [
                    r for r in rows
                    if (r["model_a"], r["model_b"]) == pair and r["rank"] == rank
                ]
                chosen = min(candidates, key=lambda r: abs(r["relative_depth"] - target))
                task_values.append(chosen["task_cka"])
                null_values.append(chosen["null_cka"])
            n = len(task_values)
            summary.append({
                "depth_group": label,
                "target_relative_depth": target,
                "rank": rank,
                "n_model_pairs": n,
                "task_cka_mean": float(np.mean(task_values)),
                "task_cka_sem": float(np.std(task_values, ddof=0) / np.sqrt(n)),
                "null_cka_mean": float(np.mean(null_values)),
                "null_cka_sem": float(np.std(null_values, ddof=0) / np.sqrt(n)),
            })
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: list[dict], output_dir: Path) -> None:
    ranks = sorted({r["rank"] for r in summary})
    x = np.arange(len(ranks))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    colors = {"task": "#2166AC", "null": "#B2182B"}

    for axis, group in zip(axes, ("early", "late"), strict=True):
        rows = [r for r in summary if r["depth_group"] == group]
        rows.sort(key=lambda r: r["rank"])
        axis.bar(
            x - width / 2,
            [r["task_cka_mean"] for r in rows],
            width,
            yerr=[r["task_cka_sem"] for r in rows],
            color=colors["task"],
            capsize=3,
            label="Task-sensitive",
        )
        axis.bar(
            x + width / 2,
            [r["null_cka_mean"] for r in rows],
            width,
            yerr=[r["null_cka_sem"] for r in rows],
            color=colors["null"],
            capsize=3,
            label="Null",
        )
        axis.set_xticks(x, [str(rank) for rank in ranks])
        axis.set_xlabel("Subspace rank")
        axis.set_title(f"{group.capitalize()} layers")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Linear CKA")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "task_null_early_late.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "task_null_early_late.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenet-val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES,
                        default=list(MODEL_NAMES))
    parser.add_argument("--ranks", nargs="+", type=int,
                        default=list(PAPER_RANKS))
    parser.add_argument("--n-calibration", type=int, default=960)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--early-depth", type=float, default=0.10)
    parser.add_argument("--late-depth", type=float, default=0.82)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    calibration, test, indices = split_imagenet(
        args.imagenet_val, args.seed, args.n_calibration, args.n_test,
    )
    calibration_loader = make_loader(
        calibration, args.batch_size, args.workers,
    )
    test_loader = make_loader(test, args.batch_size, args.workers)
    np.save(args.output_dir / "imagenet_indices.npy", indices)

    caches = {}
    for key in args.models:
        cache_path = cache_dir / f"{key}.pt"
        if args.reuse_cache and cache_path.exists():
            caches[key] = torch.load(cache_path, map_location="cpu",
                                     weights_only=False)
            continue
        print(f"Estimating {DISPLAY_NAMES[key]}")
        model = ViT(key, args.device)
        caches[key] = estimate_model(
            model, calibration_loader, test_loader, max(args.ranks),
        )
        torch.save(caches[key], cache_path)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = []
    for left_key, right_key in combinations(args.models, 2):
        rows.extend(compare_models(
            caches[left_key], caches[right_key], tuple(args.ranks),
        ))
    summary = early_late_summary(
        rows, tuple(args.ranks), args.early_depth, args.late_depth,
    )
    write_csv(args.output_dir / "task_null_all_layers.csv", rows)
    write_csv(args.output_dir / "task_null_early_late.csv", summary)
    plot_summary(summary, args.output_dir)

    config = {
        "models": args.models,
        "model_checkpoints": {key: MODEL_NAMES[key] for key in args.models},
        "ranks": args.ranks,
        "n_calibration": args.n_calibration,
        "n_test": args.n_test,
        "early_depth": args.early_depth,
        "late_depth": args.late_depth,
        "seed": args.seed,
        "task_output": "top-1 predicted ImageNet logit",
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    (args.output_dir / "results.json").write_text(json.dumps({
        "config": config,
        "all_layers": rows,
        "early_late": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
