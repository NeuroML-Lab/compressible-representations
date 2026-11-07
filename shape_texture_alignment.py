#!/usr/bin/env python3
"""Figure 5C: shape-conditioned versus texture-conditioned CKA."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
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
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from utilz.vision import IMAGENET_TRANSFORM, ViT, MLPProbe, centered_linear_cka, energy_eigenspace, project, set_seed

CATEGORIES = (
    "airplane", "bear", "bicycle", "bird", "boat", "bottle", "car",
    "cat", "chair", "clock", "dog", "elephant", "keyboard", "knife",
    "oven", "truck",
)


class CueConflictDataset(Dataset):
    """The 1,200 Geirhos images whose shape and texture labels differ."""

    def __init__(self, root: Path):
        category_to_index = {name: i for i, name in enumerate(CATEGORIES)}
        samples = []
        for path in sorted(root.rglob("*.png")):
            match = re.match(r"([a-z]+)\d+[-_]([a-z]+)\d+", path.stem)
            if match is None:
                continue
            shape, texture = match.groups()
            if shape == texture:
                continue
            if shape in category_to_index and texture in category_to_index:
                samples.append((
                    path,
                    category_to_index[shape],
                    category_to_index[texture],
                ))
        if len(samples) != 1200:
            raise ValueError(f"Expected 1,200 cue-conflict images; found {len(samples)}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, shape, texture = self.samples[index]
        with Image.open(path) as image:
            image = IMAGENET_TRANSFORM(image.convert("RGB"))
        return image, shape, texture


def make_loader(dataset, batch_size: int, workers: int,
                shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def extract_features(model: ViT, loader: DataLoader) -> dict[str, torch.Tensor]:
    """Extract penultimate CLS states and downstream features for the probe."""
    target = model.penultimate_block
    target_features, probe_features = [], []
    shape_labels, texture_labels = [], []
    with torch.no_grad():
        for images, shape, texture in loader:
            images = images.to(model.device, non_blocking=True)
            with model.capture_blocks([target]) as captured:
                final_tokens = model.model.forward_features(images)
            final_cls = final_tokens[:, 0] if final_tokens.ndim == 3 else final_tokens
            target_features.append(captured[target][:, 0].float().cpu())
            probe_features.append(final_cls.float().cpu())
            shape_labels.append(shape)
            texture_labels.append(texture)
    return {
        "target": torch.cat(target_features),
        "probe": torch.cat(probe_features),
        "shape": torch.cat(shape_labels),
        "texture": torch.cat(texture_labels),
    }


def train_probe(features: torch.Tensor, labels: torch.Tensor, device: str,
                epochs: int, seed: int) -> MLPProbe:
    set_seed(seed)
    probe = MLPProbe(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=5e-3, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs,
    )
    loader = DataLoader(
        TensorDataset(features, labels), batch_size=64, shuffle=True,
    )
    for _epoch in range(epochs):
        probe.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(probe(x), y).backward()
            optimizer.step()
        scheduler.step()
    return probe.eval()


@torch.no_grad()
def accuracy(probe: MLPProbe, features: torch.Tensor,
             labels: torch.Tensor, device: str) -> float:
    prediction = probe(features.to(device)).argmax(1).cpu()
    return float((prediction == labels).float().mean())


def estimate_matrix(model: ViT, probe: MLPProbe,
                    loader: DataLoader) -> tuple[torch.Tensor, int]:
    """Accumulate the full probe-logit Jacobian at the penultimate CLS token."""
    target = model.penultimate_block
    width = model.width
    matrix = torch.zeros(width, width, dtype=torch.float64)
    count = 0
    probe.requires_grad_(False)

    for images, _shape, _texture in loader:
        images = images.to(model.device, non_blocking=True).requires_grad_(True)
        with model.capture_blocks([target]) as captured:
            final_tokens = model.model.forward_features(images)
            final_cls = final_tokens[:, 0] if final_tokens.ndim == 3 else final_tokens
            logits = probe(final_cls)
            target_output = captured[target]
            for output_index in range(logits.shape[1]):
                gradient = torch.autograd.grad(
                    logits[:, output_index].sum(),
                    target_output,
                    retain_graph=output_index < logits.shape[1] - 1,
                )[0][:, 0]
                gradient = gradient.detach().double().cpu()
                matrix += gradient.T @ gradient
        count += images.shape[0]
    matrix /= count
    return 0.5 * (matrix + matrix.T), count


def process_model(model_key: str, train_loader: DataLoader,
                  test_loader: DataLoader, args) -> dict:
    model = ViT(model_key, args.device)
    train = extract_features(model, train_loader)
    test = extract_features(model, test_loader)

    shape_probe = train_probe(
        train["probe"], train["shape"], args.device, args.probe_epochs,
        args.seed,
    )
    texture_probe = train_probe(
        train["probe"], train["texture"], args.device, args.probe_epochs,
        args.seed + 1,
    )
    shape_accuracy = accuracy(
        shape_probe, test["probe"], test["shape"], args.device,
    )
    texture_accuracy = accuracy(
        texture_probe, test["probe"], test["texture"], args.device,
    )

    shape_matrix, n_shape = estimate_matrix(model, shape_probe, train_loader)
    texture_matrix, n_texture = estimate_matrix(model, texture_probe, train_loader)
    shape_values, shape_basis = energy_eigenspace(
        shape_matrix, args.energy_fraction,
    )
    texture_values, texture_basis = energy_eigenspace(
        texture_matrix, args.energy_fraction,
    )

    output = {
        "model": model_key,
        "display_name": model.display_name,
        "layer": model.penultimate_block,
        "width": model.width,
        "shape_test_accuracy": shape_accuracy,
        "texture_test_accuracy": texture_accuracy,
        "shape_rank": shape_basis.shape[1],
        "texture_rank": texture_basis.shape[1],
        "n_shape": n_shape,
        "n_texture": n_texture,
        "test_target_features": test["target"],
        "shape_basis": shape_basis,
        "texture_basis": texture_basis,
        "shape_eigenvalues": shape_values.float(),
        "texture_eigenvalues": texture_values.float(),
    }
    del model, shape_probe, texture_probe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def compare_models(left: dict, right: dict) -> dict:
    return {
        "model_a": left["display_name"],
        "model_b": right["display_name"],
        "shape_cka": centered_linear_cka(
            project(left["test_target_features"], left["shape_basis"]),
            project(right["test_target_features"], right["shape_basis"]),
        ),
        "texture_cka": centered_linear_cka(
            project(left["test_target_features"], left["texture_basis"]),
            project(right["test_target_features"], right["texture_basis"]),
        ),
        "shape_rank_a": left["shape_rank"],
        "shape_rank_b": right["shape_rank"],
        "texture_rank_a": left["texture_rank"],
        "texture_rank_b": right["texture_rank"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: list[dict], output_dir: Path) -> None:
    shape = np.array([r["shape_cka"] for r in rows])
    texture = np.array([r["texture_cka"] for r in rows])
    means = [shape.mean(), texture.mean()]
    sems = [shape.std(ddof=0) / np.sqrt(len(shape)),
            texture.std(ddof=0) / np.sqrt(len(texture))]

    fig, axis = plt.subplots(figsize=(2.7, 3.1))
    axis.bar(
        [0, 1], means, yerr=sems, width=0.58,
        color=["#2166AC", "#B2182B"], capsize=4,
    )
    for shape_value, texture_value in zip(shape, texture, strict=True):
        axis.plot([0, 1], [shape_value, texture_value], "-o",
                  color="0.25", alpha=0.4, linewidth=0.8, markersize=3)
    axis.set_xticks([0, 1], ["Shape", "Texture"])
    axis.set_ylabel("Task-conditioned linear CKA")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "shape_texture_cka.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "shape_texture_cka.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cue-conflict-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES,
                        default=list(MODEL_NAMES))
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--energy-fraction", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
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

    dataset = CueConflictDataset(args.cue_conflict_dir)
    permutation = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))
    train_indices = permutation[:960]
    test_indices = permutation[960:]
    torch.save(
        {"train": train_indices, "test": test_indices},
        args.output_dir / "split_indices.pt",
    )
    train_loader = make_loader(
        Subset(dataset, train_indices.tolist()), args.batch_size,
        args.workers, shuffle=False,
    )
    test_loader = make_loader(
        Subset(dataset, test_indices.tolist()), args.batch_size,
        args.workers, shuffle=False,
    )

    model_results = {}
    for model_key in args.models:
        cache_path = cache_dir / f"{model_key}.pt"
        if args.reuse_cache and cache_path.exists():
            model_results[model_key] = torch.load(
                cache_path, map_location="cpu", weights_only=False,
            )
            continue
        print(f"Processing {DISPLAY_NAMES[model_key]}")
        model_results[model_key] = process_model(
            model_key, train_loader, test_loader, args,
        )
        torch.save(model_results[model_key], cache_path)

    rows = [
        compare_models(model_results[left], model_results[right])
        for left, right in combinations(args.models, 2)
    ]
    write_csv(args.output_dir / "shape_texture_pairs.csv", rows)

    model_rows = [{
        key: value for key, value in result.items()
        if isinstance(value, (str, int, float))
    } for result in model_results.values()]
    write_csv(args.output_dir / "shape_texture_models.csv", model_rows)
    plot_results(rows, args.output_dir)

    config = {
        "models": args.models,
        "model_checkpoints": {key: MODEL_NAMES[key] for key in args.models},
        "n_train": 960,
        "n_test": 240,
        "probe": "MLP(256, ReLU, dropout=0.1)",
        "probe_epochs": args.probe_epochs,
        "optimizer": "AdamW(lr=5e-3, weight_decay=1e-4)",
        "energy_fraction": args.energy_fraction,
        "seed": args.seed,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    (args.output_dir / "results.json").write_text(json.dumps({
        "config": config,
        "models": model_rows,
        "pairs": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
