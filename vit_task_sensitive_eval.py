#!/usr/bin/env python3
"""
Task-sensitive ViT representation pruning on ImageNet with cached utility curves.

Supports two allocation modes:

A) utility_curve
   - estimate F_l at each selected layer
   - build single-layer utility curves over retained ranks
   - allocate k_l under a global retained-dimension budget by maximizing summed utility

D) spectral_normalized
   - estimate F_l at each selected layer
   - eigendecompose each F_l
   - normalize each layer's eigenvalues by its total eigenvalue mass
   - allocate k_l by keeping the largest normalized eigenvalues globally under budget

The expensive single-layer task-sensitivity utility evaluation is cached and can be
reused across different --retain-fraction runs as long as the sensitive config matches.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.models import (
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    ViT_L_16_Weights,
    ViT_L_32_Weights,
    vit_b_16,
    vit_b_32,
    vit_l_16,
    vit_l_32,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def metric_key(metric: str) -> str:
    if metric not in {"top1", "top5"}:
        raise ValueError(f"Unknown metric {metric}")
    return metric


def safe_json(obj: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> Dict[int, float]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    out: Dict[int, float] = {}
    for k in topk:
        out[k] = float(correct[:k].any(dim=0).float().sum().item())
    return out


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def choose_indices(n_total: int, n_select: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    n = min(n_total, n_select)
    return rng.choice(n_total, size=n, replace=False).tolist()


def parse_layer_subset(spec: str, n_layers: int) -> List[int]:
    spec = spec.strip()
    if spec.lower() == "all":
        return list(range(n_layers))
    out: List[int] = []
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-")
            lo, hi = int(a), int(b)
            out.extend(list(range(lo, hi + 1)))
        else:
            out.append(int(p))
    out = sorted(set(x for x in out if 0 <= x < n_layers))
    if not out:
        raise ValueError(f"Layer subset {spec!r} produced no valid indices for n_layers={n_layers}")
    return out


def parse_fraction_grid(spec: str) -> List[float]:
    vals = []
    for part in spec.split(","):
        x = float(part.strip())
        if not (0.0 < x <= 1.0):
            raise ValueError(f"Invalid retained fraction {x}; must be in (0,1].")
        vals.append(x)
    vals = sorted(set(vals))
    return vals


def nearest_int_budget(total_dims: int, retain_fraction: float) -> int:
    return int(round(total_dims * retain_fraction))


MODEL_REGISTRY = {
    "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
    "vit_b_32": (vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1),
    "vit_l_16": (vit_l_16, ViT_L_16_Weights.IMAGENET1K_V1),
    "vit_l_32": (vit_l_32, ViT_L_32_Weights.IMAGENET1K_V1),
}


@dataclass
class LoadedModel:
    model: nn.Module
    preprocess: Callable
    categories: Sequence[str]
    num_layers: int
    hidden_dim: int


def load_vit(model_name: str, device: torch.device) -> LoadedModel:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name={model_name}. Choices: {list(MODEL_REGISTRY)}")
    constructor, weights = MODEL_REGISTRY[model_name]
    model = constructor(weights=weights)
    model.eval().to(device)
    preprocess = weights.transforms()
    categories = weights.meta.get("categories", [])
    num_layers = len(model.encoder.layers)
    hidden_dim = model.hidden_dim
    return LoadedModel(
        model=model,
        preprocess=preprocess,
        categories=categories,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )


DEFAULT_CLASS_INDEX_URLS = [
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json",
    "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
]


def load_imagenet_class_index_json(class_index_json: Optional[str], output_dir: Path) -> Dict[str, Tuple[str, str]]:
    if class_index_json is not None:
        with open(class_index_json, "r") as f:
            return json.load(f)

    cache_dir = make_dir(output_dir / "imagenet_meta")
    cache_file = cache_dir / "imagenet_class_index.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)

    last_err = None
    for url in DEFAULT_CLASS_INDEX_URLS:
        try:
            urllib.request.urlretrieve(url, cache_file)
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception as e:
            last_err = e
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except Exception:
                    pass
            continue

    raise RuntimeError(
        "Could not load imagenet_class_index.json automatically. "
        "Pass --class-index-json PATH to a local copy. "
        f"Last error: {last_err}"
    )


class ImageFolderWithRemappedTargets(Dataset):
    def __init__(self, root: str, transform: Callable, class_index_json: Dict[str, Tuple[str, str]]):
        self.base = datasets.ImageFolder(root=root, transform=transform)

        wnid_to_imagenet_idx: Dict[str, int] = {}
        for idx_str, pair in class_index_json.items():
            idx = int(idx_str)
            wnid = pair[0]
            wnid_to_imagenet_idx[wnid] = idx

        missing = [wnid for wnid in self.base.classes if wnid not in wnid_to_imagenet_idx]
        if missing:
            raise ValueError(
                "Some folder names are not valid ImageNet wnids or missing from class_index_json. "
                f"First few missing: {missing[:10]}"
            )

        self.old_to_new = {
            old_idx: wnid_to_imagenet_idx[wnid]
            for wnid, old_idx in self.base.class_to_idx.items()
        }

        self.classes = self.base.classes
        self.class_to_idx = self.base.class_to_idx
        self.targets = [self.old_to_new[t] for t in self.base.targets]
        self.samples = self.base.samples

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, old_target = self.base[idx]
        return img, self.old_to_new[old_target]


class MultiCLSProjectionHook:
    def __init__(self, projectors: Dict[int, torch.Tensor], renorm: bool):
        self.projectors = projectors
        self.renorm = renorm

    def make_hook(self, layer_idx: int):
        projector = self.projectors[layer_idx]

        def hook(_module: nn.Module, _inputs: Tuple[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
            x = output.clone()
            cls = x[:, 0, :]
            proj = cls @ projector.T
            if self.renorm:
                src_norm = cls.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                proj_norm = proj.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                proj = proj * (src_norm / proj_norm)
            x[:, 0, :] = proj
            return x

        return hook


class SingleCLSProjectionHook:
    def __init__(self, projector: torch.Tensor, renorm: bool):
        self.projector = projector
        self.renorm = renorm

    def __call__(self, _module: nn.Module, _inputs: Tuple[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
        x = output.clone()
        cls = x[:, 0, :]
        proj = cls @ self.projector.T
        if self.renorm:
            src_norm = cls.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            proj_norm = proj.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            proj = proj * (src_norm / proj_norm)
        x[:, 0, :] = proj
        return x


def estimate_task_sensitive_matrix(
    model: nn.Module,
    loader: DataLoader,
    layer_idx: int,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    target_layer = model.encoder.layers[layer_idx]
    hidden_dim = model.hidden_dim
    F = torch.zeros(hidden_dim, hidden_dim, dtype=torch.float64, device="cpu")
    n_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        cache: Dict[str, torch.Tensor] = {}

        def hook_fn(_module: nn.Module, _inputs: Tuple[torch.Tensor], output: torch.Tensor):
            output.retain_grad()
            cache["layer_output"] = output

        h = target_layer.register_forward_hook(hook_fn)
        model.zero_grad(set_to_none=True)
        try:
            logits = model(images)
            true_logits = logits.gather(1, labels[:, None]).sum()
            true_logits.backward()
        finally:
            h.remove()

        layer_out = cache["layer_output"]
        grads = layer_out.grad[:, 0, :]
        grads_cpu = grads.detach().double().cpu()
        F += grads_cpu.T @ grads_cpu
        n_samples += grads_cpu.shape[0]

    if n_samples == 0:
        raise RuntimeError("No samples processed while estimating task-sensitivity matrix")

    F /= float(n_samples)
    F = 0.5 * (F + F.T)
    eigvals = torch.linalg.eigvalsh(F).cpu().float()
    stats = {
        "n_samples": float(n_samples),
        "trace": float(torch.trace(F).item()),
        "max_eig": float(eigvals.max().item()),
        "min_eig": float(eigvals.min().item()),
        "effective_rank_participation_ratio": float((eigvals.sum() ** 2 / eigvals.pow(2).sum().clamp_min(1e-12)).item()),
    }
    return F.float(), stats


def evaluate_single_layer_projection(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    layer_idx: int,
    projector: Optional[torch.Tensor],
    renorm: bool,
) -> Dict[str, float]:
    model.eval()
    hook = None
    if projector is not None:
        hook = model.encoder.layers[layer_idx].register_forward_hook(SingleCLSProjectionHook(projector.to(device), renorm=renorm))

    top1 = 0.0
    top5 = 0.0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            accs = accuracy_topk(logits, labels, topk=(1, 5))
            bs = labels.shape[0]
            top1 += accs[1]
            top5 += accs[5]
            total += bs
    if hook is not None:
        hook.remove()

    return {"top1": top1 / max(total, 1), "top5": top5 / max(total, 1), "n_eval": float(total)}


def evaluate_multi_layer_projection(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    projectors: Dict[int, torch.Tensor],
    renorm: bool,
) -> Dict[str, float]:
    model.eval()
    handles = []
    hook_factory = MultiCLSProjectionHook({k: v.to(device) for k, v in projectors.items()}, renorm=renorm)
    for layer_idx in sorted(projectors):
        handles.append(model.encoder.layers[layer_idx].register_forward_hook(hook_factory.make_hook(layer_idx)))

    top1 = 0.0
    top5 = 0.0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            accs = accuracy_topk(logits, labels, topk=(1, 5))
            bs = labels.shape[0]
            top1 += accs[1]
            top5 += accs[5]
            total += bs

    for h in handles:
        h.remove()

    return {"top1": top1 / max(total, 1), "top5": top5 / max(total, 1), "n_eval": float(total)}


def build_candidate_ranks(hidden_dim: int, fraction_grid: Sequence[float]) -> List[int]:
    ranks = {max(1, min(hidden_dim - 1, int(round(hidden_dim * frac)))) for frac in fraction_grid}
    return sorted(ranks)


def solve_allocation_dp(
    layer_candidates: Dict[int, List[Dict[str, float]]],
    budget_dims: int,
    utility_metric: str,
) -> Dict[int, Dict[str, float]]:
    layers = sorted(layer_candidates)
    dp: Dict[int, Dict[int, float]] = {0: {0: 0.0}}
    back: Dict[Tuple[int, int], Tuple[int, int, int]] = {}

    for i, layer in enumerate(layers, start=1):
        dp[i] = {}
        cand_list = layer_candidates[layer]
        for prev_b, prev_val in dp[i - 1].items():
            for j, cand in enumerate(cand_list):
                b = prev_b + int(cand["rank"])
                if b > budget_dims:
                    continue
                val = prev_val + float(cand[utility_metric])
                if (b not in dp[i]) or (val > dp[i][b]):
                    dp[i][b] = val
                    back[(i, b)] = (prev_b, layer, j)

    if not dp[len(layers)]:
        raise RuntimeError(
            f"No feasible allocation found under budget {budget_dims}. "
            f"Minimum required budget is at least {sum(min(int(c['rank']) for c in layer_candidates[l]) for l in layers)}."
        )

    final_b, _ = max(dp[len(layers)].items(), key=lambda kv: kv[1])
    chosen: Dict[int, Dict[str, float]] = {}
    cur_b = final_b
    for i in range(len(layers), 0, -1):
        prev_b, layer, j = back[(i, cur_b)]
        chosen[layer] = layer_candidates[layer][j]
        cur_b = prev_b
    return chosen


def allocate_by_normalized_spectral_mass(
    layer_eigvals: Dict[int, torch.Tensor],
    budget_dims: int,
) -> Dict[int, int]:
    entries: List[Tuple[float, int, int]] = []
    for layer_idx, eigvals in layer_eigvals.items():
        total = float(eigvals.sum().item())
        denom = total if total > 1e-12 else 1.0
        normed = (eigvals / denom).tolist()
        for i, score in enumerate(normed, start=1):
            entries.append((float(score), int(layer_idx), int(i)))

    entries.sort(key=lambda x: x[0], reverse=True)
    keep_counts = {layer_idx: 0 for layer_idx in layer_eigvals.keys()}
    for _, layer_idx, _ in entries[:budget_dims]:
        keep_counts[layer_idx] += 1

    for layer_idx, eigvals in layer_eigvals.items():
        d = eigvals.numel()
        keep_counts[layer_idx] = max(1, min(d - 1, keep_counts[layer_idx]))

    current = sum(keep_counts.values())
    if current > budget_dims:
        removable: List[Tuple[float, int]] = []
        for layer_idx, eigvals in layer_eigvals.items():
            total = float(eigvals.sum().item())
            denom = total if total > 1e-12 else 1.0
            normed = (eigvals / denom).tolist()
            k = keep_counts[layer_idx]
            for i in range(k, 0, -1):
                removable.append((float(normed[i - 1]), int(layer_idx)))
        removable.sort(key=lambda x: x[0])
        idx = 0
        while current > budget_dims and idx < len(removable):
            _, layer_idx = removable[idx]
            if keep_counts[layer_idx] > 1:
                keep_counts[layer_idx] -= 1
                current -= 1
            idx += 1

    current = sum(keep_counts.values())
    if current < budget_dims:
        addable: List[Tuple[float, int]] = []
        for layer_idx, eigvals in layer_eigvals.items():
            total = float(eigvals.sum().item())
            denom = total if total > 1e-12 else 1.0
            normed = (eigvals / denom).tolist()
            k = keep_counts[layer_idx]
            for i in range(k + 1, len(normed) + 1):
                addable.append((float(normed[i - 1]), int(layer_idx)))
        addable.sort(key=lambda x: x[0], reverse=True)
        idx = 0
        while current < budget_dims and idx < len(addable):
            _, layer_idx = addable[idx]
            d = layer_eigvals[layer_idx].numel()
            if keep_counts[layer_idx] < d - 1:
                keep_counts[layer_idx] += 1
                current += 1
            idx += 1

    return keep_counts


def build_cache_config(args, layer_indices: List[int], hidden_dim: int, fraction_grid: List[float]) -> dict:
    return {
        "model_name": args.model_name,
        "train_dir": str(args.train_dir),
        "val_dir": str(args.val_dir),
        "layer_indices": list(layer_indices),
        "hidden_dim": int(hidden_dim),
        "fraction_grid": list(fraction_grid),
        "calibration_samples": int(args.calibration_samples),
        "eval_samples": int(args.eval_samples),
        "seed": int(args.seed),
        "max_calibration_batches": None if args.max_calibration_batches is None else int(args.max_calibration_batches),
    }


def cache_matches(expected: dict, observed: dict) -> bool:
    return expected == observed


def get_cache_paths(cache_dir: Path) -> dict:
    return {
        "meta": cache_dir / "utility_cache_meta.json",
        "utility": cache_dir / "single_layer_utility_curves.csv",
        "spectra": cache_dir / "layer_spectra.csv",
        "eigenspaces": cache_dir / "task_sensitive_eigenspaces.pt"}


def maybe_load_cached_task_sensitive(
    cache_dir: Path,
    expected_cache_config: dict,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]]:
    paths = get_cache_paths(cache_dir)
    if not all(p.exists() for p in paths.values()):
        return None
    observed = load_json(paths["meta"])
    if not cache_matches(expected_cache_config, observed):
        return None

    utility_df = pd.read_csv(paths["utility"])
    spectra_df = pd.read_csv(paths["spectra"])
    saved = torch.load(paths["eigenspaces"], map_location="cpu", weights_only=False)
    eigvals_by_layer = {int(k): v for k, v in saved["eigvals_by_layer"].items()}
    eigvecs_by_layer = {int(k): v for k, v in saved["eigvecs_by_layer"].items()}
    return utility_df, spectra_df, eigvals_by_layer, eigvecs_by_layer


def save_task_sensitive_cache(
    cache_dir: Path,
    cache_config: dict,
    utility_df: pd.DataFrame,
    spectra_df: pd.DataFrame,
    eigvals_by_layer: Dict[int, torch.Tensor],
    eigvecs_by_layer: Dict[int, torch.Tensor],
) -> None:
    paths = get_cache_paths(cache_dir)
    safe_json(cache_config, paths["meta"])
    utility_df.to_csv(paths["utility"], index=False)
    spectra_df.to_csv(paths["spectra"], index=False)
    torch.save(
        {
            "eigvals_by_layer": {int(k): v.cpu() for k, v in eigvals_by_layer.items()},
            "eigvecs_by_layer": {int(k): v.cpu() for k, v in eigvecs_by_layer.items()},
        },
        paths["eigenspaces"])


def build_projector_banks_from_eigenspaces(
    eigvecs_by_layer: Dict[int, torch.Tensor],
    candidate_ranks: List[int],
    hidden_dim: int,
) -> Tuple[Dict[int, Dict[int, torch.Tensor]], Dict[int, Dict[int, torch.Tensor]]]:
    task_projectors_by_layer: Dict[int, Dict[int, torch.Tensor]] = {}
    null_projectors_by_layer: Dict[int, Dict[int, torch.Tensor]] = {}
    for layer_idx, eigvecs in eigvecs_by_layer.items():
        task_projectors_by_layer[layer_idx] = {}
        null_projectors_by_layer[layer_idx] = {}
        for rank in candidate_ranks:
            rank = max(1, min(hidden_dim - 1, int(rank)))
            U_task = eigvecs[:, :rank]
            U_null = eigvecs[:, rank:]
            task_projectors_by_layer[layer_idx][rank] = (U_task @ U_task.T).cpu()
            null_projectors_by_layer[layer_idx][rank] = (U_null @ U_null.T).cpu()
    return task_projectors_by_layer, null_projectors_by_layer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--class-index-json", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="vit_b_16", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer-subset", type=str, default="all")
    parser.add_argument("--retain-fraction", type=float, default=0.5)
    parser.add_argument("--fraction-grid", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--allocation-mode", type=str, default="utility_curve", choices=["utility_curve", "spectral_normalized"])
    parser.add_argument("--utility-condition", type=str, default="task_raw", choices=["task_raw", "task_renorm"])
    parser.add_argument("--utility-metric", type=str, default="top1", choices=["top1", "top5"])
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="vit_task_sensitive_combined")
    parser.add_argument("--utility-cache-dir", type=str, default=None, help="Directory for expensive task-sensitivity utility-curve cache. Defaults to output-dir.")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-calibration-batches", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    outdir = make_dir(args.output-dir if False else args.output_dir)
    cache_dir = make_dir(args.utility_cache_dir) if args.utility_cache_dir is not None else outdir

    loaded = load_vit(args.model_name, device=device)
    model = loaded.model
    preprocess = loaded.preprocess
    layer_indices = parse_layer_subset(args.layer_subset, loaded.num_layers)

    class_index_json = load_imagenet_class_index_json(args.class_index_json, outdir)
    train_ds = ImageFolderWithRemappedTargets(args.train_dir, preprocess, class_index_json)
    val_ds = ImageFolderWithRemappedTargets(args.val_dir, preprocess, class_index_json)

    calib_idx = choose_indices(len(train_ds), args.calibration_samples, seed=args.seed)
    eval_idx = choose_indices(len(val_ds), args.eval_samples, seed=args.seed + 1)
    calib_ds = Subset(train_ds, calib_idx)
    eval_ds = Subset(val_ds, eval_idx)

    calib_loader = DataLoader(calib_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Selected layers: {layer_indices}")
    print(f"Hidden dim: {loaded.hidden_dim}")
    print(f"Utility cache dir: {cache_dir}")

    baseline = evaluate_single_layer_projection(model=model, loader=eval_loader, device=device, layer_idx=layer_indices[0], projector=None, renorm=False)
    print(f"Baseline | top1={baseline['top1']:.4f} | top5={baseline['top5']:.4f}")

    fraction_grid = parse_fraction_grid(args.fraction_grid)
    candidate_ranks = build_candidate_ranks(loaded.hidden_dim, fraction_grid)
    utility_key = metric_key(args.utility_metric)
    print(f"Candidate retained fractions: {fraction_grid}")
    print(f"Candidate ranks: {candidate_ranks}")

    cache_config = build_cache_config(args, layer_indices, loaded.hidden_dim, fraction_grid)
    cached = maybe_load_cached_task_sensitive(cache_dir, cache_config)

    if cached is not None:
        print("Reusing cached task-sensitivity utility curves and eigenspaces.")
        utility_df, spectra_df, eigvals_by_layer, eigvecs_by_layer = cached
    else:
        print("No compatible cache found. Computing task-sensitivity matrices and utility curves.")
        curves_rows: List[Dict[str, float | int | str]] = []
        spectra_rows: List[Dict[str, float | int]] = []
        eigvals_by_layer: Dict[int, torch.Tensor] = {}
        eigvecs_by_layer: Dict[int, torch.Tensor] = {}

        task_projectors_by_layer_tmp: Dict[int, Dict[int, torch.Tensor]] = {}
        null_projectors_by_layer_tmp: Dict[int, Dict[int, torch.Tensor]] = {}

        for layer_idx in layer_indices:
            print(f"Estimating F for layer {layer_idx} / {loaded.num_layers - 1}")
            F, F_stats = estimate_task_sensitive_matrix(model=model, loader=calib_loader, layer_idx=layer_idx, device=device, max_batches=args.max_calibration_batches)

            eigvals, eigvecs = torch.linalg.eigh(F)
            order = torch.argsort(eigvals, descending=True)
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            eigvals_by_layer[layer_idx] = eigvals.cpu()
            eigvecs_by_layer[layer_idx] = eigvecs.cpu()

            total_mass = float(eigvals.sum().item())
            for i, lam in enumerate(eigvals.tolist()):
                spectra_rows.append({
                    "layer_index": layer_idx,
                    "eig_rank": i + 1,
                    "eigenvalue": float(lam),
                    "normalized_eigenvalue": float(lam / total_mass) if total_mass > 1e-12 else 0.0,
                    "cumulative_mass_fraction": float(eigvals[: i + 1].sum().item() / max(total_mass, 1e-12)),
                })

            task_projectors_by_layer_tmp[layer_idx] = {}
            null_projectors_by_layer_tmp[layer_idx] = {}
            for rank in candidate_ranks:
                rank = max(1, min(loaded.hidden_dim - 1, int(rank)))
                U_task = eigvecs[:, :rank]
                U_null = eigvecs[:, rank:]
                task_projectors_by_layer_tmp[layer_idx][rank] = (U_task @ U_task.T).cpu()
                null_projectors_by_layer_tmp[layer_idx][rank] = (U_null @ U_null.T).cpu()

            if args.allocation_mode == "utility_curve":
                print(f"  Evaluating single-layer utility curve for layer {layer_idx}")
                for rank in candidate_ranks:
                    Pi_task = task_projectors_by_layer_tmp[layer_idx][rank]
                    Pi_null = null_projectors_by_layer_tmp[layer_idx][rank]

                    m_task_raw = evaluate_single_layer_projection(model=model, loader=eval_loader, device=device, layer_idx=layer_idx, projector=Pi_task, renorm=False)
                    m_task_renorm = evaluate_single_layer_projection(model=model, loader=eval_loader, device=device, layer_idx=layer_idx, projector=Pi_task, renorm=True)
                    m_null_raw = evaluate_single_layer_projection(model=model, loader=eval_loader, device=device, layer_idx=layer_idx, projector=Pi_null, renorm=False)
                    m_null_renorm = evaluate_single_layer_projection(model=model, loader=eval_loader, device=device, layer_idx=layer_idx, projector=Pi_null, renorm=True)

                    curves_rows.append({
                        "layer_index": layer_idx,
                        "hidden_dim": loaded.hidden_dim,
                        "rank": int(rank),
                        "retain_fraction_layer": float(rank / loaded.hidden_dim),
                        "top1_task_raw": float(m_task_raw["top1"]),
                        "top5_task_raw": float(m_task_raw["top5"]),
                        "top1_task_renorm": float(m_task_renorm["top1"]),
                        "top5_task_renorm": float(m_task_renorm["top5"]),
                        "top1_null_raw": float(m_null_raw["top1"]),
                        "top5_null_raw": float(m_null_raw["top5"]),
                        "top1_null_renorm": float(m_null_renorm["top1"]),
                        "top5_null_renorm": float(m_null_renorm["top5"]),
                        "delta_top1_task_raw_vs_baseline": float(m_task_raw["top1"] - baseline["top1"]),
                        "delta_top5_task_raw_vs_baseline": float(m_task_raw["top5"] - baseline["top5"]),
                        "delta_top1_task_renorm_vs_baseline": float(m_task_renorm["top1"] - baseline["top1"]),
                        "delta_top5_task_renorm_vs_baseline": float(m_task_renorm["top5"] - baseline["top5"]),
                        "delta_top1_null_raw_vs_baseline": float(m_null_raw["top1"] - baseline["top1"]),
                        "delta_top5_null_raw_vs_baseline": float(m_null_raw["top5"] - baseline["top5"]),
                        "delta_top1_null_renorm_vs_baseline": float(m_null_renorm["top1"] - baseline["top1"]),
                        "delta_top5_null_renorm_vs_baseline": float(m_null_renorm["top5"] - baseline["top5"]),
                        **{f"F_{k}": v for k, v in F_stats.items()},
                    })
                    print(f"    rank={rank:>3d} frac={rank/loaded.hidden_dim:.3f} | task_raw top1={m_task_raw['top1']:.4f} | task_renorm top1={m_task_renorm['top1']:.4f}")

        utility_df = pd.DataFrame(curves_rows)
        spectra_df = pd.DataFrame(spectra_rows)
        save_task_sensitive_cache(cache_dir, cache_config, utility_df, spectra_df, eigvals_by_layer, eigvecs_by_layer)
        print("Saved task-sensitivity cache.")

    task_projectors_by_layer, null_projectors_by_layer = build_projector_banks_from_eigenspaces(
        eigvecs_by_layer=eigvecs_by_layer,
        candidate_ranks=candidate_ranks,
        hidden_dim=loaded.hidden_dim,
    )

    utility_df.to_csv(outdir / "single_layer_utility_curves.csv", index=False)
    spectra_df.to_csv(outdir / "layer_spectra.csv", index=False)

    total_dims = loaded.hidden_dim * len(layer_indices)
    budget_dims = nearest_int_budget(total_dims, args.retain_fraction)

    if args.allocation_mode == "utility_curve":
        if utility_df.empty:
            raise RuntimeError("utility_curve allocation requested, but cached utility curves are empty.")
        layer_candidates: Dict[int, List[Dict[str, float]]] = {}
        for layer_idx in layer_indices:
            sub = utility_df[utility_df["layer_index"] == layer_idx].sort_values("rank")
            cand_list = []
            for _, row in sub.iterrows():
                util = float(row[f"{utility_key}_{args.utility_condition}"])
                cand_list.append({
                    "rank": int(row["rank"]),
                    "utility": util,
                    "retain_fraction_layer": float(row["retain_fraction_layer"]),
                })
            layer_candidates[layer_idx] = cand_list

        chosen = solve_allocation_dp(layer_candidates, budget_dims=budget_dims, utility_metric="utility")
        allocation_rows = []
        for layer_idx in layer_indices:
            c = chosen[layer_idx]
            allocation_rows.append({
                "layer_index": layer_idx,
                "hidden_dim": loaded.hidden_dim,
                "k_l": int(c["rank"]),
                "retain_fraction_layer": float(c["retain_fraction_layer"]),
                "allocation_mode": args.allocation_mode,
                "utility_score_used_for_allocation": float(c["utility"]),
            })
    else:
        keep_counts = allocate_by_normalized_spectral_mass(eigvals_by_layer, budget_dims=budget_dims)
        allocation_rows = []
        for layer_idx in layer_indices:
            k = int(keep_counts[layer_idx])
            total_mass = float(eigvals_by_layer[layer_idx].sum().item())
            retained_mass = float(eigvals_by_layer[layer_idx][:k].sum().item())
            allocation_rows.append({
                "layer_index": layer_idx,
                "hidden_dim": loaded.hidden_dim,
                "k_l": k,
                "retain_fraction_layer": float(k / loaded.hidden_dim),
                "allocation_mode": args.allocation_mode,
                "retained_eigen_mass": retained_mass,
                "total_eigen_mass": total_mass,
                "retained_eigen_mass_fraction": retained_mass / max(total_mass, 1e-12),
            })

    allocation_df = pd.DataFrame(allocation_rows).sort_values("layer_index").reset_index(drop=True)
    allocation_df.to_csv(outdir / "allocation.csv", index=False)

    print("Allocation summary:")
    print(json.dumps({
        "allocation_mode": args.allocation_mode,
        "retain_fraction_global": args.retain_fraction,
        "num_selected_layers": len(layer_indices),
        "total_dims": total_dims,
        "target_keep_dims": budget_dims,
        "achieved_keep_dims": int(allocation_df["k_l"].sum()),
        "layer_indices": layer_indices,
        "utility_condition": args.utility_condition if args.allocation_mode == "utility_curve" else None,
        "utility_metric": args.utility_metric if args.allocation_mode == "utility_curve" else None,
    }, indent=2))
    print("k_l by layer:", allocation_df["k_l"].tolist())

    task_projectors: Dict[int, torch.Tensor] = {}
    null_projectors: Dict[int, torch.Tensor] = {}
    for _, row in allocation_df.iterrows():
        layer_idx = int(row["layer_index"])
        k = int(row["k_l"])
        task_projectors[layer_idx] = task_projectors_by_layer[layer_idx][k]
        null_projectors[layer_idx] = null_projectors_by_layer[layer_idx][k]

    print("Evaluating baseline / allocated task-pruned / allocated null-pruned conditions...")
    final_rows: List[Dict[str, float | str]] = [{"condition": "baseline", **baseline}]

    for cond_name, projectors, renorm in [
        ("keep_task_raw", task_projectors, False),
        ("keep_null_raw", null_projectors, False),
        ("keep_task_renorm", task_projectors, True),
        ("keep_null_renorm", null_projectors, True),
    ]:
        metrics = evaluate_multi_layer_projection(model=model, loader=eval_loader, device=device, projectors=projectors, renorm=renorm)
        final_rows.append({
            "condition": cond_name,
            **metrics,
            "delta_top1_vs_baseline": float(metrics["top1"] - baseline["top1"]),
            "delta_top5_vs_baseline": float(metrics["top5"] - baseline["top5"]),
        })
        print(f"{cond_name:>16s} | top1={metrics['top1']:.4f} | top5={metrics['top5']:.4f}")

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(outdir / "results.csv", index=False)

    metadata = {
        "model_name": args.model_name,
        "train_dir": args.train_dir,
        "val_dir": args.val_dir,
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "layer_subset": args.layer_subset,
        "resolved_layers": layer_indices,
        "retain_fraction": args.retain_fraction,
        "fraction_grid": fraction_grid,
        "allocation_mode": args.allocation_mode,
        "utility_condition": args.utility_condition,
        "utility_metric": args.utility_metric,
        "calibration_samples": args.calibration_samples,
        "eval_samples": args.eval_samples,
        "hidden_dim": loaded.hidden_dim,
        "num_layers_total": loaded.num_layers,
        "budget_dims": budget_dims,
        "achieved_keep_dims": int(allocation_df["k_l"].sum()),
        "utility_cache_dir": str(cache_dir),
    }
    safe_json(metadata, outdir / "metadata.json")
    torch.save({
        "task_projectors": task_projectors,
        "null_projectors": null_projectors,
        "allocation": allocation_df.to_dict(orient="records"),
    }, outdir / "projectors.pt")

    print(f"Saved results to: {outdir}")
    print(final_df.to_string(index=False))
    print("\\nChosen allocation:")
    print(allocation_df.to_string(index=False))


if __name__ == "__main__":
    main()
