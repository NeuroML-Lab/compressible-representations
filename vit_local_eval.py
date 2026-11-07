#!/usr/bin/env python3
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
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    for p in [x.strip() for x in spec.split(",") if x.strip()]:
        if "-" in p:
            a, b = p.split("-")
            out.extend(list(range(int(a), int(b) + 1)))
        else:
            out.append(int(p))
    out = sorted(set(x for x in out if 0 <= x < n_layers))
    if not out:
        raise ValueError(f"No valid layers parsed from {spec!r}")
    return out


def parse_fraction_grid(spec: str) -> List[float]:
    vals = sorted(set(float(x.strip()) for x in spec.split(",") if x.strip()))
    for x in vals:
        if not (0.0 < x <= 1.0):
            raise ValueError(f"Invalid retained fraction {x}")
    return vals


def nearest_int_budget(total_dims: int, retain_fraction: float) -> int:
    return int(round(total_dims * retain_fraction))


def metric_key(metric: str) -> str:
    if metric not in {"top1", "top5"}:
        raise ValueError(metric)
    return metric


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
    num_layers: int
    hidden_dim: int


def load_vit(model_name: str, device: torch.device) -> LoadedModel:
    constructor, weights = MODEL_REGISTRY[model_name]
    model = constructor(weights=weights)
    model.eval().to(device)
    return LoadedModel(
        model=model,
        preprocess=weights.transforms(),
        num_layers=len(model.encoder.layers),
        hidden_dim=model.hidden_dim,
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
    raise RuntimeError(f"Could not load imagenet_class_index.json automatically. Last error: {last_err}")


class ImageFolderWithRemappedTargets(Dataset):
    def __init__(self, root: str, transform: Callable, class_index_json: Dict[str, Tuple[str, str]]):
        self.base = datasets.ImageFolder(root=root, transform=transform)
        wnid_to_imagenet_idx = {pair[0]: int(idx) for idx, pair in class_index_json.items()}
        missing = [wnid for wnid in self.base.classes if wnid not in wnid_to_imagenet_idx]
        if missing:
            raise ValueError(f"Missing wnids in class index: {missing[:10]}")
        self.old_to_new = {old_idx: wnid_to_imagenet_idx[wnid] for wnid, old_idx in self.base.class_to_idx.items()}
        self.targets = [self.old_to_new[t] for t in self.base.targets]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, old_target = self.base[idx]
        return img, self.old_to_new[old_target]


class SingleCLSProjectionHook:
    def __init__(self, projector: torch.Tensor, renorm: bool):
        self.projector = projector
        self.renorm = renorm

    def __call__(self, _module: nn.Module, _inputs, output: torch.Tensor) -> torch.Tensor:
        x = output.clone()
        cls = x[:, 0, :]
        proj = cls @ self.projector.T
        if self.renorm:
            src_norm = cls.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            proj_norm = proj.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            proj = proj * (src_norm / proj_norm)
        x[:, 0, :] = proj
        return x


class MultiCLSProjectionHook:
    def __init__(self, projectors: Dict[int, torch.Tensor], renorm: bool):
        self.projectors = projectors
        self.renorm = renorm

    def make_hook(self, layer_idx: int):
        projector = self.projectors[layer_idx]
        def hook(_module: nn.Module, _inputs, output: torch.Tensor) -> torch.Tensor:
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


@torch.no_grad()
def collect_cls_representations(
    model: nn.Module,
    loader: DataLoader,
    layer_idx: int,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> torch.Tensor:
    target_layer = model.encoder.layers[layer_idx]
    all_z = []
    for batch_idx, (images, _labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        cache = {}

        def hook_fn(_module: nn.Module, _inputs, output: torch.Tensor):
            cache["layer_output"] = output.detach()

        h = target_layer.register_forward_hook(hook_fn)
        try:
            _ = model(images)
        finally:
            h.remove()

        all_z.append(cache["layer_output"][:, 0, :].cpu())
    if not all_z:
        raise RuntimeError("No representations collected.")
    return torch.cat(all_z, dim=0).float()


def pairwise_distance_matrix(Z: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "euclidean":
        return torch.cdist(Z, Z, p=2)
    if metric == "cosine":
        Z_norm = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-8)
        sim = (Z_norm @ Z_norm.T).clamp(-1.0, 1.0)
        return 1.0 - sim
    raise ValueError(metric)


def build_knn_weight_matrix(D: torch.Tensor, k: int, weight_mode: str, sigma: Optional[float]) -> Tuple[torch.Tensor, Dict[str, float]]:
    n = D.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be < n={n}")
    D_work = D.clone()
    D_work.fill_diagonal_(float("inf"))
    nn_idx = torch.topk(D_work, k=k, largest=False, dim=1).indices
    W = torch.zeros_like(D)

    if weight_mode == "binary":
        for i in range(n):
            W[i, nn_idx[i]] = 1.0
        sigma_used = None
    elif weight_mode == "gaussian":
        finite_d = D_work[D_work.isfinite()]
        sigma_used = float(torch.median(finite_d).item()) if sigma is None else float(sigma)
        sigma_used = max(sigma_used, 1e-8)
        for i in range(n):
            nbrs = nn_idx[i]
            W[i, nbrs] = torch.exp(-(D[i, nbrs] ** 2) / (2.0 * sigma_used ** 2))
    else:
        raise ValueError(weight_mode)

    W = 0.5 * (W + W.T)
    W.fill_diagonal_(0.0)
    stats = {
        "n_nodes": float(n),
        "k_neighbors": float(k),
        "avg_degree_weighted": float(W.sum(dim=1).mean().item()),
        "num_nonzero_edges_directed": float((W > 0).sum().item()),
    }
    if weight_mode == "gaussian":
        stats["sigma_used"] = sigma_used
    return W, stats


def estimate_local_geometry_matrix(
    model: nn.Module,
    loader: DataLoader,
    layer_idx: int,
    device: torch.device,
    k_neighbors: int,
    neighbor_metric: str,
    weight_mode: str,
    sigma: Optional[float],
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    Z = collect_cls_representations(model, loader, layer_idx, device, max_batches)
    D = pairwise_distance_matrix(Z, neighbor_metric)
    W, w_stats = build_knn_weight_matrix(D, k_neighbors, weight_mode, sigma)
    deg = W.sum(dim=1)
    L = torch.diag(deg) - W
    M = 2.0 * (Z.T @ L @ Z)
    M = 0.5 * (M + M.T)
    eigvals = torch.linalg.eigvalsh(M).float()
    stats = {
        "n_samples": float(Z.shape[0]),
        "trace": float(torch.trace(M).item()),
        "max_eig": float(eigvals.max().item()),
        "min_eig": float(eigvals.min().item()),
        "effective_rank_participation_ratio": float((eigvals.sum() ** 2 / eigvals.pow(2).sum().clamp_min(1e-12)).item()),
        "neighbor_metric": neighbor_metric,
        "weight_mode": weight_mode,
        **w_stats,
    }
    return M.float(), stats


def evaluate_single_layer_projection(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    layer_idx: int,
    projector: Optional[torch.Tensor],
    renorm: bool,
) -> Dict[str, float]:
    handle = None
    if projector is not None:
        handle = model.encoder.layers[layer_idx].register_forward_hook(SingleCLSProjectionHook(projector.to(device), renorm))
    top1 = top5 = 0.0
    total = 0
    model.eval()
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
    if handle is not None:
        handle.remove()
    return {"top1": top1 / max(total, 1), "top5": top5 / max(total, 1), "n_eval": float(total)}


def evaluate_multi_layer_projection(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    projectors: Dict[int, torch.Tensor],
    renorm: bool,
) -> Dict[str, float]:
    hook_factory = MultiCLSProjectionHook({k: v.to(device) for k, v in projectors.items()}, renorm)
    handles = [model.encoder.layers[layer_idx].register_forward_hook(hook_factory.make_hook(layer_idx))
               for layer_idx in sorted(projectors)]
    top1 = top5 = 0.0
    total = 0
    model.eval()
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
    return sorted({max(1, min(hidden_dim - 1, int(round(hidden_dim * frac)))) for frac in fraction_grid})


def solve_allocation_dp(layer_candidates: Dict[int, List[Dict[str, float]]], budget_dims: int) -> Dict[int, Dict[str, float]]:
    layers = sorted(layer_candidates)
    dp: Dict[int, Dict[int, float]] = {0: {0: 0.0}}
    back: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    for i, layer in enumerate(layers, start=1):
        dp[i] = {}
        for prev_b, prev_val in dp[i - 1].items():
            for j, cand in enumerate(layer_candidates[layer]):
                b = prev_b + int(cand["rank"])
                if b > budget_dims:
                    continue
                val = prev_val + float(cand["utility"])
                if (b not in dp[i]) or (val > dp[i][b]):
                    dp[i][b] = val
                    back[(i, b)] = (prev_b, layer, j)
    if not dp[len(layers)]:
        raise RuntimeError("No feasible allocation found")
    final_b, _ = max(dp[len(layers)].items(), key=lambda kv: kv[1])
    chosen = {}
    cur_b = final_b
    for i in range(len(layers), 0, -1):
        prev_b, layer, j = back[(i, cur_b)]
        chosen[layer] = layer_candidates[layer][j]
        cur_b = prev_b
    return chosen


def allocate_by_normalized_spectral_mass(layer_eigvals: Dict[int, torch.Tensor], budget_dims: int) -> Dict[int, int]:
    entries: List[Tuple[float, int, int]] = []
    for layer_idx, eigvals in layer_eigvals.items():
        total = float(eigvals.sum().item())
        normed = (eigvals / (total if total > 1e-12 else 1.0)).tolist()
        for i, score in enumerate(normed, start=1):
            entries.append((float(score), int(layer_idx), int(i)))
    entries.sort(key=lambda x: x[0], reverse=True)
    keep_counts = {layer_idx: 0 for layer_idx in layer_eigvals}
    for _, layer_idx, _ in entries[:budget_dims]:
        keep_counts[layer_idx] += 1
    for layer_idx, eigvals in layer_eigvals.items():
        d = eigvals.numel()
        keep_counts[layer_idx] = max(1, min(d - 1, keep_counts[layer_idx]))
    return keep_counts


def build_cache_config(args, layer_indices: List[int], hidden_dim: int, fraction_grid: List[float]) -> dict:
    return {
        "model_name": args.model_name,
        "train_dir": str(args.train_dir),
        "val_dir": str(args.val_dir),
        "layer_indices": list(layer_indices),
        "hidden_dim": int(hidden_dim),
        "fraction_grid": list(fraction_grid),
        "k_neighbors": int(args.k_neighbors),
        "neighbor_metric": str(args.neighbor_metric),
        "weight_mode": str(args.weight_mode),
        "sigma": None if args.sigma is None else float(args.sigma),
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
        "utility": cache_dir / "single_layer_local_geometry_utility_curves.csv",
        "spectra": cache_dir / "layer_local_geometry_spectra.csv",
        "eigenspaces": cache_dir / "local_geometry_eigenspaces.pt",
    }


def maybe_load_cached_local_geometry(
    cache_dir: Path,
    expected_cache_config: dict,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]]:
    paths = get_cache_paths(cache_dir)
    if not all(p.exists() for p in paths.values()):
        return None
    observed = load_json(paths["meta"])
    if not cache_matches(expected_cache_config, observed):
        return None

    utility_df = pd.read_csv(paths["utility"]) if paths["utility"].exists() else pd.DataFrame()
    spectra_df = pd.read_csv(paths["spectra"]) if paths["spectra"].exists() else pd.DataFrame()
    saved = torch.load(paths["eigenspaces"], map_location="cpu")
    eigvals_by_layer = {int(k): v for k, v in saved["eigvals_by_layer"].items()}
    eigvecs_by_layer = {int(k): v for k, v in saved["eigvecs_by_layer"].items()}
    return utility_df, spectra_df, eigvals_by_layer, eigvecs_by_layer


def save_local_geometry_cache(
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
        paths["eigenspaces"],
    )


def build_projector_banks_from_eigenspaces(
    eigvecs_by_layer: Dict[int, torch.Tensor],
    candidate_ranks: List[int],
    hidden_dim: int,
) -> Tuple[Dict[int, Dict[int, torch.Tensor]], Dict[int, Dict[int, torch.Tensor]]]:
    local_projectors_by_layer: Dict[int, Dict[int, torch.Tensor]] = {}
    null_projectors_by_layer: Dict[int, Dict[int, torch.Tensor]] = {}
    for layer_idx, eigvecs in eigvecs_by_layer.items():
        local_projectors_by_layer[layer_idx] = {}
        null_projectors_by_layer[layer_idx] = {}
        for rank in candidate_ranks:
            rank = max(1, min(hidden_dim - 1, int(rank)))
            U_local = eigvecs[:, :rank]
            U_null = eigvecs[:, rank:]
            local_projectors_by_layer[layer_idx][rank] = (U_local @ U_local.T).cpu()
            null_projectors_by_layer[layer_idx][rank] = (U_null @ U_null.T).cpu()
    return local_projectors_by_layer, null_projectors_by_layer


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
    parser.add_argument("--utility-condition", type=str, default="local_raw", choices=["local_raw", "local_renorm"])
    parser.add_argument("--utility-metric", type=str, default="top1", choices=["top1", "top5"])
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--neighbor-metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
    parser.add_argument("--weight-mode", type=str, default="binary", choices=["binary", "gaussian"])
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="vit_local_geometry_optionA_or_D")
    parser.add_argument("--utility-cache-dir", type=str, default=None,
                        help="Directory for expensive local-geometry utility-curve cache. Defaults to output-dir.")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-calibration-batches", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = make_dir(args.output_dir)
    cache_dir = make_dir(args.utility_cache_dir) if args.utility_cache_dir is not None else outdir
    device = get_device(args.device)

    loaded = load_vit(args.model_name, device)
    model = loaded.model
    layer_indices = parse_layer_subset(args.layer_subset, loaded.num_layers)

    class_index_json = load_imagenet_class_index_json(args.class_index_json, outdir)
    train_ds = ImageFolderWithRemappedTargets(args.train_dir, loaded.preprocess, class_index_json)
    val_ds = ImageFolderWithRemappedTargets(args.val_dir, loaded.preprocess, class_index_json)

    calib_loader = DataLoader(
        Subset(train_ds, choose_indices(len(train_ds), args.calibration_samples, args.seed)),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    eval_loader = DataLoader(
        Subset(val_ds, choose_indices(len(val_ds), args.eval_samples, args.seed + 1)),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Selected layers: {layer_indices}")
    print(f"Hidden dim: {loaded.hidden_dim}")
    print(f"Utility cache dir: {cache_dir}")

    baseline = evaluate_single_layer_projection(model, eval_loader, device, layer_indices[0], None, False)
    print(f"Baseline | top1={baseline['top1']:.4f} | top5={baseline['top5']:.4f}")

    fraction_grid = parse_fraction_grid(args.fraction_grid)
    candidate_ranks = build_candidate_ranks(loaded.hidden_dim, fraction_grid)
    utility_key = metric_key(args.utility_metric)

    cache_config = build_cache_config(args, layer_indices, loaded.hidden_dim, fraction_grid)
    cached = maybe_load_cached_local_geometry(cache_dir, cache_config)

    if cached is not None:
        print("Reusing cached local-geometry utility curves and eigenspaces.")
        utility_df, spectra_df, eigvals_by_layer, eigvecs_by_layer = cached
    else:
        print("No compatible cache found. Computing local-geometry matrices and utility curves.")
        curves_rows: List[Dict[str, float | int | str]] = []
        spectra_rows: List[Dict[str, float | int]] = []
        eigvals_by_layer: Dict[int, torch.Tensor] = {}
        eigvecs_by_layer: Dict[int, torch.Tensor] = {}

        local_projectors_by_layer_tmp: Dict[int, Dict[int, torch.Tensor]] = {}
        null_projectors_by_layer_tmp: Dict[int, Dict[int, torch.Tensor]] = {}

        for layer_idx in layer_indices:
            print(f"Estimating local-geometry matrix for layer {layer_idx} / {loaded.num_layers - 1}")
            M, M_stats = estimate_local_geometry_matrix(
                model, calib_loader, layer_idx, device,
                args.k_neighbors, args.neighbor_metric, args.weight_mode, args.sigma,
                args.max_calibration_batches,
            )
            eigvals, eigvecs = torch.linalg.eigh(M)
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
                    "normalized_eigenvalue": float(lam / max(total_mass, 1e-12)),
                    "cumulative_mass_fraction": float(eigvals[: i + 1].sum().item() / max(total_mass, 1e-12)),
                })

            local_projectors_by_layer_tmp[layer_idx] = {}
            null_projectors_by_layer_tmp[layer_idx] = {}
            for rank in candidate_ranks:
                rank = max(1, min(loaded.hidden_dim - 1, int(rank)))
                U_local = eigvecs[:, :rank]
                U_null = eigvecs[:, rank:]
                local_projectors_by_layer_tmp[layer_idx][rank] = (U_local @ U_local.T).cpu()
                null_projectors_by_layer_tmp[layer_idx][rank] = (U_null @ U_null.T).cpu()

            if args.allocation_mode == "utility_curve":
                print(f"  Evaluating single-layer local-geometry utility curve for layer {layer_idx}")
                for rank in candidate_ranks:
                    Pi_local = local_projectors_by_layer_tmp[layer_idx][rank]
                    Pi_null = null_projectors_by_layer_tmp[layer_idx][rank]
                    m_local_raw = evaluate_single_layer_projection(model, eval_loader, device, layer_idx, Pi_local, False)
                    m_local_renorm = evaluate_single_layer_projection(model, eval_loader, device, layer_idx, Pi_local, True)
                    m_null_raw = evaluate_single_layer_projection(model, eval_loader, device, layer_idx, Pi_null, False)
                    m_null_renorm = evaluate_single_layer_projection(model, eval_loader, device, layer_idx, Pi_null, True)
                    retained_mass = float(eigvals[:rank].sum().item())
                    curves_rows.append({
                        "layer_index": layer_idx,
                        "hidden_dim": loaded.hidden_dim,
                        "rank": int(rank),
                        "retain_fraction_layer": float(rank / loaded.hidden_dim),
                        "retained_local_geometry_mass": retained_mass,
                        "total_local_geometry_mass": total_mass,
                        "retained_local_geometry_mass_fraction": retained_mass / max(total_mass, 1e-12),
                        "top1_local_raw": float(m_local_raw["top1"]),
                        "top5_local_raw": float(m_local_raw["top5"]),
                        "top1_local_renorm": float(m_local_renorm["top1"]),
                        "top5_local_renorm": float(m_local_renorm["top5"]),
                        "top1_null_raw": float(m_null_raw["top1"]),
                        "top5_null_raw": float(m_null_raw["top5"]),
                        "top1_null_renorm": float(m_null_renorm["top1"]),
                        "top5_null_renorm": float(m_null_renorm["top5"]),
                        **{f"M_{k}": v for k, v in M_stats.items()},
                    })
                    print(
                        f"    rank={rank:>3d} frac={rank/loaded.hidden_dim:.3f} | "
                        f"local_raw top1={m_local_raw['top1']:.4f} | local_renorm top1={m_local_renorm['top1']:.4f}"
                    )

        utility_df = pd.DataFrame(curves_rows)
        spectra_df = pd.DataFrame(spectra_rows)
        save_local_geometry_cache(cache_dir, cache_config, utility_df, spectra_df, eigvals_by_layer, eigvecs_by_layer)
        print("Saved local-geometry utility/eigenspace cache.")

    local_projectors_by_layer, null_projectors_by_layer = build_projector_banks_from_eigenspaces(
        eigvecs_by_layer=eigvecs_by_layer,
        candidate_ranks=candidate_ranks,
        hidden_dim=loaded.hidden_dim,
    )

    utility_df.to_csv(outdir / "single_layer_local_geometry_utility_curves.csv", index=False)
    spectra_df.to_csv(outdir / "layer_local_geometry_spectra.csv", index=False)

    total_dims = loaded.hidden_dim * len(layer_indices)
    budget_dims = nearest_int_budget(total_dims, args.retain_fraction)

    if args.allocation_mode == "utility_curve":
        if utility_df.empty:
            raise RuntimeError(
                "utility_curve allocation requested, but utility cache/data is empty. "
                "Populate the cache first with a utility_curve run."
            )
        layer_candidates: Dict[int, List[Dict[str, float]]] = {}
        for layer_idx in layer_indices:
            sub = utility_df[utility_df["layer_index"] == layer_idx].sort_values("rank")
            layer_candidates[layer_idx] = [{
                "rank": int(row["rank"]),
                "utility": float(row[f"{utility_key}_{args.utility_condition}"]),
                "retain_fraction_layer": float(row["retain_fraction_layer"]),
            } for _, row in sub.iterrows()]
        chosen = solve_allocation_dp(layer_candidates, budget_dims)
        allocation_rows = [{
            "layer_index": layer_idx,
            "hidden_dim": loaded.hidden_dim,
            "k_l": int(chosen[layer_idx]["rank"]),
            "retain_fraction_layer": float(chosen[layer_idx]["retain_fraction_layer"]),
            "allocation_mode": args.allocation_mode,
            "utility_score_used_for_allocation": float(chosen[layer_idx]["utility"]),
        } for layer_idx in layer_indices]
    else:
        keep_counts = allocate_by_normalized_spectral_mass(eigvals_by_layer, budget_dims)
        allocation_rows = []
        for layer_idx in layer_indices:
            raw_k = int(keep_counts[layer_idx])
            nearest_rank = min(candidate_ranks, key=lambda r: abs(r - raw_k))
            total_mass = float(eigvals_by_layer[layer_idx].sum().item())
            retained_mass = float(eigvals_by_layer[layer_idx][:nearest_rank].sum().item())
            allocation_rows.append({
                "layer_index": layer_idx,
                "hidden_dim": loaded.hidden_dim,
                "k_l": nearest_rank,
                "retain_fraction_layer": float(nearest_rank / loaded.hidden_dim),
                "allocation_mode": args.allocation_mode,
                "raw_allocated_k": raw_k,
                "retained_local_geometry_mass": retained_mass,
                "total_local_geometry_mass": total_mass,
                "retained_local_geometry_mass_fraction": retained_mass / max(total_mass, 1e-12),
            })

    allocation_df = pd.DataFrame(allocation_rows).sort_values("layer_index").reset_index(drop=True)
    allocation_df.to_csv(outdir / "allocation.csv", index=False)

    local_projectors = {int(layer): local_projectors_by_layer[int(layer)][int(k)]
                        for layer, k in zip(allocation_df["layer_index"], allocation_df["k_l"])}
    null_projectors = {int(layer): null_projectors_by_layer[int(layer)][int(k)]
                       for layer, k in zip(allocation_df["layer_index"], allocation_df["k_l"])}

    print("Evaluating baseline / allocated local-geometry / allocated null conditions...")
    final_rows = [{"condition": "baseline", **baseline}]
    for cond_name, projectors, renorm in [
        ("keep_local_raw", local_projectors, False),
        ("keep_null_raw", null_projectors, False),
        ("keep_local_renorm", local_projectors, True),
        ("keep_null_renorm", null_projectors, True),
    ]:
        metrics = evaluate_multi_layer_projection(model, eval_loader, device, projectors, renorm)
        final_rows.append({
            "condition": cond_name,
            **metrics,
            "delta_top1_vs_baseline": float(metrics["top1"] - baseline["top1"]),
            "delta_top5_vs_baseline": float(metrics["top5"] - baseline["top5"]),
        })
        print(f"{cond_name:>18s} | top1={metrics['top1']:.4f} | top5={metrics['top5']:.4f}")

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(outdir / "results.csv", index=False)

    safe_json({
        "model_name": args.model_name,
        "train_dir": args.train_dir,
        "val_dir": args.val_dir,
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "layer_subset": args.layer_subset,
        "resolved_layers": layer_indices,
        "fraction_grid": fraction_grid,
        "retain_fraction": args.retain_fraction,
        "allocation_mode": args.allocation_mode,
        "utility_condition": args.utility_condition,
        "utility_metric": args.utility_metric,
        "k_neighbors": args.k_neighbors,
        "neighbor_metric": args.neighbor_metric,
        "weight_mode": args.weight_mode,
        "sigma": args.sigma,
        "calibration_samples": args.calibration_samples,
        "eval_samples": args.eval_samples,
        "hidden_dim": loaded.hidden_dim,
        "num_layers_total": loaded.num_layers,
        "budget_dims": budget_dims,
        "achieved_keep_dims": int(allocation_df["k_l"].sum()),
        "utility_cache_dir": str(cache_dir),
    }, outdir / "metadata.json")

    torch.save({
        "local_projectors": local_projectors,
        "null_projectors": null_projectors,
        "allocation": allocation_df.to_dict(orient="records"),
    }, outdir / "projectors.pt")

    print(f"Saved results to: {outdir}")
    print(final_df.to_string(index=False))
    print("\nChosen allocation:")
    print(allocation_df.to_string(index=False))


if __name__ == "__main__":
    main()
