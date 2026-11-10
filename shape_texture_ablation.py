"""
Task-Specific Representational Directions: Shape vs Texture
=============================================================

Double dissociation across 5 pretrained vision models:
  - ViT-B/16 (ImageNet-1k)
  - ViT-B/16 (ImageNet-21k → 1k)
  - DeiT3-B/16
  - ViT-L/16
  - DINOv2 ViT-B/14

For each model at the penultimate layer:
  1. Train linear probes for shape and texture on Geirhos cue-conflict images
  2. Compute task-relevance matrices F_shape and F_texture (on train split)
  3. Ablation: remove top-k eigenvectors of F_shape → shape drops, texture spared
     (and vice versa) — evaluated on held-out test split
  4. Perturbation: perturb along F_shape top eigenvector → shape flips before
     texture (and vice versa) — evaluated on held-out test split

Generates paper-ready figures (PDF, 300 dpi).

Usage:
    python task_eigendistortions_vit.py \
        --stimuli-dir ./data/geirhos/style-transfer-preprocessed-512 \
        --output-dir ./task_eigendistortions_vit \
        --device cuda

Estimated runtime: ~20-30 min on A100 (5 models, 1 layer each, ~1200 images).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

import timm


# =====================================================================
# Config
# =====================================================================
GEIRHOS_CATEGORIES = [
    "airplane", "bear", "bicycle", "bird", "boat", "bottle", "car", "cat",
    "chair", "clock", "dog", "elephant", "keyboard", "knife", "oven", "truck",
]
CAT_TO_IDX = {c: i for i, c in enumerate(GEIRHOS_CATEGORIES)}
NUM_CLASSES = len(GEIRHOS_CATEGORIES)

# timm models
TIMM_MODELS = {
    "vit_b_16":       "vit_base_patch16_224",
    "vit_b_16_in21k": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "deit3_b_16":     "deit3_base_patch16_224",
    "vit_l_16":       "vit_large_patch16_224",
}

DISPLAY_NAMES = {
    "vit_base_patch16_224":                      "ViT-B/16",
    "vit_base_patch16_224.augreg_in21k_ft_in1k": "ViT-B/16\n(IN-21k)",
    "deit3_base_patch16_224":                    "DeiT3-B/16",
    "vit_large_patch16_224":                     "ViT-L/16",
    "dinov2_vitb14":                             "DINOv2\nViT-B/14",
}

SHORT_NAMES = {
    "vit_base_patch16_224":                      "ViT-B/16",
    "vit_base_patch16_224.augreg_in21k_ft_in1k": "ViT-B/16 (IN-21k)",
    "deit3_base_patch16_224":                    "DeiT3-B/16",
    "vit_large_patch16_224":                     "ViT-L/16",
    "dinov2_vitb14":                             "DINOv2 ViT-B/14",
}

# All model keys in display order
ALL_MODEL_KEYS = list(TIMM_MODELS.values()) + ["dinov2_vitb14"]


# =====================================================================
# Dataset
# =====================================================================
def parse_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    stem = Path(filename).stem
    for cat in sorted(GEIRHOS_CATEGORIES, key=len, reverse=True):
        if stem.startswith(cat):
            remainder = stem[len(cat):]
            match = re.match(r"\d+[-_](.*)", remainder)
            if match:
                texture_part = match.group(1)
                for tcat in sorted(GEIRHOS_CATEGORIES, key=len, reverse=True):
                    if texture_part.startswith(tcat):
                        return cat, tcat
            return cat, None
    return None, None


class GeirhosCueConflict(Dataset):
    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples: List[Tuple[str, int, int]] = []
        root = Path(root)
        for img_path in sorted(root.rglob("*.png")):
            s, t = parse_filename(img_path.name)
            if s is None or t is None or s == t:
                continue
            self.samples.append((str(img_path), CAT_TO_IDX[s], CAT_TO_IDX[t]))
        print(f"  Loaded {len(self.samples)} cue-conflict images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, si, ti = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, si, ti


# =====================================================================
# Model wrappers
# =====================================================================
class TimmViTWrapper:
    """Wrapper for timm ViT models."""
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.display_name = SHORT_NAMES.get(model_name, model_name)
        print(f"  Loading {model_name} ({self.display_name})...")
        self.model = timm.create_model(model_name, pretrained=True).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n_blocks = len(self.model.blocks)
        self.hidden_dim = self.model.num_features
        self.device = device
        print(f"    {self.n_blocks} blocks, dim={self.hidden_dim}")

    @torch.no_grad()
    def extract_cls(self, images: torch.Tensor, layer_idx: int) -> torch.Tensor:
        captured = {}
        def hook_fn(mod, inp, out):
            captured["act"] = out[0] if isinstance(out, tuple) else out
        hook = self.model.blocks[layer_idx].register_forward_hook(hook_fn)
        feats = []
        for i in range(0, len(images), 32):
            batch = images[i:i+32].to(self.device)
            self.model(batch)
            feats.append(captured["act"][:, 0, :].float().cpu())
        hook.remove()
        return torch.cat(feats, 0)

    def penultimate_layer(self) -> int:
        return self.n_blocks - 2

    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()
        import gc; gc.collect()


class DINOv2Wrapper:
    """Wrapper for DINOv2 models loaded via torch.hub."""
    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cuda"):
        self.model_name = model_name
        self.display_name = SHORT_NAMES.get(model_name, model_name)
        print(f"  Loading {model_name} ({self.display_name})...")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n_blocks = len(self.model.blocks)
        self.hidden_dim = self.model.embed_dim
        self.device = device
        self.final_norm = getattr(self.model, "norm", None)
        print(f"    {self.n_blocks} blocks, dim={self.hidden_dim}")

    @torch.no_grad()
    def extract_cls(self, images: torch.Tensor, layer_idx: int) -> torch.Tensor:
        captured = {}
        def hook_fn(mod, inp, out):
            captured["act"] = out[0] if isinstance(out, tuple) else out
        hook = self.model.blocks[layer_idx].register_forward_hook(hook_fn)
        feats = []
        for i in range(0, len(images), 32):
            batch = images[i:i+32].to(self.device)
            self.model(batch)
            cls = captured["act"][:, 0, :].float()
            # Apply final norm only at last block
            if layer_idx == self.n_blocks - 1 and self.final_norm is not None:
                cls = self.final_norm(cls)
            feats.append(cls.cpu())
        hook.remove()
        return torch.cat(feats, 0)

    def penultimate_layer(self) -> int:
        return self.n_blocks - 2

    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()
        import gc; gc.collect()


def load_model(model_key: str, device: str):
    """Factory: return the appropriate wrapper for a model key."""
    if model_key == "dinov2_vitb14":
        return DINOv2Wrapper("dinov2_vitb14", device)
    else:
        return TimmViTWrapper(model_key, device)


# =====================================================================
# Linear probe
# =====================================================================
class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


def train_probe(features: torch.Tensor, labels: torch.Tensor,
                n_classes: int, epochs: int = 200, lr: float = 0.005,
                device: str = "cuda") -> Tuple[LinearProbe, float]:
    probe = LinearProbe(features.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N = features.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(N)
        correct = 0
        for i in range(0, N, 128):
            idx = perm[i:i+128]
            x, y = features[idx].to(device), labels[idx].to(device)
            opt.zero_grad()
            logits = probe(x)
            F_torch.cross_entropy(logits, y).backward()
            opt.step()
            correct += (logits.detach().argmax(1) == y).sum().item()
        sched.step()
    acc = correct / N
    probe.eval()
    return probe, acc


# =====================================================================
# Task-relevance matrix F
# =====================================================================
def compute_F_matrix(probe: LinearProbe, features: torch.Tensor,
                     labels: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    probe = probe.to(device).eval()
    D = features.shape[1]
    N = features.shape[0]
    F_mat = torch.zeros(D, D, dtype=torch.float64, device="cpu")
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for i in range(0, N, 128):
        x = features[i:i+128].clone().to(device).requires_grad_(True)
        y = labels[i:i+128].to(device)
        loss = criterion(probe(x), y)
        grad = torch.autograd.grad(loss, x)[0]
        g = grad.detach().double().cpu()
        F_mat += g.T @ g
    F_mat /= float(N)
    F_mat = 0.5 * (F_mat + F_mat.T)
    return F_mat


def psd_effective_rank(matrix: torch.Tensor) -> Tuple[float, dict]:
    """Participation-ratio rank for a theoretically PSD matrix.

    Work in float64 and clip negative eigensolver roundoff before evaluating
    (sum lambda)^2 / sum lambda^2.  Without the clipping, cancellation from
    small negative eigenvalues can produce impossible effective ranks below 1.
    """
    matrix64 = matrix.detach().to(device="cpu", dtype=torch.float64)
    matrix64 = 0.5 * (matrix64 + matrix64.T)
    eigvals = torch.linalg.eigvalsh(matrix64)
    clipped = eigvals.clamp_min(0.0)
    denominator = clipped.square().sum()
    if denominator <= 0:
        effective_rank = 0.0
    else:
        effective_rank = (
            clipped.sum().square() / denominator
        ).item()
    diagnostics = {
        "min_eigenvalue_float64": eigvals.min().item(),
        "max_eigenvalue_float64": eigvals.max().item(),
        "negative_eigenvalue_count": int((eigvals < 0).sum().item()),
        "negative_eigenvalue_mass": eigvals[eigvals < 0].abs().sum().item(),
    }
    return effective_rank, diagnostics


# =====================================================================
# Ablation
# =====================================================================
def run_ablation(feats, shape_labels, texture_labels,
                 shape_probe, texture_probe,
                 F_shape, F_texture,
                 shape_acc_base, texture_acc_base,
                 n_remove_list=[1, 2, 5, 10, 25, 50, 100, 200]):
    D = feats.shape[1]
    evals_s, evecs_s = torch.linalg.eigh(F_shape.float())
    V_shape = evecs_s.flip(1)
    evals_t, evecs_t = torch.linalg.eigh(F_texture.float())
    V_texture = evecs_t.flip(1)

    sp = shape_probe.cpu().eval()
    tp = texture_probe.cpu().eval()

    results = []
    for k in n_remove_list:
        if k >= D:
            continue
        # Remove top-k of F_shape
        P = torch.eye(D) - V_shape[:, :k] @ V_shape[:, :k].T
        feats_abl = feats @ P.T
        with torch.no_grad():
            sa = (sp(feats_abl).argmax(1) == shape_labels).float().mean().item()
            ta = (tp(feats_abl).argmax(1) == texture_labels).float().mean().item()

        # Remove top-k of F_texture
        P_t = torch.eye(D) - V_texture[:, :k] @ V_texture[:, :k].T
        feats_abl_t = feats @ P_t.T
        with torch.no_grad():
            sa2 = (sp(feats_abl_t).argmax(1) == shape_labels).float().mean().item()
            ta2 = (tp(feats_abl_t).argmax(1) == texture_labels).float().mean().item()

        results.append({
            "n_removed": k,
            "remove_F_shape": {"shape_acc": sa, "texture_acc": ta},
            "remove_F_texture": {"shape_acc": sa2, "texture_acc": ta2},
        })
        print(f"      k={k:>3}: rm F_shape → shp {sa:.1%} tex {ta:.1%}  |  "
              f"rm F_tex → shp {sa2:.1%} tex {ta2:.1%}")

    return results


# =====================================================================
# Perturbation
# =====================================================================
def run_perturbation(feats, shape_labels, texture_labels,
                     shape_probe, texture_probe,
                     F_shape, F_texture,
                     alphas=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]):
    _, evecs_s = torch.linalg.eigh(F_shape.float())
    v_shape = evecs_s[:, -1]
    v_shape = v_shape / (v_shape.norm() + 1e-8)

    _, evecs_t = torch.linalg.eigh(F_texture.float())
    v_texture = evecs_t[:, -1]
    v_texture = v_texture / (v_texture.norm() + 1e-8)

    feat_norm = feats.norm(dim=1).mean().item()
    sp = shape_probe.cpu().eval()
    tp = texture_probe.cpu().eval()

    with torch.no_grad():
        base_sp = sp(feats).argmax(1)
        base_tp = tp(feats).argmax(1)

    results = {"alphas": alphas, "along_F_shape": [], "along_F_texture": []}

    for alpha in alphas:
        s_alpha = alpha * feat_norm / 100.0

        fp = feats + s_alpha * v_shape.unsqueeze(0)
        with torch.no_grad():
            sf = (sp(fp).argmax(1) != base_sp).float().mean().item()
            tf = (tp(fp).argmax(1) != base_tp).float().mean().item()
        results["along_F_shape"].append({"alpha": alpha, "shape_flip": sf, "texture_flip": tf})

        fp2 = feats + s_alpha * v_texture.unsqueeze(0)
        with torch.no_grad():
            sf2 = (sp(fp2).argmax(1) != base_sp).float().mean().item()
            tf2 = (tp(fp2).argmax(1) != base_tp).float().mean().item()
        results["along_F_texture"].append({"alpha": alpha, "shape_flip": sf2, "texture_flip": tf2})

    return results


# =====================================================================
# Paper-ready figures
# =====================================================================
SHAPE_COLOR = "#2E86AB"
TEXTURE_COLOR = "#D85A30"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "figure.dpi": 300,
})


def plot_ablation_all_models(all_results: Dict[str, dict], output_dir: str):
    """2 rows × N columns. Top: remove F_shape dirs. Bottom: remove F_texture dirs."""
    model_names = list(all_results.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(2, n_models, figsize=(2.5 * n_models, 4.2),
                             sharey=True, sharex=True)
    if n_models == 1:
        axes = axes[:, np.newaxis]

    for col, mname in enumerate(model_names):
        abls = all_results[mname]["ablations"]
        ns = [a["n_removed"] for a in abls]
        display = DISPLAY_NAMES.get(mname, mname)

        ax = axes[0, col]
        ax.plot(ns, [a["remove_F_shape"]["shape_acc"] for a in abls],
                "o-", color=SHAPE_COLOR, label="Shape acc", lw=1.5, ms=3)
        ax.plot(ns, [a["remove_F_shape"]["texture_acc"] for a in abls],
                "s-", color=TEXTURE_COLOR, label="Texture acc", lw=1.5, ms=3)
        ax.set_title(display, fontsize=7.5, pad=4)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xscale("log")
        if col == 0:
            ax.set_ylabel("Accuracy", fontsize=8)
            ax.legend(fontsize=6, loc="lower left")

        ax = axes[1, col]
        ax.plot(ns, [a["remove_F_texture"]["shape_acc"] for a in abls],
                "o-", color=SHAPE_COLOR, lw=1.5, ms=3)
        ax.plot(ns, [a["remove_F_texture"]["texture_acc"] for a in abls],
                "s-", color=TEXTURE_COLOR, lw=1.5, ms=3)
        ax.set_xlabel("Dirs removed", fontsize=7)
        ax.set_xscale("log")
        if col == 0:
            ax.set_ylabel("Accuracy", fontsize=8)

    fig.text(0.008, 0.75, "Remove\nF_shape\ndirs", va="center", ha="center",
             fontsize=6.5, fontstyle="italic", color="#555555")
    fig.text(0.008, 0.3, "Remove\nF_texture\ndirs", va="center", ha="center",
             fontsize=6.5, fontstyle="italic", color="#555555")

    plt.tight_layout(rect=[0.035, 0, 1, 1])
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir, f"ablation_double_dissociation.{fmt}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved ablation figure")


def plot_perturbation_all_models(all_results: Dict[str, dict], output_dir: str):
    """2 rows × N columns. Top: perturb along F_shape. Bottom: perturb along F_texture."""
    model_names = list(all_results.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(2, n_models, figsize=(2.5 * n_models, 4.2),
                             sharey=True, sharex=True)
    if n_models == 1:
        axes = axes[:, np.newaxis]

    for col, mname in enumerate(model_names):
        pert = all_results[mname]["perturbations"]
        alphas = pert["alphas"]
        display = DISPLAY_NAMES.get(mname, mname)

        ax = axes[0, col]
        ax.plot(alphas, [p["shape_flip"] for p in pert["along_F_shape"]],
                "o-", color=SHAPE_COLOR, label="Shape flips", lw=1.5, ms=3)
        ax.plot(alphas, [p["texture_flip"] for p in pert["along_F_shape"]],
                "s-", color=TEXTURE_COLOR, label="Texture flips", lw=1.5, ms=3)
        ax.set_title(display, fontsize=7.5, pad=4)
        ax.set_ylim(-0.05, 1.05)
        if col == 0:
            ax.set_ylabel("Flip rate", fontsize=8)
            ax.legend(fontsize=6, loc="upper left")

        ax = axes[1, col]
        ax.plot(alphas, [p["shape_flip"] for p in pert["along_F_texture"]],
                "o-", color=SHAPE_COLOR, lw=1.5, ms=3)
        ax.plot(alphas, [p["texture_flip"] for p in pert["along_F_texture"]],
                "s-", color=TEXTURE_COLOR, lw=1.5, ms=3)
        ax.set_xlabel("Perturbation α", fontsize=7)
        if col == 0:
            ax.set_ylabel("Flip rate", fontsize=8)

    fig.text(0.008, 0.75, "Along\nF_shape\ndir", va="center", ha="center",
             fontsize=6.5, fontstyle="italic", color="#555555")
    fig.text(0.008, 0.3, "Along\nF_texture\ndir", va="center", ha="center",
             fontsize=6.5, fontstyle="italic", color="#555555")

    plt.tight_layout(rect=[0.035, 0, 1, 1])
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir, f"perturbation_double_dissociation.{fmt}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved perturbation figure")


def plot_ablation_overlay(all_results: Dict[str, dict], output_dir: str):
    """
    Overlay figure: all models on the same axes.
    Panel A: remove F_shape dirs → shape acc (solid) vs texture acc (dashed) per model.
    Panel B: remove F_texture dirs → same.
    Each model gets a distinct color; shape=solid, texture=dashed.
    """
    model_names = list(all_results.keys())

    # One color per model
    cmap = plt.cm.get_cmap("tab10", len(model_names))
    model_colors = {m: cmap(i) for i, m in enumerate(model_names)}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)

    for ax, remove_key, title in [
        (axes[0], "remove_F_shape", "Remove shape-sensitive directions"),
        (axes[1], "remove_F_texture", "Remove texture-sensitive directions"),
    ]:
        for mname in model_names:
            abls = all_results[mname]["ablations"]
            ns = [a["n_removed"] for a in abls]
            c = model_colors[mname]
            label_s = SHORT_NAMES.get(mname, mname)

            ax.plot(ns, [a[remove_key]["shape_acc"] for a in abls],
                    "o-", color=c, lw=1.5, ms=3, label=f"{label_s} (shape)")
            ax.plot(ns, [a[remove_key]["texture_acc"] for a in abls],
                    "s--", color=c, lw=1.2, ms=3, alpha=0.6)

        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Directions removed", fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.15)

    axes[0].set_ylabel("Accuracy", fontsize=8)

    # Build custom legend: one entry per model + shape/texture line style
    from matplotlib.lines import Line2D
    model_handles = [Line2D([0], [0], color=model_colors[m], lw=1.5,
                            label=SHORT_NAMES.get(m, m)) for m in model_names]
    style_handles = [
        Line2D([0], [0], color="gray", lw=1.5, ls="-", marker="o", ms=3,
               label="Shape acc"),
        Line2D([0], [0], color="gray", lw=1.2, ls="--", marker="s", ms=3,
               alpha=0.6, label="Texture acc"),
    ]
    axes[1].legend(handles=model_handles + style_handles, fontsize=5.5,
                   loc="lower left", ncol=1)

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir, f"ablation_overlay.{fmt}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved ablation overlay figure")


def plot_perturbation_overlay(all_results: Dict[str, dict], output_dir: str):
    """
    Overlay figure: all models on the same axes.
    Panel A: perturb along F_shape dir → shape flips (solid) vs texture flips (dashed).
    Panel B: perturb along F_texture dir → same.
    """
    model_names = list(all_results.keys())

    cmap = plt.cm.get_cmap("tab10", len(model_names))
    model_colors = {m: cmap(i) for i, m in enumerate(model_names)}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)

    for ax, along_key, title in [
        (axes[0], "along_F_shape", "Perturb along shape-sensitive direction"),
        (axes[1], "along_F_texture", "Perturb along texture-sensitive direction"),
    ]:
        for mname in model_names:
            pert = all_results[mname]["perturbations"]
            alphas = pert["alphas"]
            c = model_colors[mname]
            label_s = SHORT_NAMES.get(mname, mname)

            ax.plot(alphas, [p["shape_flip"] for p in pert[along_key]],
                    "o-", color=c, lw=1.5, ms=3, label=f"{label_s} (shape)")
            ax.plot(alphas, [p["texture_flip"] for p in pert[along_key]],
                    "s--", color=c, lw=1.2, ms=3, alpha=0.6)

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Perturbation α", fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.15)

    axes[0].set_ylabel("Flip rate", fontsize=8)

    from matplotlib.lines import Line2D
    model_handles = [Line2D([0], [0], color=model_colors[m], lw=1.5,
                            label=SHORT_NAMES.get(m, m)) for m in model_names]
    style_handles = [
        Line2D([0], [0], color="gray", lw=1.5, ls="-", marker="o", ms=3,
               label="Shape flips"),
        Line2D([0], [0], color="gray", lw=1.2, ls="--", marker="s", ms=3,
               alpha=0.6, label="Texture flips"),
    ]
    axes[1].legend(handles=model_handles + style_handles, fontsize=5.5,
                   loc="upper left", ncol=1)

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir, f"perturbation_overlay.{fmt}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved perturbation overlay figure")


def plot_summary_bar(all_results: Dict[str, dict], output_dir: str):
    """Probe accuracies and effective ranks for all models."""
    model_names = list(all_results.keys())
    n = len(model_names)
    x = np.arange(n)
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.5))

    ax = axes[0]
    ax.bar(x - w/2, [all_results[m]["shape_acc"] for m in model_names],
           w, color=SHAPE_COLOR, label="Shape", edgecolor="white", lw=0.5)
    ax.bar(x + w/2, [all_results[m]["texture_acc"] for m in model_names],
           w, color=TEXTURE_COLOR, label="Texture", edgecolor="white", lw=0.5)
    ax.set_ylabel("Test accuracy", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in model_names], fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Probe accuracy", fontsize=9, fontweight="bold")

    ax = axes[1]
    ax.bar(x - w/2, [all_results[m]["shape_eff_rank"] for m in model_names],
           w, color=SHAPE_COLOR, label="F_shape", edgecolor="white", lw=0.5)
    ax.bar(x + w/2, [all_results[m]["texture_eff_rank"] for m in model_names],
           w, color=TEXTURE_COLOR, label="F_texture", edgecolor="white", lw=0.5)
    ax.set_ylabel("Effective rank", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in model_names], fontsize=6)
    ax.legend(fontsize=7)
    ax.set_title("Task subspace dimensionality", fontsize=9, fontweight="bold")

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir, f"model_summary.{fmt}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved summary figure")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli-dir", required=True,
                        help="Path to Geirhos style-transfer-preprocessed-512")
    parser.add_argument("--output-dir", default="./task_eigendistortions_vit")
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load dataset ──
    # Use 224 transform for timm models; DINOv2 also uses 224
    print("Loading Geirhos cue-conflict stimuli...")
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = GeirhosCueConflict(args.stimuli_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    all_images, all_shape, all_texture = [], [], []
    for imgs, sy, ty in loader:
        all_images.append(imgs)
        all_shape.append(sy)
        all_texture.append(ty)
    all_images = torch.cat(all_images, 0)
    all_shape = torch.cat(all_shape, 0)
    all_texture = torch.cat(all_texture, 0)
    N = len(all_shape)

    # ── Train/test split ──
    n_test = int(N * args.test_fraction)
    n_train = N - n_test
    perm = torch.randperm(N)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    shape_train, shape_test = all_shape[train_idx], all_shape[test_idx]
    texture_train, texture_test = all_texture[train_idx], all_texture[test_idx]
    print(f"  N={N}, train={n_train}, test={n_test}")

    # ── Run per model ──
    all_results: Dict[str, dict] = {}

    for model_key in ALL_MODEL_KEYS:
        print(f"\n{'='*70}")
        print(f"  {SHORT_NAMES[model_key]}")
        print(f"{'='*70}")

        wrapper = load_model(model_key, device)
        layer_idx = wrapper.penultimate_layer()
        print(f"  Using layer {layer_idx} (penultimate of {wrapper.n_blocks})")

        # Extract features
        print("  Extracting CLS features...")
        feats = wrapper.extract_cls(all_images, layer_idx)
        feats_train = feats[train_idx]
        feats_test = feats[test_idx]

        # Train probes on TRAIN data
        print("  Training shape probe...")
        shape_probe, shape_acc_train = train_probe(
            feats_train, shape_train, NUM_CLASSES,
            epochs=args.probe_epochs, device=device)
        sp_cpu = shape_probe.cpu().eval()
        with torch.no_grad():
            shape_acc = (sp_cpu(feats_test).argmax(1) == shape_test).float().mean().item()
        print(f"    Shape: train={shape_acc_train:.1%}, test={shape_acc:.1%}")

        print("  Training texture probe...")
        texture_probe, texture_acc_train = train_probe(
            feats_train, texture_train, NUM_CLASSES,
            epochs=args.probe_epochs, device=device)
        tp_cpu = texture_probe.cpu().eval()
        with torch.no_grad():
            texture_acc = (tp_cpu(feats_test).argmax(1) == texture_test).float().mean().item()
        print(f"    Texture: train={texture_acc_train:.1%}, test={texture_acc:.1%}")

        # Compute F matrices on TRAIN data
        print("  Computing F_shape...")
        F_shape = compute_F_matrix(shape_probe, feats_train, shape_train, device=device)
        print("  Computing F_texture...")
        F_texture = compute_F_matrix(texture_probe, feats_train, texture_train, device=device)

        # Effective ranks
        shape_er, shape_er_diagnostics = psd_effective_rank(F_shape)
        texture_er, texture_er_diagnostics = psd_effective_rank(F_texture)
        print(f"    F_shape eff rank: {shape_er:.1f}")
        print(f"    F_texture eff rank: {texture_er:.1f}")

        # Ablation on TEST data
        print("  Running ablation (test set)...")
        ablations = run_ablation(
            feats_test, shape_test, texture_test,
            shape_probe, texture_probe, F_shape, F_texture,
            shape_acc, texture_acc)

        # Perturbation on TEST data
        print("  Running perturbation (test set)...")
        perturbations = run_perturbation(
            feats_test, shape_test, texture_test,
            shape_probe, texture_probe, F_shape, F_texture)

        all_results[model_key] = {
            "layer_idx": layer_idx,
            "shape_acc": shape_acc,
            "texture_acc": texture_acc,
            "shape_acc_train": shape_acc_train,
            "texture_acc_train": texture_acc_train,
            "shape_eff_rank": shape_er,
            "texture_eff_rank": texture_er,
            "shape_eff_rank_diagnostics": shape_er_diagnostics,
            "texture_eff_rank_diagnostics": texture_er_diagnostics,
            "ablations": ablations,
            "perturbations": perturbations,
        }

        # Free GPU memory before loading next model
        del feats, feats_train, feats_test
        del shape_probe, texture_probe, F_shape, F_texture
        wrapper.cleanup()

    # ── Generate figures ──
    print("\nGenerating paper-ready figures...")
    plot_ablation_all_models(all_results, args.output_dir)
    plot_perturbation_all_models(all_results, args.output_dir)
    plot_ablation_overlay(all_results, args.output_dir)
    plot_perturbation_overlay(all_results, args.output_dir)
    plot_summary_bar(all_results, args.output_dir)

    # ── Save JSON ──
    json_results = {}
    for mname, r in all_results.items():
        json_results[mname] = {
            "display_name": SHORT_NAMES[mname],
            "layer_idx": r["layer_idx"],
            "shape_acc_test": r["shape_acc"],
            "texture_acc_test": r["texture_acc"],
            "shape_acc_train": r["shape_acc_train"],
            "texture_acc_train": r["texture_acc_train"],
            "shape_eff_rank": r["shape_eff_rank"],
            "texture_eff_rank": r["texture_eff_rank"],
            "shape_eff_rank_diagnostics": r["shape_eff_rank_diagnostics"],
            "texture_eff_rank_diagnostics": r["texture_eff_rank_diagnostics"],
            "ablations": r["ablations"],
            "perturbations": r["perturbations"],
        }

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nAll results saved to {args.output_dir}/")
    print(f"  ablation_double_dissociation.pdf  — per-model ablation (2×N grid)")
    print(f"  ablation_overlay.pdf              — all models overlaid (1×2)")
    print(f"  perturbation_double_dissociation.pdf — per-model perturbation (2×N grid)")
    print(f"  perturbation_overlay.pdf          — all models overlaid (1×2)")
    print(f"  model_summary.pdf — probe accuracy & effective rank")
    print(f"  results.json — all numerical results")


if __name__ == "__main__":
    main()
