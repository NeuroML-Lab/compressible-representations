"""
Continual learning benchmark: Split-CIFAR10 and Split-CIFAR100.

Methods:
    naive  — sequential fine-tuning, no protection (lower bound)
    gpm    — Gradient Projection Memory (Saha et al. 2021)
    fgpm   — F-GPM (ours): Jacobian-based task-relevant subspace protection
    mixed  — GPM on early (conv) layers + FGPM on late (fc) layers
    mixed-rev — FGPM on early layers + GPM on late layers (control)
    joint  — train on all tasks simultaneously (upper bound)

Architectures (each with optional BatchNorm variant via --bn flag):
    alexnet   — conv(3→32→64→128) + fc(2048→256→256)
    alexnet-l — conv(3→64→128→256) + fc(→2048→2048) [matches GPM paper dims]
    lenet     — conv(3→20→50) + fc(→800→500)
    resnet18  — torchvision ResNet18 (adapted for 32x32)
    vgg11     — simplified VGG11 for 32x32

Task-incremental setup (multi-head, task-id given at test time).

Usage:
    # Full sweep (no BN):
    python cl_benchmark_v3.py --benchmark both --mode sweep \
        --arch alexnet,alexnet-l,lenet --epochs 1,10,50

    # With BatchNorm:
    python cl_benchmark_v3.py --benchmark cifar10 --mode sweep \
        --arch alexnet,alexnet-l --epochs 10 --bn

    # Specific run with mixed methods:
    python cl_benchmark_v3.py --benchmark cifar10 --mode single \
        --methods naive,gpm,fgpm,mixed,mixed-rev,joint --epochs 10 --arch alexnet-l
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# =====================================================================
# Reproducibility
# =====================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =====================================================================
# Architecture helpers
# =====================================================================
def _maybe_bn(use_bn: bool, num_features: int):
    """Return a BatchNorm layer or Identity."""
    return nn.BatchNorm2d(num_features) if use_bn else nn.Identity()

def _maybe_bn1d(use_bn: bool, num_features: int):
    return nn.BatchNorm1d(num_features) if use_bn else nn.Identity()


# =====================================================================
# Architectures
# =====================================================================
class AlexNetCIFAR(nn.Module):
    """AlexNet-like CNN for 32×32.
    conv1(3→32) → [BN] → ReLU → Pool
    conv2(32→64) → [BN] → ReLU → Pool
    conv3(64→128) → [BN] → ReLU → Pool
    fc1(2048→256) → [BN1d] → ReLU → drop
    fc2(256→256) → [BN1d] → ReLU → drop
    """
    arch_name = "alexnet"

    def __init__(self, num_tasks, classes_per_task, use_bn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = _maybe_bn(use_bn, 32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = _maybe_bn(use_bn, 64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = _maybe_bn(use_bn, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn_fc1 = _maybe_bn1d(use_bn, 256)
        self.fc2 = nn.Linear(256, 256)
        self.bn_fc2 = _maybe_bn1d(use_bn, 256)
        self.heads = nn.ModuleList(
            [nn.Linear(256, classes_per_task) for _ in range(num_tasks)])
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.protected_layer_names = ["conv1", "conv2", "conv3", "fc1", "fc2"]
        self.early_layers = ["conv1", "conv2", "conv3"]
        self.late_layers = ["fc1", "fc2"]
        self.use_bn = use_bn

    def features(self, x):
        x = self.pool(F_torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(F_torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(F_torch.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = self.drop(F_torch.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop(F_torch.relu(self.bn_fc2(self.fc2(x))))
        return x

    def forward(self, x, task_id: int):
        return self.heads[task_id](self.features(x))


class AlexNetLarge(nn.Module):
    """Larger AlexNet matching GPM paper dimensions.
    conv1(3→64, k=4) → [BN] → ReLU → Drop(0.2) → Pool
    conv2(64→128, k=3) → [BN] → ReLU → Drop(0.2) → Pool
    conv3(128→256, k=2) → [BN] → ReLU → Drop(0.5) → Pool
    fc1(256*s*s → 2048) → [BN1d] → ReLU → Drop(0.5)
    fc2(2048 → 2048) → [BN1d] → ReLU → Drop(0.5)
    """
    arch_name = "alexnet-l"

    def __init__(self, num_tasks, classes_per_task, input_size=32, use_bn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 4)
        self.bn1 = _maybe_bn(use_bn, 64)
        s = input_size - 4 + 1; s = s // 2
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.bn2 = _maybe_bn(use_bn, 128)
        s = s - 3 + 1; s = s // 2
        self.conv3 = nn.Conv2d(128, 256, 2)
        self.bn3 = _maybe_bn(use_bn, 256)
        s = s - 2 + 1; s = s // 2
        self.smid = s
        self.fc1 = nn.Linear(256 * s * s, 2048)
        self.bn_fc1 = _maybe_bn1d(use_bn, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn_fc2 = _maybe_bn1d(use_bn, 2048)
        self.pool = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.5)
        self.heads = nn.ModuleList(
            [nn.Linear(2048, classes_per_task) for _ in range(num_tasks)])
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.protected_layer_names = ["conv1", "conv2", "conv3", "fc1", "fc2"]
        self.early_layers = ["conv1", "conv2", "conv3"]
        self.late_layers = ["fc1", "fc2"]
        self.use_bn = use_bn

    def features(self, x):
        x = self.pool(self.drop1(F_torch.relu(self.bn1(self.conv1(x)))))
        x = self.pool(self.drop1(F_torch.relu(self.bn2(self.conv2(x)))))
        x = self.pool(self.drop2(F_torch.relu(self.bn3(self.conv3(x)))))
        x = x.flatten(1)
        x = self.drop2(F_torch.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop2(F_torch.relu(self.bn_fc2(self.fc2(x))))
        return x

    def forward(self, x, task_id: int):
        return self.heads[task_id](self.features(x))


class LeNetCIFAR(nn.Module):
    """LeNet-like: conv(3→20→50) + fc(1250→800→500)"""
    arch_name = "lenet"

    def __init__(self, num_tasks, classes_per_task, use_bn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.bn1 = _maybe_bn(use_bn, 20)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.bn2 = _maybe_bn(use_bn, 50)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(50 * 5 * 5, 800)
        self.bn_fc1 = _maybe_bn1d(use_bn, 800)
        self.fc2 = nn.Linear(800, 500)
        self.bn_fc2 = _maybe_bn1d(use_bn, 500)
        self.heads = nn.ModuleList(
            [nn.Linear(500, classes_per_task) for _ in range(num_tasks)])
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.protected_layer_names = ["conv1", "conv2", "fc1", "fc2"]
        self.early_layers = ["conv1", "conv2"]
        self.late_layers = ["fc1", "fc2"]
        self.use_bn = use_bn

    def features(self, x):
        x = self.pool(F_torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(F_torch.relu(self.bn2(self.conv2(x))))
        x = x.flatten(1)
        x = self.drop(F_torch.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop(F_torch.relu(self.bn_fc2(self.fc2(x))))
        return x

    def forward(self, x, task_id: int):
        return self.heads[task_id](self.features(x))


class VGG11CIFAR(nn.Module):
    """Simplified VGG11 for 32×32.
    conv blocks: 64 → 128 → 256×2 → 512×2
    fc: 512*1*1 → 512 → 512
    """
    arch_name = "vgg11"

    def __init__(self, num_tasks, classes_per_task, use_bn=False):
        super().__init__()
        # Block 1: 32→16
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = _maybe_bn(use_bn, 64)
        # Block 2: 16→8
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = _maybe_bn(use_bn, 128)
        # Block 3: 8→4
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = _maybe_bn(use_bn, 256)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = _maybe_bn(use_bn, 256)
        # Block 4: 4→2
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn5 = _maybe_bn(use_bn, 512)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn6 = _maybe_bn(use_bn, 512)
        # Block 5: 2→1
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn7 = _maybe_bn(use_bn, 512)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn8 = _maybe_bn(use_bn, 512)

        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 512)
        self.bn_fc1 = _maybe_bn1d(use_bn, 512)
        self.fc2 = nn.Linear(512, 512)
        self.bn_fc2 = _maybe_bn1d(use_bn, 512)
        self.heads = nn.ModuleList(
            [nn.Linear(512, classes_per_task) for _ in range(num_tasks)])
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.protected_layer_names = [
            "conv1", "conv2", "conv3", "conv4", "conv5", "conv6",
            "conv7", "conv8", "fc1", "fc2"]
        self.early_layers = [
            "conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv8"]
        self.late_layers = ["fc1", "fc2"]
        self.use_bn = use_bn

    def features(self, x):
        x = self.pool(F_torch.relu(self.bn1(self.conv1(x))))       # 32→16
        x = self.pool(F_torch.relu(self.bn2(self.conv2(x))))       # 16→8
        x = F_torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(F_torch.relu(self.bn4(self.conv4(x))))       # 8→4
        x = F_torch.relu(self.bn5(self.conv5(x)))
        x = self.pool(F_torch.relu(self.bn6(self.conv6(x))))       # 4→2
        x = F_torch.relu(self.bn7(self.conv7(x)))
        x = self.pool(F_torch.relu(self.bn8(self.conv8(x))))       # 2→1
        x = x.flatten(1)
        x = self.drop(F_torch.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop(F_torch.relu(self.bn_fc2(self.fc2(x))))
        return x

    def forward(self, x, task_id: int):
        return self.heads[task_id](self.features(x))


class ResNet18CIFAR(nn.Module):
    """ResNet18 adapted for 32×32 (no initial maxpool, smaller first conv).
    Protected layers: conv1 + all conv layers in blocks + fc.
    """
    arch_name = "resnet18"

    def __init__(self, num_tasks, classes_per_task, use_bn=False):
        super().__init__()
        self.use_bn = use_bn

        # Initial conv
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=not use_bn)
        self.bn1 = _maybe_bn(use_bn, 64)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1, use_bn=use_bn)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, use_bn=use_bn)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, use_bn=use_bn)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, use_bn=use_bn)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 512)
        self.bn_fc1 = _maybe_bn1d(use_bn, 512)
        self.heads = nn.ModuleList(
            [nn.Linear(512, classes_per_task) for _ in range(num_tasks)])
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task

        # Collect all conv layer names for protection
        self.protected_layer_names = ["conv1", "fc1"]
        self.early_layers = ["conv1"]
        self.late_layers = ["fc1"]

        # Add block conv layers
        for li, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4], 1):
            for bi, block in enumerate(layer):
                for ci, (name, mod) in enumerate(block.named_modules()):
                    if isinstance(mod, nn.Conv2d) and mod is not self.conv1:
                        attr_name = f"block{li}_{bi}_{name.replace('.', '_')}"
                        setattr(self, attr_name, mod)  # register for getattr access
                        self.protected_layer_names.append(attr_name)
                        self.early_layers.append(attr_name)

    @staticmethod
    def _make_layer(in_ch, out_ch, num_blocks, stride, use_bn):
        layers = []
        layers.append(_BasicBlock(in_ch, out_ch, stride, use_bn))
        for _ in range(1, num_blocks):
            layers.append(_BasicBlock(out_ch, out_ch, 1, use_bn))
        return nn.Sequential(*layers)

    def features(self, x):
        x = F_torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        x = F_torch.relu(self.bn_fc1(self.fc1(x)))
        return x

    def forward(self, x, task_id: int):
        return self.heads[task_id](self.features(x))


class _BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, use_bn):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=not use_bn)
        self.bn1 = _maybe_bn(use_bn, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=not use_bn)
        self.bn2 = _maybe_bn(use_bn, out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=not use_bn),
                _maybe_bn(use_bn, out_ch))

    def forward(self, x):
        out = F_torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F_torch.relu(out)


ARCH_REGISTRY = {
    "alexnet": AlexNetCIFAR,
    "alexnet-l": AlexNetLarge,
    "lenet": LeNetCIFAR,
    "vgg11": VGG11CIFAR,
    "resnet18": ResNet18CIFAR,
}


def build_model(arch_name, num_tasks, classes_per_task, use_bn=False):
    cls = ARCH_REGISTRY[arch_name]
    return cls(num_tasks=num_tasks, classes_per_task=classes_per_task, use_bn=use_bn)


# =====================================================================
# Data: Split-CIFAR10 and Split-CIFAR100
# =====================================================================
@dataclass
class Task:
    task_id: int
    classes: Tuple[int, ...]
    train_loader: DataLoader
    test_loader: DataLoader


class RemapLabels(torch.utils.data.Dataset):
    def __init__(self, subset, class_map):
        self.subset = subset
        self.class_map = class_map

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return x, self.class_map[int(y)]


def build_split_cifar(benchmark, data_root, batch_size, num_workers, seed):
    if benchmark == "cifar10":
        n_classes, n_tasks, cpt = 10, 5, 2
        DSClass = datasets.CIFAR10
    else:
        n_classes, n_tasks, cpt = 100, 20, 5
        DSClass = datasets.CIFAR100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_full = DSClass(data_root, train=True, download=True, transform=transform)
    test_full = DSClass(data_root, train=False, download=True, transform=transform)

    class_order = list(range(n_classes))
    train_targets = np.array(train_full.targets)
    test_targets = np.array(test_full.targets)
    rng = np.random.RandomState(seed)

    tasks = []
    for t in range(n_tasks):
        cls = tuple(class_order[t * cpt:(t + 1) * cpt])
        class_map = {c: i for i, c in enumerate(cls)}
        train_idx = np.where(np.isin(train_targets, cls))[0]
        test_idx = np.where(np.isin(test_targets, cls))[0]
        rng.shuffle(train_idx)

        tasks.append(Task(
            task_id=t, classes=cls,
            train_loader=DataLoader(
                RemapLabels(Subset(train_full, train_idx.tolist()), class_map),
                batch_size=batch_size, shuffle=True, num_workers=num_workers),
            test_loader=DataLoader(
                RemapLabels(Subset(test_full, test_idx.tolist()), class_map),
                batch_size=256, shuffle=False, num_workers=num_workers),
        ))
    return tasks, cpt


# =====================================================================
# Subspace projection core
# =====================================================================
class SubspaceProjector:
    def __init__(self, model, protected_layers=None):
        self.model = model
        self.protected_layers = protected_layers or model.protected_layer_names
        self.bases: Dict[str, Optional[torch.Tensor]] = {
            n: None for n in self.protected_layers}

    def project_gradients(self):
        for name in self.protected_layers:
            U = self.bases[name]
            if U is None:
                continue
            mod = getattr(self.model, name)
            w = mod.weight
            if w.grad is None:
                continue
            g = w.grad.data
            U_dev = U.to(g.device)
            if isinstance(mod, nn.Linear):
                g.sub_((g @ U_dev) @ U_dev.t())
            elif isinstance(mod, nn.Conv2d):
                out_c = g.shape[0]
                g_flat = g.reshape(out_c, -1)
                g_flat.sub_((g_flat @ U_dev) @ U_dev.t())
                w.grad.data.copy_(g_flat.reshape_as(g))

    def expand_basis(self, name, new_directions, tol=1e-6):
        U = self.bases[name]
        if U is None:
            Q, _ = torch.linalg.qr(new_directions)
            keep = Q.norm(dim=0) > tol
            self.bases[name] = Q[:, keep].contiguous()
            return
        in_dim = U.shape[0]
        if U.shape[1] >= in_dim:
            return
        residual = new_directions - U @ (U.t() @ new_directions)
        Q, R = torch.linalg.qr(residual)
        norms = torch.diag(R).abs() if R.dim() == 2 else torch.tensor([])
        if norms.numel() == 0:
            return
        keep = norms > tol
        Q_new = Q[:, keep]
        if Q_new.shape[1] == 0:
            return
        max_add = in_dim - U.shape[1]
        if Q_new.shape[1] > max_add:
            Q_new = Q_new[:, :max_add]
        self.bases[name] = torch.cat([U, Q_new.contiguous()], dim=1)

    def basis_ranks(self):
        return {n: (0 if U is None else U.shape[1]) for n, U in self.bases.items()}


# =====================================================================
# Hook-based activation / gradient capture
# =====================================================================
class InputCapturer:
    def __init__(self, model, layer_names, max_rows=5000):
        self.model = model
        self.layer_names = layer_names
        self.max_rows = max_rows
        self.buffers: Dict[str, List[torch.Tensor]] = {n: [] for n in layer_names}
        self._hooks = []
        self._counts = {n: 0 for n in layer_names}

    def _make_hook(self, name):
        mod = getattr(self.model, name)
        def hook(module, inputs, outputs):
            if self._counts[name] >= self.max_rows:
                return
            x = inputs[0].detach()
            if isinstance(module, nn.Linear):
                batch = x.reshape(-1, x.shape[-1]).cpu()
            elif isinstance(module, nn.Conv2d):
                unf = F_torch.unfold(x, module.kernel_size, module.dilation,
                                     module.padding, module.stride)
                batch = unf.transpose(1, 2).reshape(-1, unf.shape[1]).cpu()
            else:
                return
            self.buffers[name].append(batch)
            self._counts[name] += batch.shape[0]
        return hook

    def __enter__(self):
        for name in self.layer_names:
            mod = getattr(self.model, name)
            self._hooks.append(mod.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *a):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def stack(self, max_rows=None):
        mr = max_rows or self.max_rows
        out = {}
        for name, bufs in self.buffers.items():
            if not bufs:
                continue
            M = torch.cat(bufs, 0)
            if M.shape[0] > mr:
                M = M[torch.randperm(M.shape[0])[:mr]]
            out[name] = M
        return out


class OutputGradCapturer:
    def __init__(self, model, layer_names, max_rows=5000):
        self.model = model
        self.layer_names = layer_names
        self.max_rows = max_rows
        self.buffers: Dict[str, List[torch.Tensor]] = {n: [] for n in layer_names}
        self._hooks = []
        self._counts = {n: 0 for n in layer_names}

    def _make_hook(self, name):
        mod = getattr(self.model, name)
        def hook(module, grad_input, grad_output):
            if self._counts[name] >= self.max_rows:
                return
            g = grad_output[0].detach()
            if isinstance(module, nn.Linear):
                batch = g.reshape(-1, g.shape[-1]).cpu()
            elif isinstance(module, nn.Conv2d):
                B, Cout, H, W = g.shape
                batch = g.permute(0, 2, 3, 1).reshape(-1, Cout).cpu()
            else:
                return
            self.buffers[name].append(batch)
            self._counts[name] += batch.shape[0]
        return hook

    def __enter__(self):
        for name in self.layer_names:
            mod = getattr(self.model, name)
            self._hooks.append(mod.register_full_backward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *a):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def stack(self, max_rows=None):
        mr = max_rows or self.max_rows
        out = {}
        for name, bufs in self.buffers.items():
            if not bufs:
                continue
            M = torch.cat(bufs, 0)
            if M.shape[0] > mr:
                M = M[torch.randperm(M.shape[0])[:mr]]
            out[name] = M
        return out


# =====================================================================
# Basis construction
# =====================================================================
def compute_gpm_basis(inputs_mat, energy_threshold=0.97):
    X = inputs_mat.float()
    if X.shape[0] == 0:
        return None
    cov = X.t() @ X / max(X.shape[0], 1)
    try:
        eigvals, eigvecs = torch.linalg.eigh(cov)
    except Exception:
        try:
            _, S, Vh = torch.linalg.svd(X, full_matrices=False)
            eigvals = S.pow(2) / max(X.shape[0], 1)
            eigvecs = Vh.t()
        except Exception:
            return None
    eigvals = torch.flip(eigvals, [0]).clamp(min=0)
    eigvecs = torch.flip(eigvecs, [1])
    total = eigvals.sum()
    if total <= 0:
        return eigvecs[:, :1]
    cum = torch.cumsum(eigvals, 0)
    r = int((cum / total < energy_threshold).sum().item()) + 1
    return eigvecs[:, :min(r, eigvecs.shape[1])].contiguous()


def compute_fgpm_basis(out_grads_mat, layer_module, energy_threshold=0.95):
    G = out_grads_mat.float()
    if not torch.isfinite(G).all():
        G = G[torch.isfinite(G).all(dim=1)]
        if G.shape[0] == 0:
            return None

    row_norms = G.norm(dim=1, keepdim=True).clamp(min=1e-8)
    G = G / row_norms

    if isinstance(layer_module, nn.Linear):
        W = layer_module.weight.detach().cpu().float()
    elif isinstance(layer_module, nn.Conv2d):
        W = layer_module.weight.detach().cpu().float().reshape(
            layer_module.weight.shape[0], -1)
    else:
        return None

    N = max(G.shape[0], 1)
    H = (G @ W) / (N ** 0.5)

    if not torch.isfinite(H).all():
        H = H[torch.isfinite(H).all(dim=1)]
        if H.shape[0] == 0:
            return None

    try:
        _, S, Vh = torch.linalg.svd(H, full_matrices=False)
    except Exception:
        return None

    V = Vh.t()
    eigvals = S.pow(2)
    if eigvals.numel() == 0 or eigvals[0].item() <= 0:
        return V[:, :1].contiguous()

    nonzero = eigvals > 1e-10 * eigvals[0]
    eigvals = eigvals[nonzero]
    V = V[:, nonzero]

    total = eigvals.sum()
    cum = torch.cumsum(eigvals, 0)
    r = int((cum / total < energy_threshold).sum().item()) + 1
    return V[:, :min(r, V.shape[1])].contiguous()


# =====================================================================
# Basis building — supports mixed GPM/FGPM per layer
# =====================================================================
def build_basis_for_task(
    method, model, loader, task_id, device,
    energy_gpm=0.99, energy_fgpm=0.95, max_samples=1500,
    method_by_layer=None,
):
    """Build per-layer basis. method_by_layer: {layer_name: 'gpm'|'fgpm'}"""
    model.eval()
    layers = list(model.protected_layer_names)

    # Determine per-layer method
    if method_by_layer is None:
        per_layer = {n: method for n in layers}
    else:
        per_layer = method_by_layer

    needs_gpm = any(m == "gpm" for m in per_layer.values())
    needs_fgpm = any(m == "fgpm" for m in per_layer.values())

    gpm_inputs = {}
    fgpm_grads = {}

    # Collect GPM inputs
    if needs_gpm:
        gpm_layers = [n for n in layers if per_layer.get(n) == "gpm"]
        with InputCapturer(model, gpm_layers, max_rows=max_samples) as ic:
            seen = 0
            for x, y in loader:
                x = x.to(device)
                with torch.no_grad():
                    _ = model(x, task_id=task_id)
                seen += x.shape[0]
                if seen >= max_samples:
                    break
        gpm_inputs = ic.stack(max_rows=max_samples)

    # Collect FGPM gradients
    if needs_fgpm:
        fgpm_layers = [n for n in layers if per_layer.get(n) == "fgpm"]
        for _, p in model.named_parameters():
            p.requires_grad_(True)
        criterion = nn.CrossEntropyLoss()
        with OutputGradCapturer(model, fgpm_layers, max_rows=max_samples) as gc:
            seen = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                model.zero_grad()
                logits = model(x, task_id=task_id)
                loss = criterion(logits, y)
                loss.backward()
                seen += x.shape[0]
                if seen >= max_samples:
                    break
        fgpm_grads = gc.stack(max_rows=max_samples)

    # Build bases
    bases = {}
    for name in layers:
        m = per_layer.get(name)
        if m == "gpm" and name in gpm_inputs:
            basis = compute_gpm_basis(gpm_inputs[name], energy_gpm)
            if basis is not None:
                bases[name] = basis
        elif m == "fgpm" and name in fgpm_grads:
            G = fgpm_grads[name][:max_samples]
            mod = getattr(model, name)
            basis = compute_fgpm_basis(G, mod, energy_fgpm)
            if basis is not None:
                bases[name] = basis

    return bases


# =====================================================================
# Training / evaluation
# =====================================================================
@dataclass
class Config:
    method: str = "naive"
    arch: str = "alexnet"
    use_bn: bool = False
    epochs_per_task: int = 10
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    energy_gpm: float = 0.97
    energy_fgpm: float = 0.97
    threshold_increment: float = 0.003
    max_basis_samples: int = 1500
    seed: int = 0
    device: str = "cpu"
    label: str = ""


def evaluate(model, loader, task_id, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x, task_id=task_id).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def train_one_task(model, loader, task_id, cfg, projector=None):
    model.train()
    params = []
    for name, p in model.named_parameters():
        if name.startswith("heads."):
            if name.startswith(f"heads.{task_id}."):
                params.append(p); p.requires_grad_(True)
            else:
                p.requires_grad_(False)
        else:
            params.append(p); p.requires_grad_(True)

    opt = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs_per_task):
        model.train()
        for x, y in loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            opt.zero_grad()
            loss = criterion(model(x, task_id=task_id), y)
            loss.backward()
            if projector is not None:
                projector.project_gradients()
            opt.step()


# =====================================================================
# Experiment drivers
# =====================================================================
@dataclass
class Results:
    method: str
    arch: str = ""
    acc_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    basis_ranks_per_task: List[Dict[str, int]] = field(default_factory=list)

    @property
    def avg_final_acc(self):
        return float(self.acc_matrix[-1].mean()) if self.acc_matrix.size else 0.0

    @property
    def backward_transfer(self):
        T = self.acc_matrix.shape[0]
        if T < 2:
            return 0.0
        diag = np.array([self.acc_matrix[i, i] for i in range(T - 1)])
        return float((self.acc_matrix[-1, :T-1] - diag).mean())

    @property
    def forgetting(self):
        T = self.acc_matrix.shape[0]
        if T < 2:
            return 0.0
        forgets = []
        for j in range(T - 1):
            max_past = self.acc_matrix[:T, j].max()
            forgets.append(max_past - self.acc_matrix[-1, j])
        return float(np.mean(forgets))


def _get_method_by_layer(method, model):
    """Return per-layer method dict for mixed methods."""
    if method == "mixed":
        return {n: "gpm" if n in model.early_layers else "fgpm"
                for n in model.protected_layer_names}
    elif method == "mixed-rev":
        return {n: "fgpm" if n in model.early_layers else "gpm"
                for n in model.protected_layer_names}
    else:
        return None  # uniform method


def run_continual(tasks, classes_per_task, cfg, verbose=True):
    set_seed(cfg.seed)
    model = build_model(cfg.arch, num_tasks=len(tasks),
                        classes_per_task=classes_per_task,
                        use_bn=cfg.use_bn).to(cfg.device)

    needs_projector = cfg.method in {"gpm", "fgpm", "mixed", "mixed-rev"}
    projector = SubspaceProjector(model) if needs_projector else None

    method_by_layer = _get_method_by_layer(cfg.method, model)
    # For build_basis_for_task, determine the base method
    base_method = cfg.method if cfg.method in {"gpm", "fgpm"} else None

    T = len(tasks)
    acc_mat = np.zeros((T, T))
    basis_ranks = []

    for t, task in enumerate(tasks):
        t0 = time.time()
        train_one_task(model, task.train_loader, task.task_id, cfg, projector)
        dt = time.time() - t0

        if projector is not None:
            energy_gpm_t = min(cfg.energy_gpm + t * cfg.threshold_increment, 1.0)
            energy_fgpm_t = min(cfg.energy_fgpm + t * cfg.threshold_increment, 1.0)
            new_bases = build_basis_for_task(
                base_method, model, task.train_loader, task.task_id, cfg.device,
                energy_gpm_t, energy_fgpm_t, cfg.max_basis_samples,
                method_by_layer=method_by_layer,
            )
            for name, U in new_bases.items():
                projector.expand_basis(name, U)
            basis_ranks.append(projector.basis_ranks())

        for j in range(T):
            acc_mat[t, j] = evaluate(model, tasks[j].test_loader,
                                     tasks[j].task_id, cfg.device)

        if verbose:
            past = acc_mat[t, :t+1]
            ranks_str = ""
            if projector and basis_ranks:
                ranks_str = "  ranks=" + ",".join(
                    f"{n}:{basis_ranks[-1][n]}"
                    for n in model.protected_layer_names)
            print(f"  [{cfg.label}] t{t} ({dt:.1f}s)  "
                  f"acc=[{', '.join(f'{a:.1%}' for a in past)}]{ranks_str}")

    return Results(method=cfg.method, arch=cfg.arch, acc_matrix=acc_mat,
                   basis_ranks_per_task=basis_ranks)


def run_joint(tasks, classes_per_task, cfg):
    set_seed(cfg.seed)
    model = build_model(cfg.arch, num_tasks=len(tasks),
                        classes_per_task=classes_per_task,
                        use_bn=cfg.use_bn).to(cfg.device)
    for p in model.parameters():
        p.requires_grad_(True)

    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    total_epochs = cfg.epochs_per_task * len(tasks)
    for epoch in range(total_epochs):
        model.train()
        loaders = [iter(t.train_loader) for t in tasks]
        steps = max(len(t.train_loader) for t in tasks)
        for _ in range(steps):
            for ti, task in enumerate(tasks):
                try:
                    x, y = next(loaders[ti])
                except StopIteration:
                    loaders[ti] = iter(task.train_loader)
                    x, y = next(loaders[ti])
                x, y = x.to(cfg.device), y.to(cfg.device)
                opt.zero_grad()
                loss = criterion(model(x, task_id=task.task_id), y)
                loss.backward()
                opt.step()

    T = len(tasks)
    acc_mat = np.zeros((T, T))
    final = [evaluate(model, t.test_loader, t.task_id, cfg.device) for t in tasks]
    for row in range(T):
        acc_mat[row] = final
    return Results(method="joint", arch=cfg.arch, acc_matrix=acc_mat)


# =====================================================================
# Main
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="cifar10",
                   choices=["cifar10", "cifar100", "both"])
    p.add_argument("--arch", default="alexnet",
                   help="Comma-separated: alexnet,alexnet-l,lenet,vgg11,resnet18")
    p.add_argument("--bn", action="store_true",
                   help="Add BatchNorm to all architectures")
    p.add_argument("--mode", default="sweep", choices=["sweep", "single"])
    p.add_argument("--methods", default="naive,gpm,fgpm,mixed,mixed-rev,joint")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--out_dir", default="./cl_results_v3")
    p.add_argument("--epochs", default="10",
                   help="Comma-separated: 1,10,50")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--basis_samples", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--energy_gpm", type=float, default=0.97)
    p.add_argument("--energy_fgpm", type=float, default=0.97)
    p.add_argument("--thresholds", default=None,
                   help="Comma-separated thresholds: 0.90,0.95,0.97")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    benchmarks = ["cifar10", "cifar100"] if args.benchmark == "both" else [args.benchmark]
    archs = [a.strip() for a in args.arch.split(",")]
    epoch_list = [int(e.strip()) for e in args.epochs.split(",")]
    bn_suffix = "_bn" if args.bn else ""

    for benchmark in benchmarks:
        print("\n" + "=" * 80)
        print(f"  Benchmark: Split-{benchmark.upper()}  BN={args.bn}")
        print("=" * 80)

        tasks, cpt = build_split_cifar(benchmark, args.data_root, args.batch_size,
                                       args.num_workers, args.seed)
        T = len(tasks)
        print(f"  {T} tasks × {cpt} classes/task  device={device}")

        all_results = {}

        for arch in archs:
            for n_epochs in epoch_list:
                print(f"\n{'━' * 80}")
                print(f"  Architecture: {arch}{bn_suffix}  |  Epochs/task: {n_epochs}")
                print(f"{'━' * 80}")

                def make_cfg(**kw):
                    return Config(
                        arch=arch, use_bn=args.bn, epochs_per_task=n_epochs,
                        lr=args.lr, seed=args.seed, device=device,
                        max_basis_samples=args.basis_samples, **kw)

                if args.mode == "sweep":
                    thresholds = [0.90, 0.95, 0.97]
                    if args.thresholds:
                        thresholds = [float(t) for t in args.thresholds.split(",")]

                    configs = [("naive", make_cfg(method="naive"))]

                    for thresh in thresholds:
                        configs.append(
                            (f"gpm e={thresh}", make_cfg(
                                method="gpm", energy_gpm=thresh,
                                threshold_increment=0.003)))
                        configs.append(
                            (f"fgpm e={thresh}", make_cfg(
                                method="fgpm", energy_fgpm=thresh,
                                threshold_increment=0.003)))

                    # Mixed methods at default threshold
                    configs.append(
                        ("mixed", make_cfg(method="mixed",
                                          energy_gpm=0.97, energy_fgpm=0.97,
                                          threshold_increment=0.003)))
                    configs.append(
                        ("mixed-rev", make_cfg(method="mixed-rev",
                                              energy_gpm=0.97, energy_fgpm=0.97,
                                              threshold_increment=0.003)))
                    configs.append(("joint", make_cfg(method="joint")))

                else:  # single mode
                    methods = [m.strip() for m in args.methods.split(",")]
                    configs = []
                    for m in methods:
                        configs.append((m, make_cfg(
                            method=m, energy_gpm=args.energy_gpm,
                            energy_fgpm=args.energy_fgpm,
                            threshold_increment=0.003)))

                import gc as gc_mod
                summaries = []
                for label, cfg in configs:
                    cfg.label = label
                    print(f"\n  ── {label} ──")

                    try:
                        t0 = time.time()
                        if cfg.method == "joint":
                            r = run_joint(tasks, cpt, cfg)
                        else:
                            r = run_continual(tasks, cpt, cfg)
                        dt = time.time() - t0
                        gc_mod.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        summary = {
                            "label": label,
                            "arch": arch + bn_suffix,
                            "epochs": n_epochs,
                            "method": cfg.method,
                            "use_bn": args.bn,
                            "avg_final_acc": round(r.avg_final_acc * 100, 2),
                            "backward_transfer": round(r.backward_transfer * 100, 2),
                            "forgetting": round(r.forgetting * 100, 2),
                            "per_task_final": [round(a * 100, 1) for a in r.acc_matrix[-1]],
                            "acc_matrix": [[round(a * 100, 1) for a in row]
                                           for row in r.acc_matrix],
                            "n_tasks": T,
                            "runtime_s": round(dt, 1),
                        }
                        if r.basis_ranks_per_task:
                            summary["final_basis_ranks"] = r.basis_ranks_per_task[-1]
                        summaries.append(summary)

                        print(f"  → avg={r.avg_final_acc:.1%}  BWT={r.backward_transfer:+.3f}  "
                              f"forget={r.forgetting:.3f}  ({dt:.0f}s)")

                    except Exception as e:
                        print(f"  ✗ FAILED: {e}")
                        import traceback
                        traceback.print_exc()
                        summaries.append({
                            "label": label, "arch": arch + bn_suffix,
                            "epochs": n_epochs, "method": cfg.method,
                            "use_bn": args.bn, "error": str(e),
                        })

                # Summary table
                print(f"\n{'=' * 80}")
                print(f"  {benchmark.upper()} | {arch}{bn_suffix} | {n_epochs} ep/task")
                print(f"{'=' * 80}")
                print(f"  {'Config':>22} | {'Acc':>7} | {'BWT':>7} | {'Forget':>7}")
                print(f"  {'-'*22}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
                for s in summaries:
                    if "error" in s:
                        print(f"  {s['label']:>22} | FAILED")
                    else:
                        print(f"  {s['label']:>22} | {s['avg_final_acc']:6.1f}% | "
                              f"{s['backward_transfer']:+6.2f}% | {s['forgetting']:5.2f}%")

                key = f"{benchmark}_{arch}{bn_suffix}_{n_epochs}ep"
                all_results[key] = summaries

                # Save incrementally after each arch/epoch combo
                fname = os.path.join(args.out_dir, f"results_{benchmark}{bn_suffix}.json")
                with open(fname, "w") as f:
                    json.dump(all_results, f, indent=2)

        fname = os.path.join(args.out_dir, f"results_{benchmark}{bn_suffix}.json")
        print(f"\n  Saved to {fname}")


if __name__ == "__main__":
    main()