"""Shared utilities for the two task-conditioned CKA experiments."""

from __future__ import annotations

import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import timm


MODEL_NAMES = {
    "vit_b_16": "vit_base_patch16_224",
    "vit_b_16_in21k": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "deit3_b_16": "deit3_base_patch16_224",
    "vit_l_16": "vit_large_patch16_224",
}

DISPLAY_NAMES = {
    "vit_b_16": "ViT-B/16",
    "vit_b_16_in21k": "ViT-B/16 (IN-21k)",
    "deit3_b_16": "DeiT3-B/16",
    "vit_l_16": "ViT-L/16",
}

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size: int, workers: int,
                shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


class ViT:
    """Small adapter around the four pretrained ViTs used in Figure 5."""

    def __init__(self, key: str, device: str):
        if key not in MODEL_NAMES:
            raise KeyError(f"Unknown model {key!r}")
        self.key = key
        self.display_name = DISPLAY_NAMES[key]
        self.device = torch.device(device)
        self.model = timm.create_model(MODEL_NAMES[key], pretrained=True)
        self.model.eval().to(self.device)
        self.model.requires_grad_(False)
        self.blocks = self.model.blocks
        self.width = int(self.model.num_features)

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def penultimate_block(self) -> int:
        return self.num_blocks - 2

    @contextmanager
    def capture_blocks(self, layer_indices: list[int]):
        """Capture block outputs without changing the model computation."""
        outputs: dict[int, torch.Tensor] = {}
        handles = []

        for index in layer_indices:
            def hook(_module, _inputs, output, index=index):
                outputs[index] = output[0] if isinstance(output, tuple) else output
            handles.append(self.blocks[index].register_forward_hook(hook))
        try:
            yield outputs
        finally:
            for handle in handles:
                handle.remove()

    def layer_indices_for_depth_experiment(self) -> list[int]:
        # Base models are compared at every block. ViT-L is sampled at eight
        # evenly spaced depths, as specified in Appendix G.1.
        if self.num_blocks == 24:
            return sorted(set(np.linspace(0, 23, 8, dtype=int).tolist()))
        return list(range(self.num_blocks))


def centered_linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA computed without constructing sample-by-sample Gram matrices."""
    x = x.double() - x.double().mean(dim=0, keepdim=True)
    y = y.double() - y.double().mean(dim=0, keepdim=True)
    cross = x.T @ y
    numerator = cross.square().sum()
    denominator = torch.sqrt(
        (x.T @ x).square().sum() * (y.T @ y).square().sum()
    )
    return float((numerator / denominator.clamp_min(1e-30)).item())


def eigenspaces(matrix: torch.Tensor, max_rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return eigenvalues plus the leading and trailing ``max_rank`` vectors."""
    matrix = matrix.double()
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = torch.linalg.eigh(matrix)
    order = torch.argsort(values, descending=True)
    values = values[order].clamp_min(0)
    vectors = vectors[:, order]
    rank = min(max_rank, vectors.shape[1] // 2)
    return values, vectors[:, :rank].float(), vectors[:, -rank:].float()


def energy_eigenspace(matrix: torch.Tensor, fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Smallest leading eigenspace containing ``fraction`` of spectral mass."""
    matrix = 0.5 * (matrix.double() + matrix.double().T)
    values, vectors = torch.linalg.eigh(matrix)
    order = torch.argsort(values, descending=True)
    values = values[order].clamp_min(0)
    vectors = vectors[:, order]
    total = values.sum()
    if total <= 0:
        rank = 1
    else:
        rank = int((torch.cumsum(values, 0) / total < fraction).sum()) + 1
    return values, vectors[:, :rank].float()


def project(features: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project row-wise features into the original space through ``basis``."""
    return (features @ basis) @ basis.T


class MLPProbe(nn.Module):
    def __init__(self, width: int, classes: int = 16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

