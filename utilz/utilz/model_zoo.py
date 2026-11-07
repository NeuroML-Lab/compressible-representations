from __future__ import annotations
from dataclasses import dataclass
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from typing import Callable, Dict, List, Optional, Sequence, Tuple

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

TIMM_REGISTRY = {
    "deit3_b_16":      "deit3_base_patch16_224.fb_in1k",
    "vit_b_16_in21k": "vit_base_patch16_224.augreg2_in21k_ft_in1k"
}


CUSTOM_REGISTRY = {
    "vit_b_16_c_seed1": "./checkpoints/checkpoint_vit_seed1.pth",
    "vit_b_16_c_seed2": "./checkpoints/checkpoint_vit_seed2.pth",
}

TORCH_REGISTRY = {
    "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
    "vit_b_32": (vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1),
    "vit_l_16": (vit_l_16, ViT_L_16_Weights.IMAGENET1K_V1),
    "vit_l_32": (vit_l_32, ViT_L_32_Weights.IMAGENET1K_V1),
}

MODEL_REGISTRY = {**TIMM_REGISTRY, **TORCH_REGISTRY, **CUSTOM_REGISTRY}

@dataclass
class LoadedModel:
    model: nn.Module
    preprocess: Callable
    categories: Sequence[str]
    num_layers: int
    hidden_dim: int


def load_timm_vit(model_name: str, device: torch.device) -> LoadedModel:
    if model_name not in TIMM_REGISTRY:
        raise ValueError(f"Unknown model_name={model_name}. Choices: {list(TIMM_REGISTRY)}")

    timm_name = TIMM_REGISTRY[model_name]
    model = timm.create_model(timm_name, pretrained=True)
    model.eval().to(device)

    cfg = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**cfg, is_training=False)

    # make vit  look like encoder blocks 
    model.encoder = type("_Encoder", (), {"layers": model.blocks})()

    return LoadedModel(
        model=model,
        preprocess=preprocess,
        categories=[],
        num_layers=len(model.blocks),
        hidden_dim=model.embed_dim,
    )
    

def load_vit(model_name: str, device: torch.device) -> LoadedModel:
    if model_name not in TORCH_REGISTRY:
        raise ValueError(f"Unknown model_name={model_name}. Choices: {list(TORCH_REGISTRY)}")
    constructor, weights = TORCH_REGISTRY[model_name]
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
        # categories=[],
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )

def load_custom_vit(model_name: str, device: torch.device) -> LoadedModel:
    if model_name not in CUSTOM_REGISTRY:
        raise ValueError(f"Unknown model_name={model_name}. Choices: {list(CUSTOM_REGISTRY)}")
    checkpoint_path = CUSTOM_REGISTRY[model_name]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    model = vit_b_16(weights=None)
    model.load_state_dict(model_state_dict)
    model.eval().to(device)

    preprocess = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
    categories = ViT_B_16_Weights.IMAGENET1K_V1.meta.get("categories", [])
    num_layers = len(model.encoder.layers)
    hidden_dim = model.hidden_dim

    return LoadedModel(
        model=model,
        preprocess=preprocess,
        categories=categories,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )

def load_model(model_name: str, device: torch.device) -> LoadedModel:
    if model_name in TIMM_REGISTRY:
        return load_timm_vit(model_name, device)
    elif model_name in TORCH_REGISTRY:
        return load_vit(model_name, device)
    elif model_name in CUSTOM_REGISTRY:
        return load_custom_vit(model_name, device)
    else:
        raise ValueError(f"Unknown model_name={model_name}. Choices: {list(MODEL_REGISTRY)}")
    


