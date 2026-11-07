import torchattacks
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from utilz.model_evals import accuracy_topk
from utilz.act_compression import SingleCLSProjectionHook, MultiCLSProjectionHook
# from autoattack import AutoAttack

@dataclass
class AdvEvalResult:
    # Accuracy
    clean_top1: float = 0.0
    clean_top5: float = 0.0
    robust_top1: float = 0.0
    robust_top5: float = 0.0

    # Derived
    top1_accuracy_drop: float = 0.0
    top5_accuracy_drop: float = 0.0
    robustness_retention_ratio: float = 0.0   # robust_top1 / clean_top1

    # Perturbation stats
    mean_linf: float = 0.0
    mean_l2: float = 0.0
    mean_l1: float = 0.0

    # Attack config
    attack_type: str = ""
    eps: float = 0.0
    layer_idx: int = -1
    subspace: str = ""           # "task", "pca", "local", "{space}_null", "none"
    rank: int = -1
    renorm: bool = False
    n_eval: int = 0

    def finalize(self):
        self.top1_accuracy_drop = self.clean_top1 - self.robust_top1
        self.top5_accuracy_drop = self.clean_top5 - self.robust_top5
        denom = self.clean_top1 if self.clean_top1 > 1e-8 else 1e-8
        self.robustness_retention_ratio = self.robust_top1 / denom
        return self

    def to_dict(self):
        return asdict(self)


def _perturbation_stats(clean: torch.Tensor, adv: torch.Tensor) -> Dict[str, float]:
    """Compute per-image perturbation norms, averaged over batch."""
    delta = (adv - clean).view(clean.shape[0], -1)
    return {
        "mean_linf": delta.abs().max(dim=1).values.mean().item(),
        "mean_l2":   delta.norm(p=2, dim=1).mean().item(),
        "mean_l1":   delta.norm(p=1, dim=1).mean().item(),
    }


def _build_attack(
    model: nn.Module,
    attack_type: str,
    eps: float,
    n_classes: int = 1000,
) -> object:
    """Build a torchattacks attacker."""
    if attack_type == "pgd20":
        return torchattacks.PGD(model, eps=eps, alpha=1/255, steps=20)
    elif attack_type == "pgd100":
        return torchattacks.PGD(model, eps=eps, alpha=0.5/255, steps=100)
    elif attack_type == "fgsm":
        return torchattacks.FGSM(model, eps=eps)
    # elif attack_type == "autoattack":
    #     return torchattacks.AutoAttack(
    #         model, norm="Linf", eps=eps,
    #         version="standard", n_classes=n_classes
    #     )
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")


def evaluate_single_layer_projection_adv(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    layer_idx: int,
    projector: Optional[torch.Tensor],
    renorm: bool,
    attack_type: str = "pgd20",
    eps: float = 4 / 255,
    subspace_label: str = "task",
    rank: int = -1,
    adv_images_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> AdvEvalResult:
    """
    Adversarial evaluation for a single projected layer.

    Args:
        adv_images_cache: If provided, uses pre-generated adversarial images
                          (for transfer attack scenario). Each entry is
                          (adv_images, labels) already on CPU.
                          If None, generates attacks on the projected model (white-box).
    """
    model.eval()

    hook = None
    if projector is not None:
        hook = model.encoder.layers[layer_idx].register_forward_hook(
            SingleCLSProjectionHook(projector.to(device), renorm=renorm)
        )

    result = AdvEvalResult(
        attack_type=attack_type,
        eps=eps,
        layer_idx=layer_idx,
        subspace=subspace_label,
        rank=rank,
        renorm=renorm,
    )

    # Build attacker on hooked model (for compressed_model condition) only if no cache provided
    attacker = None
    if adv_images_cache is None:
        attacker = _build_attack(model, attack_type, eps)

    linf_acc, l2_acc, l1_acc = [], [], []
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # --- Clean accuracy ---
        with torch.no_grad():
            clean_logits = model(images)
            
        accs_clean = accuracy_topk(clean_logits, labels, topk=(1, 5))
        bs = labels.shape[0]
        total += bs
        result.clean_top1 += accs_clean[1]
        result.clean_top5 += accs_clean[5]

        # --- Adversarial images ---
        if adv_images_cache is not None:
            adv_imgs, _ = adv_images_cache[batch_idx]
            adv_imgs = adv_imgs.to(device)
        else:
            adv_imgs = attacker(images, labels)  # gradient flows through hooks

        # --- Perturbation stats ---
        # stats = _perturbation_stats(images.cpu(), adv_imgs.cpu())
        # linf_acc.append(stats["mean_linf"])
        # l2_acc.append(stats["mean_l2"])
        # l1_acc.append(stats["mean_l1"])

        # --- Robust accuracy ---
        with torch.no_grad():
            adv_logits = model(adv_imgs)
        accs_adv = accuracy_topk(adv_logits, labels, topk=(1, 5))
        result.robust_top1 += accs_adv[1]
        result.robust_top5 += accs_adv[5]


    if hook is not None:
        hook.remove()

    n = max(total, 1)
    result.clean_top1 /= n
    result.clean_top5 /= n
    result.robust_top1 /= n
    result.robust_top5 /= n
    result.mean_linf = float(np.mean(linf_acc))
    result.mean_l2   = float(np.mean(l2_acc))
    result.mean_l1   = float(np.mean(l1_acc))
    result.n_eval = total

    return result.finalize()


def evaluate_multi_layer_projection_adv(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    projectors: Dict[int, torch.Tensor],
    renorm: bool,
    attack_type: str = "pgd20",
    eps: float = 4 / 255,
    subspace_label: str = "task",
    rank: int = -1,
    adv_images_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> AdvEvalResult:
    """Multi-layer version of adversarial evaluation."""
    model.eval()

    handles = []
    hook_factory = MultiCLSProjectionHook(
        {k: v.to(device) for k, v in projectors.items()}, renorm=renorm
    )
    for layer_idx in sorted(projectors):
        handles.append(
            model.encoder.layers[layer_idx].register_forward_hook(
                hook_factory.make_hook(layer_idx)
            )
        )

    result = AdvEvalResult(
        attack_type=attack_type,
        eps=eps,
        layer_idx=-1,  # -1 signals multi-layer
        subspace=subspace_label,
        rank=rank,
        renorm=renorm,
    )

    attacker = None
    if adv_images_cache is None:
        attacker = _build_attack(model, attack_type, eps)

    linf_acc, l2_acc, l1_acc = [], [], []
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            clean_logits = model(images)
        accs_clean = accuracy_topk(clean_logits, labels, topk=(1, 5))
        bs = labels.shape[0]
        result.clean_top1 += accs_clean[1]
        result.clean_top5 += accs_clean[5]

        if adv_images_cache is not None:
            adv_imgs, _ = adv_images_cache[batch_idx]
            adv_imgs = adv_imgs.to(device)
        else:
            adv_imgs = attacker(images, labels)

        # stats = _perturbation_stats(images.cpu(), adv_imgs.cpu())
        # linf_acc.append(stats["mean_linf"])
        # l2_acc.append(stats["mean_l2"])
        # l1_acc.append(stats["mean_l1"])

        with torch.no_grad():
            adv_logits = model(adv_imgs)
        accs_adv = accuracy_topk(adv_logits, labels, topk=(1, 5))
        result.robust_top1 += accs_adv[1]
        result.robust_top5 += accs_adv[5]
        total += bs

    for h in handles:
        h.remove()

    n = max(total, 1)
    result.clean_top1 /= n
    result.clean_top5 /= n
    result.robust_top1 /= n
    result.robust_top5 /= n
    result.mean_linf = float(np.mean(linf_acc))
    result.mean_l2   = float(np.mean(l2_acc))
    result.mean_l1   = float(np.mean(l1_acc))
    result.n_eval = total

    return result.finalize()


def generate_attack_cache_on_fullmodel(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    attack_type: str = "pgd20",
    eps: float = 4 / 255,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate adversarial examples on the UNCOMPRESSED model.
    Returns a list of (adv_images, labels) on CPU for reuse across all
    projection configurations (transfer attack scenario).
    """
    if attack_type == "autoattack":
        # cache = generate_autoattack_cache_on_fullmodel(model, loader, device)
        raise NotImplementedError("AutoAttack cache generation not implemented yet.")
    else:
        model.eval()
        attacker = _build_attack(model, attack_type, eps)
        cache = []
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            adv_imgs = attacker(images, labels)
            cache.append((adv_imgs.cpu(), labels.cpu()))
    return cache

# def generate_autoattack_cache_on_fullmodel(
#     model: nn.Module,
#     loader: DataLoader,
#     device: torch.device,
#     eps: float = 4 / 255,
#     internal_bs: int = 32,  
# ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
#     model.eval()
#     adversary = AutoAttack(model, norm='Linf', eps=eps, version='standard', device=device, verbose=False)
#     cache = []

#     # Collect all images/labels first, then run AA in controlled chunks
#     all_images, all_labels = [], []
#     for images, labels in loader:
#         all_images.append(images)
#         all_labels.append(labels)
#     all_images = torch.cat(all_images)
#     all_labels = torch.cat(all_labels)

#     # AA handles chunking internally with internal_bs
#     adv_images = adversary.run_standard_evaluation(
#         all_images.to(device), all_labels.to(device), bs=internal_bs
#     )

#     # Split back into per-batch cache to match your existing interface
#     batch_size = loader.batch_size
#     for i in range(0, len(adv_images), batch_size):
#         cache.append((
#             adv_images[i:i+batch_size].cpu(),
#             all_labels[i:i+batch_size].cpu()
#         ))
#     torch.cuda.empty_cache()
#     return cache