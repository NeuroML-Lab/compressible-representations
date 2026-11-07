import torch
import numpy as np
from typing import Dict, Optional, Tuple
from torch import nn
from torch.utils.data import DataLoader
from utilz.act_compression import SingleCLSProjectionHook, MultiCLSProjectionHook
from utilz.alignment_metrics import compute_RSA, compute_CKA
from utilz.utilz.brain_data import get_tripleN_fr
def _collect_reps(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    layer_idx: int,
    cls_token: bool = True,
) -> np.ndarray:
    """Hook-based  token extractor for a single layer. Returns [N, D] numpy array.
        If cls_token is True, extracts CLS token (output[:, 0, :]), else extracts mean of all patch tokens.
    """

    cache = []

    def hook_fn(_module, _inputs, output):
        if cls_token:
            cache.append(output[:, 0, :].detach().cpu().float())
        else:
            cache.append(output[:, 1:, :].mean(dim=1).detach().cpu().float())

    h = model.encoder.layers[layer_idx].register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        for images in loader:
            images = images.to(device, non_blocking=True)
            model(images)
    h.remove()

    return torch.cat(cache, dim=0).numpy()  # [N, D]


def evaluate_single_layer_brain_alignment(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    layer_idx: int,
    projector: Optional[torch.Tensor],
    renorm: bool,
    rsa_dist_metric: str = "correlation",
    rsa_corr_metric: str = "spearman",
    cka_kernel: str = "linear",
    cls_token: bool = True,
    args = None,
) -> Dict[str, float]:
    """
    Evaluates brain alignment of (optionally projected) CLS reps at a single layer.
    Mirrors evaluate_single_layer_projection interface.

    Returns RSA and linear CKA scores between model reps and brain_data.
    """
    model.eval()
    hook = None
    if projector is not None:
        hook = model.encoder.layers[layer_idx].register_forward_hook(
            SingleCLSProjectionHook(projector.to(device), renorm=renorm)
        )

    # Collect projected (or raw) CLS reps
    model_reps = _collect_reps(model, loader, device, layer_idx, cls_token=cls_token)  # [N, D]

    if hook is not None:
        hook.remove()


    sampling_seeds = range(5)
    rsa_folds = []    
    cka_folds = []
    for seed in sampling_seeds:
        brain_data = get_tripleN_fr(args.triplenFR_path, n_ch_samples=args.n_ch_samples, seed=seed)
        rsa = compute_RSA(model_reps, brain_data, dist_metric=rsa_dist_metric, corr_metric=rsa_corr_metric)
        cka = compute_CKA(model_reps, brain_data, kernel=cka_kernel)
        rsa_folds.append(rsa)
        cka_folds.append(cka)

    assert model_reps.shape[0] == brain_data.shape[0], (
        f"Stimulus count mismatch: model={model_reps.shape[0]}, brain={brain_data.shape[0]}"
    )

    return {
        "rsa_mean": float(np.mean(rsa_folds)),
        "rsa_std": float(np.std(rsa_folds)),
        "cka_mean": float(np.mean(cka_folds)),
        "cka_std": float(np.std(cka_folds)),
    }


def evaluate_multi_layer_brain_alignment(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    layer_idx: int,                          # which layer to extract reps from
    projectors: Dict[int, torch.Tensor],     # all layers to project (multi-layer hooks)
    renorm: bool,
    rsa_dist_metric: str = "correlation",
    rsa_corr_metric: str = "spearman",
    cka_kernel: str = "linear",
    args = None,
) -> Dict[str, float]:
    """
    Evaluates brain alignment when multiple layers are simultaneously projected.
    Reps are extracted from `layer_idx` after all projection hooks have fired.
    Mirrors evaluate_multi_layer_projection interface.
    """
    model.eval()
    handles = []
    hook_factory = MultiCLSProjectionHook(
        {k: v.to(device) for k, v in projectors.items()}, renorm=renorm
    )
    for l in sorted(projectors):
        handles.append(
            model.encoder.layers[l].register_forward_hook(hook_factory.make_hook(l))
        )

    try:
        model_reps = _collect_reps(model, loader, device, layer_idx, cls_token=True)  # [N, D]
    finally:
        for h in handles:
            h.remove()


    sampling_seeds = range(5)
    rsa_folds = []    
    cka_folds = []
    for seed in sampling_seeds:
        brain_data = get_tripleN_fr(args.triplenFR_path, n_ch_samples=args.n_ch_samples, seed=seed)
        rsa = compute_RSA(model_reps, brain_data, dist_metric=rsa_dist_metric, corr_metric=rsa_corr_metric)
        cka = compute_CKA(model_reps, brain_data, kernel=cka_kernel)
        rsa_folds.append(rsa)
        cka_folds.append(cka)

    assert model_reps.shape[0] == brain_data.shape[0], (
        f"Stimulus count mismatch: model={model_reps.shape[0]}, brain={brain_data.shape[0]}"
    )

    return {
        "rsa_mean": float(np.mean(rsa_folds)),
        "rsa_std": float(np.std(rsa_folds)),
        "cka_mean": float(np.mean(cka_folds)),
        "cka_std": float(np.std(cka_folds)),
    }