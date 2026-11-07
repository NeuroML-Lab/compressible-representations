import torch
import numpy as np
from typing import Dict, Optional, Tuple
from torch import nn
from torch.utils.data import DataLoader
from utilz.act_compression import SingleCLSProjectionHook, MultiCLSProjectionHook
from utilz.alignment_metrics import compute_RSA, compute_CKA
from utilz.utilz.brain_data import get_tripleN_fr
from scipy.linalg import subspace_angles


def collect_reps(
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
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            model(images)
    h.remove()

    return torch.cat(cache, dim=0).numpy()  # [N, D]


def evaluate_single_layer_model_alignment(
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
    target_model_reps = None,
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
    model_reps = collect_reps(model, loader, device, layer_idx, cls_token=cls_token)  # [N, D]

    if hook is not None:
        hook.remove()
    # print(f"Shape of model representations: {model_reps.shape}, target reps: {target_model_reps.shape}")
    rsa = compute_RSA(model_reps, target_model_reps, dist_metric=rsa_dist_metric, corr_metric=rsa_corr_metric)
    cka = compute_CKA(model_reps, target_model_reps, kernel=cka_kernel)


    assert model_reps.shape[0] == target_model_reps.shape[0], (
        f"Stimulus count mismatch: model={model_reps.shape[0]}, brain={target_model_reps.shape[0]}"
    )

    return {
        "rsa": float(rsa),
        "cka": float(cka),
    }


def evaluate_single_layer_cross_model_alignment(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    layer_idx1: int,
    layer_idx2: int,
    projector1: Optional[torch.Tensor],
    projector2: Optional[torch.Tensor],
    renorm: bool,
    rsa_dist_metric: str = "correlation",
    rsa_corr_metric: str = "spearman",
    cka_kernel: str = "linear",
    cls_token: bool = True,
) -> Dict[str, float]:
    """
    Evaluates brain alignment of (optionally projected) CLS reps at a single layer.
    Mirrors evaluate_single_layer_projection interface.

    Returns RSA and linear CKA scores between model reps and brain_data.
    """
    model.eval()
    hook = None
    if projector1 is not None:
        hook = model.encoder.layers[layer_idx1].register_forward_hook(
            SingleCLSProjectionHook(projector1.to(device), renorm=renorm)
        )

    # Collect projected (or raw) CLS reps
    model_reps1 = collect_reps(model, loader, device, layer_idx1, cls_token=cls_token)  # [N, D]

    if hook is not None:
        hook.remove()


    # Apply second projection to the collected reps
    hook = None
    if projector2 is not None:
        hook = model.encoder.layers[layer_idx2].register_forward_hook(
            SingleCLSProjectionHook(projector2.to(device), renorm=renorm)
        )

    # Collect projected (or raw) CLS reps
    model_reps2 = collect_reps(model, loader, device, layer_idx2, cls_token=cls_token)  # [N, D]

    if hook is not None:
        hook.remove()

    # print(f"Shape of model representations: {model_reps.shape}, target reps: {target_model_reps.shape}")
    rsa = compute_RSA(model_reps1, model_reps2, dist_metric=rsa_dist_metric, corr_metric=rsa_corr_metric)
    cka = compute_CKA(model_reps1, model_reps2, kernel=cka_kernel)


    assert model_reps1.shape[0] == model_reps2.shape[0], (
        f"Stimulus count mismatch: model={model_reps1.shape[0]}, brain={model_reps2.shape[0]}"
    )

    return {
        "rsa": float(rsa),
        "cka": float(cka),
    }


# function that computes subspace agreement metrics between two sets of eigenvectors (eigenspaces)
def compute_subspace_agreement(V_f: np.ndarray, V_g: np.ndarray) -> dict:
    r_f, r_g = V_f.shape[1], V_g.shape[1]

    if r_f != r_g:
        max_r = max(r_f, r_g)
        if r_f < max_r:
            V_f = np.pad(V_f, ((0, 0), (0, max_r - r_f)))
        if r_g < max_r:
            V_g = np.pad(V_g, ((0, 0), (0, max_r - r_g)))

    angles = subspace_angles(V_f, V_g)
    
    return {
        "mean_principal_angle_deg": float(np.degrees(angles).mean()),
        "grassmann_distance":       float(np.sqrt((np.sin(angles) ** 2).sum())),
        "mean_cosine_similarity":   float(np.cos(angles).mean()),
    }



def evaluate_multi_layer_model_alignment(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    layer_idx: int,                          # which layer to extract reps from
    projectors: Dict[int, torch.Tensor],     # all layers to project (multi-layer hooks)
    renorm: bool,
    rsa_dist_metric: str = "correlation",
    rsa_corr_metric: str = "spearman",
    cka_kernel: str = "linear",
    target_model_reps = None,
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
        model_reps = collect_reps(model, loader, device, layer_idx, cls_token=True)  # [N, D]
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

    assert model_reps.shape[0] == target_model_reps.shape[0], (
        f"Stimulus count mismatch: model={model_reps.shape[0]}, brain={target_model_reps    .shape[0]}"
    )

    return {
        "rsa_mean": float(np.mean(rsa_folds)),
        "rsa_std": float(np.std(rsa_folds)),
        "cka_mean": float(np.mean(cka_folds)),
        "cka_std": float(np.std(cka_folds)),
    }