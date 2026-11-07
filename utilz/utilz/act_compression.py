import torch
from torch import nn
from typing import Dict, List, Sequence, Tuple, Optional
from torch.utils.data import DataLoader



def nearest_int_budget(total_dims: int, retain_fraction: float) -> int:
    return int(round(total_dims * retain_fraction))


def build_candidate_ranks(hidden_dim: int, fraction_grid: Sequence[float]) -> List[int]:
    ranks = {max(1, min(hidden_dim - 1, int(round(hidden_dim * frac)))) for frac in fraction_grid}
    return sorted(ranks)

def safe_quantile(x: torch.Tensor, q: float) -> torch.Tensor:
    x_flat = x.reshape(-1)
    n = x_flat.numel()
    if n < 16_000_000:
        return torch.quantile(x_flat, q)
    sorted_x, _ = torch.sort(x_flat)
    idx_f = q * (n - 1)
    idx_lo = int(idx_f)
    idx_hi = min(idx_lo + 1, n - 1)
    frac = idx_f - idx_lo
    return sorted_x[idx_lo] + frac * (sorted_x[idx_hi] - sorted_x[idx_lo])

# instantiated class acts like a hook that can be registered on a layer to project that layer's CLS reps
class SingleCLSProjectionHook:
    def __init__(self, projector: torch.Tensor, renorm: bool):
        self.projector = projector
        self.renorm = renorm

    def __call__(self, _module: nn.Module, _inputs: Tuple[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
        cls = output[:, 0, :]                              
        proj = cls @ self.projector.T                     

        if self.renorm:
            src_norm  = cls.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            proj_norm = proj.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            proj = proj * (src_norm / proj_norm)           

        return torch.cat([proj.unsqueeze(1), output[:, 1:, :]], dim=1)
    
# instantiated class acts like a hook factory can be called on a layer index to generate a hook to be registered
class MultiCLSProjectionHook:
    def __init__(self, projectors: Dict[int, torch.Tensor], renorm: bool):
        self.projectors = projectors
        self.renorm = renorm

    def make_hook(self, layer_idx: int):
        projector = self.projectors[layer_idx]

        def hook(_module: nn.Module, _inputs: Tuple[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
            cls = output[:, 0, :]                    
            proj = cls @ projector.T                

            if self.renorm:
                src_norm  = cls.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                proj_norm = proj.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                proj = proj * (src_norm / proj_norm)

            return torch.cat([proj.unsqueeze(1), output[:, 1:, :]], dim=1)

        return hook

# builds projectors for each layer using pre-computed eigenspectra
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


# helper functions for estimating projector matrices
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





# for Task-sensitivity projectors
def estimate_task_sensitive_matrix(
    model: nn.Module,
    loader: DataLoader,
    layer_idx: int,
    hidden_dim: int,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    target_layer = model.encoder.layers[layer_idx]
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

#for PCA projectors
def estimate_representation_covariance(
    model: nn.Module,
    loader: DataLoader,
    layer_idx: int,
    hidden_dim: int,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    target_layer = model.encoder.layers[layer_idx]
    d = hidden_dim

    n = 0
    sum_z = torch.zeros(d, dtype=torch.float64, device="cpu")
    sum_zz = torch.zeros(d, d, dtype=torch.float64, device="cpu")

    for batch_idx, (images, _labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        cache = {}

        def hook_fn(_module: nn.Module, _inputs: Tuple[torch.Tensor], output: torch.Tensor):
            cache["layer_output"] = output.detach()

        h = target_layer.register_forward_hook(hook_fn)
        try:
            _ = model(images)
        finally:
            h.remove()

        z = cache["layer_output"][:, 0, :]
        z_cpu = z.double().cpu()
        sum_z += z_cpu.sum(dim=0)
        sum_zz += z_cpu.T @ z_cpu
        n += z_cpu.shape[0]

    if n == 0:
        raise RuntimeError("No samples processed while estimating covariance")

    mean = sum_z / float(n)
    cov = (sum_zz / float(n)) - torch.outer(mean, mean)
    cov = 0.5 * (cov + cov.T)

    eigvals = torch.linalg.eigvalsh(cov).float()
    stats = {
        "n_samples": float(n),
        "trace": float(torch.trace(cov).item()),
        "max_eig": float(eigvals.max().item()),
        "min_eig": float(eigvals.min().item()),
        "effective_rank_participation_ratio": float((eigvals.sum() ** 2 / eigvals.pow(2).sum().clamp_min(1e-12)).item()),
    }
    return mean.float(), cov.float(), stats

# for Local projectors
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



def estimate_geotopology_matrix(
    model: nn.Module,
    loader: DataLoader,
    layer_idx: int,
    device: torch.device,
    distance_metric: str,
    gt_lower_quantile: float,
    gt_upper_quantile: float,
    gt_weight_mode: str,
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    
    
    def _offdiag_values(D: torch.Tensor) -> torch.Tensor:
        n = D.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=D.device)
        vals = D[mask]
        vals = vals[torch.isfinite(vals)]
        if vals.numel() == 0:
            raise RuntimeError("No finite off-diagonal distances found.")
        return vals


    def build_geotopology_weight_matrix(
        D: torch.Tensor,
        lower_quantile: float,
        upper_quantile: float,
        weight_mode: str,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        
        if not (0.0 <= lower_quantile < upper_quantile <= 1.0):
            raise ValueError("Need 0 <= lower_quantile < upper_quantile <= 1.")

        vals = _offdiag_values(D)
        l = float(safe_quantile(vals, lower_quantile))
        u = float(safe_quantile(vals, upper_quantile))
        if u <= l:
            u = l + 1e-8

        t = (D - l) / max(u - l, 1e-8)
        W = torch.zeros_like(D)

        if weight_mode == "middle_binary":
            W[(D > l) & (D < u)] = 1.0
        elif weight_mode == "middle_triangular":
            # band-pass emphasis with zero weight at the two thresholds and peak in the middle
            mid = (t > 0.0) & (t < 1.0)
            W[mid] = 4.0 * t[mid] * (1.0 - t[mid])
        elif weight_mode == "paper_gt":
            # direct piecewise-linear GT transform from the paper: 0 below l, linear in (l,u), 1 above u
            mid = (D > l) & (D < u)
            hi = D >= u
            W[mid] = t[mid]
            W[hi] = 1.0
        else:
            raise ValueError(weight_mode)

        W = 0.5 * (W + W.T)
        W.fill_diagonal_(0.0)

        nz = W[W > 0]
        stats = {
            "lower_quantile": float(lower_quantile),
            "upper_quantile": float(upper_quantile),
            "lower_threshold": float(l),
            "upper_threshold": float(u),
            "num_nonzero_edges_directed": float((W > 0).sum().item()),
            "avg_weighted_degree": float(W.sum(dim=1).mean().item()),
            "mean_nonzero_weight": float(nz.mean().item()) if nz.numel() > 0 else 0.0,
            "median_offdiag_distance": float(torch.median(vals).item()),
        }
        return W, stats

    Z = collect_cls_representations(model, loader, layer_idx, device, max_batches)
    D = pairwise_distance_matrix(Z, distance_metric)
    W, w_stats = build_geotopology_weight_matrix(
        D=D,
        lower_quantile=gt_lower_quantile,
        upper_quantile=gt_upper_quantile,
        weight_mode=gt_weight_mode,
    )
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
        "distance_metric": distance_metric,
        "gt_weight_mode": gt_weight_mode,
        **w_stats,
    }
    return M.float(), stats



def estimate_projector(
    compression_type: str,
    model: nn.Module,
    loader: DataLoader,
    layer_idx: int,
    device: torch.device,
    candidate_ranks: List[int],
    hidden_dim: int = None,
    max_batches: Optional[int] = None,
    rand_seed: Optional[int] = 0,
    args = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    
    projectors = {}
    null_projectors = {}

    if compression_type == "task_sensitive":
        P, P_stats = estimate_task_sensitive_matrix(model, loader, layer_idx, hidden_dim, device, max_batches)
        eigvals, eigvecs = torch.linalg.eigh(P)
    elif compression_type == "pca":
        _, P, P_stats = estimate_representation_covariance(
                model=model,
                loader=loader,
                layer_idx=layer_idx,
                hidden_dim=hidden_dim,
                device=device,
                max_batches=max_batches,
            )
        eigvals, eigvecs = torch.linalg.eigh(P)
    
    elif compression_type == "local_geometry":
        P, P_stats = estimate_local_geometry_matrix(
            model=model,
            loader=loader,
            layer_idx=layer_idx,
            device=device,
            k_neighbors=args.k_neighbors,
            neighbor_metric=args.neighbor_metric,
            weight_mode=args.weight_mode,
            sigma=args.sigma,
            max_batches=max_batches,
        )
        eigvals, eigvecs = torch.linalg.eigh(P)

    elif compression_type == "geotopology":
        P, P_stats = estimate_geotopology_matrix(
            model=model,
            loader=loader,
            layer_idx=layer_idx,
            device=device,
            distance_metric=args.distance_metric,
            gt_lower_quantile=args.gt_lower_quantile,
            gt_upper_quantile=args.gt_upper_quantile,
            gt_weight_mode=args.gt_weight_mode,
            max_batches=max_batches,
        )
        eigvals, eigvecs = torch.linalg.eigh(P)

    elif compression_type == "random":
        torch.manual_seed(rand_seed)
        P = torch.randn(hidden_dim, hidden_dim, dtype=torch.float64)
        P = 0.5 * (P + P.T)
        P_stats = {}
        _, eigvecs = torch.linalg.eigh(P)
        eigvals = torch.rand(hidden_dim, dtype=torch.float64)

        
    else:
        raise ValueError(f"Unknown compression type {compression_type}. Cannot estimate proj matrix.")
    
    eigvals = eigvals.float()
    eigvecs = eigvecs.float()
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].cpu()
    eigvecs = eigvecs[:, order].cpu()


    for rank in candidate_ranks:
        rank = max(1, min(hidden_dim - 1, int(rank)))
        U_compressed = eigvecs[:, :rank]
        U_null = eigvecs[:, rank:]
        projectors[rank] = (U_compressed @ U_compressed.T).cpu()
        null_projectors[rank] = (U_null @ U_null.T).cpu()

    projector_res = {
        "projectors": projectors,
        "null_projectors": null_projectors,
        "P_stats": P_stats,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
    }
    


    


         
    
    return projector_res