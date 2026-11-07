"""
Decomposing Representational Alignment into Task-sensitive and Null Subspaces
=============================================================================
V2: Per-layer head functions that route through remaining transformer blocks.

Usage:
  python decomposed_alignment.py \
      --model_type vision \
      --model_f vit_small_patch16_224 \
      --model_g deit_small_patch16_224 \
      --n_stimuli 256 \
      --imagenet_path /path/to/imagenet/val \
      --save_dir ./results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Literal, Union
from pathlib import Path
import warnings
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import subspace_angles


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DecompositionConfig:
    variance_threshold: float = 0.90
    max_task_dim: int = 512
    min_null_dim: int = 10
    n_stimuli: int = 512
    jacobian_batch_size: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    metrics: list = field(default_factory=lambda: ["cka", "procrustes"])
    save_dir: str = "./results"
    seed: int = 42
    num_workers: int = 2


# ============================================================================
# Activation pooling
# ============================================================================

def pool_activations(act: torch.Tensor, mode: str = "auto") -> torch.Tensor:
    if act.ndim == 2:
        return act
    elif act.ndim == 3:
        if mode == "cls":
            return act[:, 0, :]
        return act.mean(dim=1)
    elif act.ndim == 4:
        return act.mean(dim=(2, 3))
    else:
        raise ValueError(f"Unexpected activation shape: {act.shape}")


# ============================================================================
# Per-layer head function factories
# ============================================================================

def make_vit_head_fn(model: nn.Module, layer_idx: int):
    """
    Head function that routes activations from layer `layer_idx` through
    ALL remaining transformer blocks, layer norm, and classification head.
    """
    remaining_blocks = list(model.blocks[layer_idx + 1:])
    norm = getattr(model, 'norm', None) or getattr(model, 'fc_norm', None)
    head = model.head

    def head_fn(z):
        # z: (B, hidden_dim) -- pooled activation at layer_idx
        x = z.unsqueeze(1)  # (B, 1, D)
        for block in remaining_blocks:
            x = block(x)
        x = x.squeeze(1)    # (B, D)
        if norm is not None:
            x = norm(x)
        return head(x)

    return head_fn


def make_llm_head_fn(model: nn.Module, layer_idx: int, output_dim: int = 100):
    """
    Head function for language models: routes through remaining transformer
    layers, final layer norm, and LM head.
    """
    if hasattr(model, 'transformer'):  # GPT2
        blocks = list(model.transformer.h[layer_idx + 1:])
        ln_f = model.transformer.ln_f
        lm_head = model.lm_head
    elif hasattr(model, 'gpt_neox'):   # Pythia
        blocks = list(model.gpt_neox.layers[layer_idx + 1:])
        ln_f = model.gpt_neox.final_layer_norm
        lm_head = model.embed_out
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):  # LLaMA
        blocks = list(model.model.layers[layer_idx + 1:])
        ln_f = model.model.norm
        lm_head = model.lm_head
    else:
        raise ValueError(f"Unknown model structure: {type(model)}")

    def head_fn(z):
        x = z.unsqueeze(1)  # (B, 1, D)
        for block in blocks:
            out = block(x)
            x = out[0] if isinstance(out, tuple) else out
        x = x.squeeze(1)
        x = ln_f(x)
        logits = lm_head(x)
        return logits[:, :output_dim]

    return head_fn


# ============================================================================
# Jacobian computation
# ============================================================================

class JacobianComputer:
    def __init__(self, model, layer_module, head_fn, pool_mode="auto", config=None):
        self.model = model
        self.layer_module = layer_module
        self.head_fn = head_fn
        self.pool_mode = pool_mode
        self.config = config or DecompositionConfig()

    def compute_task_sensitive_matrix(self, dataloader):
        device = self.config.device
        dtype = self.config.dtype
        self.model.eval()

        all_activations = []
        F_matrix = None
        n_samples = 0
        output_dim = None

        captured = {}
        def capture_hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            captured['act'] = out

        hook = self.layer_module.register_forward_hook(capture_hook)

        try:
            for batch_idx, batch in enumerate(dataloader):
                if n_samples >= self.config.n_stimuli:
                    break

                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device, dtype=dtype)
                elif isinstance(batch, dict):
                    inputs = {k: v.to(device) if torch.is_tensor(v) else v
                              for k, v in batch.items()}
                else:
                    inputs = batch.to(device, dtype=dtype)

                with torch.no_grad():
                    if isinstance(inputs, dict):
                        self.model(**inputs)
                    else:
                        self.model(inputs)

                raw_act = captured['act'].detach()
                act_pooled = pool_activations(raw_act, self.pool_mode)
                B = act_pooled.shape[0]

                all_activations.append(act_pooled.cpu())
                n = act_pooled.shape[1]

                if F_matrix is None:
                    F_matrix = torch.zeros(n, n, device=device, dtype=dtype)

                for i in range(B):
                    if n_samples >= self.config.n_stimuli:
                        break

                    z_i = act_pooled[i:i+1].detach().clone().requires_grad_(True)

                    try:
                        output_i = self.head_fn(z_i)
                    except Exception as e:
                        warnings.warn(f"head_fn failed for sample {n_samples}: {e}")
                        continue

                    if output_dim is None:
                        output_dim = output_i.shape[-1]

                    J_i = torch.zeros(output_dim, n, device=device, dtype=dtype)
                    for c in range(output_dim):
                        if z_i.grad is not None:
                            z_i.grad.zero_()
                        output_i[0, c].backward(retain_graph=(c < output_dim - 1))
                        if z_i.grad is not None:
                            J_i[c] = z_i.grad[0].detach()

                    F_matrix += J_i.t() @ J_i
                    n_samples += 1

                    if n_samples % 50 == 0:
                        print(f"    {n_samples}/{self.config.n_stimuli} samples processed")

        finally:
            hook.remove()

        F_matrix /= n_samples
        activations = torch.cat(all_activations, dim=0)[:n_samples]

        print(f"  Computed task-sensitivity matrix: {n_samples} samples, "
              f"layer dim = {F_matrix.shape[0]}, output dim = {output_dim}")

        return F_matrix.cpu(), activations, output_dim


# ============================================================================
# Subspace decomposition
# ============================================================================

@dataclass
class SubspaceDecomposition:
    V_task: torch.Tensor
    V_null: torch.Tensor
    eigenvalues: torch.Tensor
    task_dim: int
    total_dim: int
    variance_explained: float
    participation_ratio: float


def decompose_subspaces(F_matrix, config):
    n = F_matrix.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(F_matrix)
    idx = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[idx].clamp(min=0)
    eigenvectors = eigenvectors[:, idx]

    total_var = eigenvalues.sum()
    if total_var < 1e-12:
        warnings.warn("Task-sensitivity matrix is near-zero. Using r=1.")
        r = 1
    else:
        cumvar = eigenvalues.cumsum(0) / total_var
        r = int((cumvar < config.variance_threshold).sum().item()) + 1
        r = min(r, config.max_task_dim, n - config.min_null_dim)
        r = max(r, 1)

    V_task = eigenvectors[:, :r]
    V_null = eigenvectors[:, r:]

    eigs_pos = eigenvalues[eigenvalues > 1e-12]
    pr = (eigs_pos.sum() ** 2) / (eigs_pos ** 2).sum() if len(eigs_pos) > 0 else 0.0
    ve = eigenvalues[:r].sum() / total_var if total_var > 0 else 0.0

    return SubspaceDecomposition(
        V_task=V_task, V_null=V_null, eigenvalues=eigenvalues,
        task_dim=r, total_dim=n,
        variance_explained=ve.item() if torch.is_tensor(ve) else ve,
        participation_ratio=pr.item() if torch.is_tensor(pr) else pr,
    )


# ============================================================================
# Alignment metrics
# ============================================================================

def linear_cka(X, Y):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    K_X, K_Y = X @ X.t(), Y @ Y.t()
    hsic_xy = (K_X * K_Y).sum()
    denom = torch.sqrt((K_X * K_X).sum() * (K_Y * K_Y).sum())
    return (hsic_xy / denom).item() if denom > 1e-12 else 0.0


def procrustes_distance(X, Y):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    X = X / (X.norm() + 1e-12)
    Y = Y / (Y.norm() + 1e-12)
    p, q = X.shape[1], Y.shape[1]
    if p < q:
        X = F.pad(X, (0, q - p))
    elif q < p:
        Y = F.pad(Y, (0, p - q))
    U, S, Vt = torch.linalg.svd(X.t() @ Y)
    return (2.0 - 2.0 * S.sum()).clamp(min=0).sqrt().item()


def compute_alignment(Z_f, Z_g, metrics=("cka", "procrustes")):
    results = {}
    if "cka" in metrics:
        results["cka"] = linear_cka(Z_f, Z_g)
    if "procrustes" in metrics:
        results["procrustes"] = procrustes_distance(Z_f, Z_g)
    return results


# ============================================================================
# Subspace agreement
# ============================================================================

def compute_subspace_agreement(decomp_f, decomp_g):
    V_f, V_g = decomp_f.V_task.numpy(), decomp_g.V_task.numpy()
    r_f, r_g = V_f.shape[1], V_g.shape[1]
    if r_f != r_g:
        max_r = max(r_f, r_g)
        if r_f < max_r:
            V_f = np.pad(V_f, ((0, 0), (0, max_r - r_f)))
        if r_g < max_r:
            V_g = np.pad(V_g, ((0, 0), (0, max_r - r_g)))
    angles = subspace_angles(V_f, V_g)
    return {
        "principal_angles_deg": np.degrees(angles).tolist(),
        "mean_principal_angle_deg": float(np.degrees(angles).mean()),
        "grassmann_distance": float(np.sqrt((np.sin(angles) ** 2).sum())),
        "mean_cosine_similarity": float(np.cos(angles).mean()),
        "n_angles": len(angles),
    }


# ============================================================================
# Core analysis
# ============================================================================

@dataclass
class LayerResult:
    layer_name: str
    layer_idx: int
    task_dim_f: int
    task_dim_g: int
    total_dim: int
    variance_explained_f: float
    variance_explained_g: float
    participation_ratio_f: float
    participation_ratio_g: float
    global_alignment: dict
    task_alignment: dict       # each model projected onto its OWN task subspace
    null_alignment: dict       # each model projected onto its OWN null subspace
    subspace_agreement: dict


def run_decomposed_alignment(
    model_f, model_g,
    layer_specs_f, layer_specs_g,
    head_fn_factory_f,   # callable: layer_idx -> head_fn
    head_fn_factory_g,   # callable: layer_idx -> head_fn
    dataloader,
    config=None,
    pool_mode="auto",
):
    config = config or DecompositionConfig()
    results = []
    layer_names = list(layer_specs_f.keys())
    assert set(layer_names) == set(layer_specs_g.keys())

    for layer_name in layer_names:
        layer_idx = int(layer_name.split("_")[-1])

        print(f"\n{'='*60}")
        print(f"Layer: {layer_name} (idx={layer_idx})")
        print(f"{'='*60}")

        # Per-layer head functions
        head_fn_f = head_fn_factory_f(layer_idx)
        head_fn_g = head_fn_factory_g(layer_idx)

        print(f"  Computing F for model f...")
        jac_f = JacobianComputer(
            model_f, layer_specs_f[layer_name], head_fn_f, pool_mode, config)
        F_f, Z_f, k_f = jac_f.compute_task_sensitive_matrix(dataloader)

        print(f"  Computing F for model g...")
        jac_g = JacobianComputer(
            model_g, layer_specs_g[layer_name], head_fn_g, pool_mode, config)
        F_g, Z_g, k_g = jac_g.compute_task_sensitive_matrix(dataloader)

        n = F_f.shape[0]
        M = min(Z_f.shape[0], Z_g.shape[0])
        Z_f, Z_g = Z_f[:M], Z_g[:M]

        print(f"  Decomposing subspaces...")
        decomp_f = decompose_subspaces(F_f, config)
        decomp_g = decompose_subspaces(F_g, config)

        print(f"    Model f: task_dim={decomp_f.task_dim}/{n}, "
              f"var_explained={decomp_f.variance_explained:.3f}, "
              f"PR={decomp_f.participation_ratio:.1f}")
        print(f"    Model g: task_dim={decomp_g.task_dim}/{n}, "
              f"var_explained={decomp_g.variance_explained:.3f}, "
              f"PR={decomp_g.participation_ratio:.1f}")

        # Project each model's activations onto its OWN subspaces.
        # Cross-projection (applying f's eigenvectors to g's activations) is
        # ill-defined: the eigenvectors of F^f live in model f's activation
        # space and have no meaningful correspondence to model g's dimensions,
        # even when both have the same hidden size. Instead, each model is
        # projected onto its own task/null subspace, and alignment is measured
        # via CKA in stimulus space (M x M kernels), which is dimension-agnostic.
        Z_f_task = Z_f @ decomp_f.V_task @ decomp_f.V_task.t()
        Z_f_null = Z_f @ decomp_f.V_null @ decomp_f.V_null.t()
        Z_g_task = Z_g @ decomp_g.V_task @ decomp_g.V_task.t()
        Z_g_null = Z_g @ decomp_g.V_null @ decomp_g.V_null.t()

        print(f"  Computing alignment metrics...")
        global_align = compute_alignment(Z_f, Z_g, config.metrics)
        task_align = compute_alignment(Z_f_task, Z_g_task, config.metrics)
        null_align = compute_alignment(Z_f_null, Z_g_null, config.metrics)

        if decomp_f.total_dim == decomp_g.total_dim:
            sub_agree = compute_subspace_agreement(decomp_f, decomp_g)
        else:
            sub_agree = {"note": "dimensions differ, skipped"}

        print(f"  Global CKA:      {global_align.get('cka', 0):.4f}")
        print(f"  Task CKA:        {task_align.get('cka', 0):.4f}")
        print(f"  Null CKA:        {null_align.get('cka', 0):.4f}")
        print(f"  Gap (task-null):  {task_align.get('cka', 0) - null_align.get('cka', 0):.4f}")

        results.append(LayerResult(
            layer_name=layer_name, layer_idx=layer_idx,
            task_dim_f=decomp_f.task_dim, task_dim_g=decomp_g.task_dim, total_dim=n,
            variance_explained_f=decomp_f.variance_explained,
            variance_explained_g=decomp_g.variance_explained,
            participation_ratio_f=decomp_f.participation_ratio,
            participation_ratio_g=decomp_g.participation_ratio,
            global_alignment=global_align,
            task_alignment=task_align,
            null_alignment=null_align,
            subspace_agreement=sub_agree,
        ))

    return results


# ============================================================================
# Visualization
# ============================================================================

def plot_decomposed_alignment(results, model_f_name="Model A",
                               model_g_name="Model B", metric="cka",
                               save_path=None):
    layers = [r.layer_name for r in results]
    x = range(len(layers))
    global_s = [r.global_alignment[metric] for r in results]
    task_s = [r.task_alignment[metric] for r in results]
    null_s = [r.null_alignment[metric] for r in results]
    task_dims_f = [r.task_dim_f for r in results]
    task_dims_g = [r.task_dim_g for r in results]
    total_dims = [r.total_dim for r in results]
    pr_f = [r.participation_ratio_f for r in results]
    pr_g = [r.participation_ratio_g for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{model_f_name} vs {model_g_name}: Decomposed Alignment",
                 fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(x, global_s, 'k-o', lw=2, label='Global', ms=6)
    ax.plot(x, task_s, 'b-s', lw=2, label='Task-sensitive', ms=6)
    ax.plot(x, null_s, 'r-^', lw=2, label='Null space', ms=6)
    ax.fill_between(x, null_s, task_s, alpha=0.15, color='purple', label='Gap')
    ax.set_ylabel(metric.upper()); ax.set_xlabel('Layer')
    ax.set_xticks(list(x)); ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9); ax.set_title('A) Alignment by Subspace')
    ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    null_dims = [t - d for t, d in zip(total_dims, task_dims_f)]
    ax.bar(x, task_dims_f, color='steelblue', alpha=0.8, label='Task dim (f)')
    ax.bar(x, null_dims, bottom=task_dims_f, color='salmon', alpha=0.8, label='Null dim (f)')
    ax.scatter(x, task_dims_g, color='navy', marker='_', s=200, lw=2,
               label='Task dim (g)', zorder=5)
    ax.set_ylabel('Dimensions'); ax.set_xlabel('Layer')
    ax.set_xticks(list(x)); ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8); ax.set_title('B) Subspace Dimensionality'); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x, pr_f, 'b-o', lw=2, label=model_f_name, ms=6)
    ax.plot(x, pr_g, 'r-s', lw=2, label=model_g_name, ms=6)
    ax.set_ylabel('Participation Ratio of F'); ax.set_xlabel('Layer')
    ax.set_xticks(list(x)); ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9); ax.set_title('C) Effective Task Dimensionality'); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    agr = [r.subspace_agreement.get('mean_cosine_similarity', np.nan) for r in results]
    ax.plot(x, agr, 'g-D', lw=2, ms=6)
    ax.set_ylabel('Mean Cosine (principal angles)'); ax.set_xlabel('Layer')
    ax.set_xticks(list(x)); ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    ax.set_title('D) Task Subspace Agreement'); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    plt.close()


def save_results_json(results, path):
    data = []
    for r in results:
        data.append({
            "layer_name": r.layer_name, "layer_idx": r.layer_idx,
            "task_dim_f": r.task_dim_f, "task_dim_g": r.task_dim_g,
            "total_dim": r.total_dim,
            "variance_explained_f": r.variance_explained_f,
            "variance_explained_g": r.variance_explained_g,
            "participation_ratio_f": r.participation_ratio_f,
            "participation_ratio_g": r.participation_ratio_g,
            "global_alignment": r.global_alignment,
            "task_alignment": r.task_alignment,
            "null_alignment": r.null_alignment,
            "subspace_agreement": r.subspace_agreement,
        })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved results to {path}")


# ============================================================================
# Model setup helpers
# ============================================================================

def setup_vision_transformer_pair(
    model_name_f="vit_small_patch16_224",
    model_name_g="vit_small_patch16_224",
    pretrained_f=True, pretrained_g=True,
    device="cuda",
):
    import timm
    model_f = timm.create_model(model_name_f, pretrained=pretrained_f).to(device).eval()
    model_g = timm.create_model(model_name_g, pretrained=pretrained_g).to(device).eval()

    def get_vit_layers(model):
        layers = {}
        n_blocks = len(model.blocks)
        indices = np.linspace(0, n_blocks - 1, min(8, n_blocks), dtype=int)
        for idx in indices:
            layers[f"block_{idx}"] = model.blocks[idx]
        return layers

    layer_specs_f = get_vit_layers(model_f)
    layer_specs_g = get_vit_layers(model_g)

    # Return head function FACTORIES
    head_fn_factory_f = lambda idx: make_vit_head_fn(model_f, idx)
    head_fn_factory_g = lambda idx: make_vit_head_fn(model_g, idx)

    return (model_f, model_g,
            layer_specs_f, layer_specs_g,
            head_fn_factory_f, head_fn_factory_g)


def setup_llm_pair(
    model_name_f="gpt2",
    model_name_g="gpt2",
    device="cuda",
    output_dim=100,
):
    from transformers import AutoModelForCausalLM
    model_f = AutoModelForCausalLM.from_pretrained(model_name_f).to(device).eval()
    model_g = AutoModelForCausalLM.from_pretrained(model_name_g).to(device).eval()

    def get_llm_layers(model):
        layers = {}
        if hasattr(model, 'transformer'):
            blocks = model.transformer.h
        elif hasattr(model, 'gpt_neox'):
            blocks = model.gpt_neox.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            blocks = model.model.layers
        else:
            raise ValueError(f"Unknown model structure: {type(model)}")
        n_blocks = len(blocks)
        indices = np.linspace(0, n_blocks - 1, min(8, n_blocks), dtype=int)
        for idx in indices:
            layers[f"block_{idx}"] = blocks[idx]
        return layers

    layer_specs_f = get_llm_layers(model_f)
    layer_specs_g = get_llm_layers(model_g)

    head_fn_factory_f = lambda idx: make_llm_head_fn(model_f, idx, output_dim)
    head_fn_factory_g = lambda idx: make_llm_head_fn(model_g, idx, output_dim)

    return (model_f, model_g,
            layer_specs_f, layer_specs_g,
            head_fn_factory_f, head_fn_factory_g)


# ============================================================================
# Data loading
# ============================================================================

def make_imagenet_dataloader(imagenet_path, batch_size=8, num_workers=2):
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(imagenet_path, transform=transform)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def make_text_dataloader(model_name, n_samples=512, max_length=128, batch_size=8):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Simple synthetic corpus -- replace with real data for production
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning has revolutionized many fields of science.",
        "Neural networks learn hierarchical representations of data.",
        "The president announced new economic policies today.",
    ] * (n_samples // 4 + 1)
    texts = texts[:n_samples]
    encodings = tokenizer(texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length)
    dataset = torch.utils.data.TensorDataset(
        encodings['input_ids'], encodings['attention_mask'])
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False)


# ============================================================================
# Main
# ============================================================================

def run_analysis(
    model_type="vision",
    model_name_f="vit_small_patch16_224",
    model_name_g="deit_small_patch16_224",
    n_stimuli=256,
    save_dir="./results",
    imagenet_path=None,
):
    config = DecompositionConfig(n_stimuli=n_stimuli, save_dir=save_dir)

    print(f"Running decomposed alignment analysis")
    print(f"  Model type: {model_type}")
    print(f"  Model f: {model_name_f}")
    print(f"  Model g: {model_name_g}")
    print(f"  Device: {config.device}")
    print(f"  N stimuli: {config.n_stimuli}")

    if model_type == "vision":
        if imagenet_path and Path(imagenet_path).exists():
            print(f"  Loading ImageNet from {imagenet_path}")
            dataloader = make_imagenet_dataloader(
                imagenet_path, config.jacobian_batch_size, config.num_workers)
        else:
            print(f"  WARNING: No imagenet_path provided, using synthetic data")
            dataset = torch.utils.data.TensorDataset(
                torch.randn(n_stimuli, 3, 224, 224),
                torch.randint(0, 1000, (n_stimuli,)))
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=config.jacobian_batch_size, shuffle=False)

        (model_f, model_g,
         layer_specs_f, layer_specs_g,
         head_fn_factory_f, head_fn_factory_g) = setup_vision_transformer_pair(
            model_name_f, model_name_g, device=config.device)

    elif model_type == "language":
        dataloader = make_text_dataloader(
            model_name_f, n_stimuli, batch_size=config.jacobian_batch_size)
        (model_f, model_g,
         layer_specs_f, layer_specs_g,
         head_fn_factory_f, head_fn_factory_g) = setup_llm_pair(
            model_name_f, model_name_g, device=config.device)

    results = run_decomposed_alignment(
        model_f, model_g,
        layer_specs_f, layer_specs_g,
        head_fn_factory_f, head_fn_factory_g,
        dataloader, config=config,
        pool_mode="cls" if model_type == "vision" else "auto",
    )

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    tag = f"{model_name_f.split('/')[-1]}_vs_{model_name_g.split('/')[-1]}"
    save_results_json(results, f"{save_dir}/{tag}_results.json")
    plot_decomposed_alignment(
        results,
        model_f_name=model_name_f.split('/')[-1],
        model_g_name=model_name_g.split('/')[-1],
        metric="cka",
        save_path=f"{save_dir}/{tag}_alignment.png",
    )
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Decomposed representational alignment analysis")
    parser.add_argument("--model_type", choices=["vision", "language"], default="vision")
    parser.add_argument("--model_f", type=str, default="vit_small_patch16_224")
    parser.add_argument("--model_g", type=str, default="deit_small_patch16_224")
    parser.add_argument("--n_stimuli", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--imagenet_path", type=str, default=None,
                        help="Path to ImageNet validation set (e.g., /data/imagenet/val)")
    args = parser.parse_args()

    run_analysis(
        model_type=args.model_type,
        model_name_f=args.model_f,
        model_name_g=args.model_g,
        n_stimuli=args.n_stimuli,
        save_dir=args.save_dir,
        imagenet_path=args.imagenet_path,
    )