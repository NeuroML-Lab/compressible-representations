import json
import torch
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from utilz.general_utils import get_device, choose_indices, metric_key
from utilz.model_zoo import load_vit, MODEL_REGISTRY
from utilz.parsing import parse_fraction_grid, parse_layer_subset
from utilz.act_compression import build_candidate_ranks, build_projector_banks_from_eigenspaces, nearest_int_budget
from utilz.caching import build_cache_config, maybe_load_cached
from utilz.rank_allocation import solve_allocation_dp
from utilz.model_alignment import evaluate_single_layer_model_alignment, evaluate_multi_layer_model_alignment, collect_reps
from utilz.imagenet import load_imagenet_class_index_json, ImageFolderWithRemappedTargets
from utilz.alignment_metrics import compute_RSA, compute_CKA
from torch.utils.data import Subset, DataLoader


parser = argparse.ArgumentParser()
parser.add_argument("--train-dir", type=str, required=True)
parser.add_argument("--val-dir", type=str, required=True)



parser.add_argument("--model-name", type=str, default="vit_b_16", choices=list(MODEL_REGISTRY.keys()))
parser.add_argument("--compression-type", type=str, choices=["task_sensitive", "pca", "local_geometry", "geotopology"])
parser.add_argument('--layerwise', dest='layerwise', action='store_true')
parser.add_argument('--no-layerwise', dest='layerwise', action='store_false')
parser.set_defaults(layerwise=True)

parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--n_ch_samples", type=int, default=2000)
parser.add_argument("--alignment-metric", type=str, default="rsa", choices=["rsa", "cka"])

parser.add_argument("--class-index-json", type=str, default=None)
parser.add_argument("--device-id", type=str, default="auto")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-workers", type=int, default=2)

parser.add_argument("--layer-subset", type=str, default="all")
parser.add_argument("--fraction-grid", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
parser.add_argument("--retain-fraction", type=float, default=0.5)
parser.add_argument("--allocation-mode", type=str, default="utility_curve", choices=["utility_curve", "spectral_normalized"])
parser.add_argument("--utility-condition", type=str, default="raw", choices=["raw", "renorm"])
parser.add_argument("--utility-metric", type=str, default="top1", choices=["top1", "top5"])
parser.add_argument("--calibration-samples", type=int, default=512)
parser.add_argument("--eval-samples", type=int, default=2000)
parser.add_argument("--output-dir", type=str, default="")
parser.add_argument("--pin-memory", action="store_true")
parser.add_argument("--max-calibration-batches", type=int, default=None) 
parser.add_argument("--k-neighbors", type=int, default=10)
parser.add_argument("--neighbor-metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
parser.add_argument("--weight-mode", type=str, default="binary", choices=["binary", "gaussian"])
parser.add_argument("--sigma", type=float, default=None)
parser.add_argument("--distance-metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
parser.add_argument("--gt-lower-quantile", type=float, default=0.2)
parser.add_argument("--gt-upper-quantile", type=float, default=0.8)
parser.add_argument("--gt-weight-mode", type=str, default="middle_binary",
                    choices=["middle_binary", "middle_triangular", "paper_gt"])

args = parser.parse_args()

compression_shorthands = {
    "task_sensitive": "task",
    "pca": "pca",
    "local_geometry": "local",
    "geotopology": "gt",
    "random": "random"
}

def compute_normalized_frob_dist(matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> float:
    """
    Compute the normalized Frobenius distance between two matrices.
    The distance is normalized by the frob_norm of matrix_a

    Args:
        matrix_a (torch.Tensor): First matrix of shape (n_samples, n_features).
        matrix_b (torch.Tensor): Second matrix of shape (n_samples, n_features).

    Returns:
        float: Normalized Frobenius distance between the two matrices.
    """
    assert matrix_a.shape == matrix_b.shape, "Input matrices must have the same shape"
    assert matrix_a.dtype == matrix_b.dtype, "Input matrices must have the same dtype"

    frob_dist = torch.linalg.norm(matrix_a - matrix_b, ord='fro')
    normalization_factor = torch.linalg.norm(matrix_a, ord='fro')
    normalized_frob_dist = frob_dist / normalization_factor
    return normalized_frob_dist

def main():

    # load model
    device = get_device(args.device_id)
    loaded = load_vit(args.model_name, device)
    model = loaded.model
    preprocess = loaded.preprocess
    layer_indices = parse_layer_subset('all', loaded.num_layers)
    
    # load data
    outdir = Path(f'{args.output_dir}/{args.model_name}/')
    if not outdir.exists():
        outdir.mkdir(parents=True)

    class_index_json = load_imagenet_class_index_json(None, outdir)
    val_ds = ImageFolderWithRemappedTargets(args.val_dir, preprocess, class_index_json)
    eval_idx = choose_indices(len(val_ds), args.eval_samples, seed=args.seed + 1)
    eval_ds = Subset(val_ds, eval_idx)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)


    # get fraction ranks
    fraction_grid = parse_fraction_grid(args.fraction_grid)
    candidate_ranks = build_candidate_ranks(loaded.hidden_dim, fraction_grid)
    utility_key = metric_key(args.utility_metric)
    
    comp_types = ["task_sensitive", "pca", "local_geometry", "geotopology", "random"]
    
    base_model_reps = {li: None for li in layer_indices}

    for layer_idx in layer_indices:
        print(f"Collecting base model representations for layer {layer_idx} ...")
        base_model_reps[layer_idx] = collect_reps(model=model, loader=eval_loader, device=device, layer_idx=layer_idx, cls_token=True)
    
    compressed_result_path = f"{outdir}/self_align_layerwise.csv"
    alignment_rows = []
    # check if results already exist
    if Path(compressed_result_path).exists():
        print(f"Compressed self-alignment results already exist at {compressed_result_path}, loading ...")
        alignment_df = pd.read_csv(compressed_result_path)
        print(f"Loaded compressed self-alignment results with shape {alignment_df.shape}")
        # remove comp_types already present
        existing_comp_types = set(alignment_df['compression_type'].unique())
        comp_types = [ct for ct in comp_types if ct not in existing_comp_types]
        alignment_rows = alignment_df.to_dict('records')



    for ci, comp_type in enumerate(comp_types):
        print(f"\n\n=== Compression type: {comp_type} ({ci+1}/{len(comp_types)}) ===")
        REPEAT_ITERS = 1 if comp_type != "random" else 3

        for repeat_iter in range(REPEAT_ITERS):
            # get cached (of all layers)
            comp = compression_shorthands[comp_type]
            utility_cache_dir = f"vit_results_v3/{args.model_name}/{comp}_cache"
            if comp_type == "random":
                utility_cache_dir += f"_seed{repeat_iter}"

            cache_dir = Path(utility_cache_dir)
            cache_config = build_cache_config(args, layer_indices, loaded.hidden_dim, fraction_grid, comp_type)
            print(f"Looking for cache in {cache_dir} with config: {cache_config} ...")
            cached = maybe_load_cached(cache_dir, cache_config, comp_type)
            print(f'Cache hit: {cached is not None}')

            if cached:
                utility_df, spectra_df, eigvals_by_layer, eigvecs_by_layer = cached
                projectors_by_layer, null_projectors_by_layer = build_projector_banks_from_eigenspaces(
                        eigvecs_by_layer=eigvecs_by_layer,
                        candidate_ranks=candidate_ranks,
                        hidden_dim=loaded.hidden_dim,
                    )
            else:
                raise ValueError(f"No cache found at {cache_dir}, need to compute the representations again ...")
            
            print("Evaluating self alignment for compressed representations across layers ...")
            

            for layer_idx in layer_indices:
                if base_model_reps[layer_idx] is None:
                    print(f"Collecting base model representations for layer {layer_idx} ...")
                    base_model_reps[layer_idx] = collect_reps(model=model, loader=eval_loader, device=device, layer_idx=layer_idx, cls_token=True)
                
                
                print(f"Evaluating layer {layer_idx} / {loaded.num_layers} ...")
                for rank in candidate_ranks:
                    P = projectors_by_layer[layer_idx][rank]
                    null_P = null_projectors_by_layer[layer_idx][rank]
                    if isinstance(P, torch.Tensor):
                        P = P.detach().cpu().numpy()
                    if isinstance(null_P, torch.Tensor):
                        null_P = null_P.detach().cpu().numpy()


                    

                    proj = base_model_reps[layer_idx] @ P

                    # renorm
                    src_norm = np.linalg.norm(base_model_reps[layer_idx], axis=1, keepdims=True) 
                    src_norm = np.where(src_norm < 1e-8, 1e-8, src_norm)

                    proj_norm = np.linalg.norm(proj, axis=1, keepdims=True)
                    proj_norm = np.where(proj_norm < 1e-8, 1e-8, proj_norm)
                    proj_reps = proj * (src_norm / proj_norm)

                    null_proj = base_model_reps[layer_idx] @ null_P
                    null_proj_norm = np.linalg.norm(null_proj, axis=1, keepdims=True)
                    null_proj_norm = np.where(null_proj_norm < 1e-8, 1e-8, null_proj_norm)
                    
                    null_proj_reps = null_proj * (src_norm / null_proj_norm)

                    rsa_renorm = compute_RSA(proj_reps, base_model_reps[layer_idx])
                    cka_renorm = compute_CKA(proj_reps, base_model_reps[layer_idx])
                    rsa_null_renorm = compute_RSA(null_proj_reps, base_model_reps[layer_idx])
                    cka_null_renorm = compute_CKA(null_proj_reps, base_model_reps[layer_idx])

                    
                    norm_rep_distance = compute_normalized_frob_dist(
                        torch.tensor(base_model_reps[layer_idx]),
                        torch.tensor(proj_reps)
                    ).item()
                    null_norm_rep_distance = compute_normalized_frob_dist(
                        torch.tensor(base_model_reps[layer_idx]),
                        torch.tensor(null_proj_reps)
                    ).item()
                    
                    alignment_rows.append({
                        "compression_type": comp_type,
                        "repeat_iter": repeat_iter,
                        "layer_idx": layer_idx,
                        "rank": rank,
                        "retain_fraction_layer": float(rank / loaded.hidden_dim),

                        # proj renorm
                        "rsa_renorm": rsa_renorm,
                        "cka_renorm": cka_renorm,

                        # null-proj renorm
                        "null_rsa_renorm": rsa_null_renorm,
                        "null_cka_renorm": cka_null_renorm,

                        # rep distances
                        "norm_rep_distance": norm_rep_distance,
                        "null_norm_rep_distance": null_norm_rep_distance,

                        "n_eval_samples": args.eval_samples,
                        "n_ch_samples": args.n_ch_samples,
                    })

            alignment_df = pd.DataFrame(alignment_rows)
            alignment_df.to_csv(compressed_result_path)

    
        

if __name__ == "__main__":
    main()