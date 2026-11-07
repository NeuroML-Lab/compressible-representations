import json
import torch
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from utilz.utilz.alignment_metrics import compute_CKA, compute_RSA
from utilz.utilz.general_utils import get_device, choose_indices, metric_key
from utilz.utilz.model_zoo import load_model, MODEL_REGISTRY
from utilz.utilz.parsing import parse_fraction_grid, parse_layer_subset
from utilz.utilz.act_compression import build_candidate_ranks, build_projector_banks_from_eigenspaces, nearest_int_budget
from utilz.utilz.caching import build_cache_config, maybe_load_cached
from utilz.utilz.rank_allocation import solve_allocation_dp
from utilz.utilz.model_alignment import evaluate_single_layer_cross_model_alignment, collect_reps, compute_subspace_agreement
from utilz.utilz.imagenet import load_imagenet_class_index_json, ImageFolderWithRemappedTargets
from torch.utils.data import Subset, DataLoader


parser = argparse.ArgumentParser()
parser.add_argument("--train-dir", type=str, required=True)
parser.add_argument("--val-dir", type=str, required=True)



parser.add_argument("--model1-name", type=str, default="vit_b_16", choices=list(MODEL_REGISTRY.keys()))
parser.add_argument("--model2-name", type=str, default="vit_b_16", choices=list(MODEL_REGISTRY.keys()))

parser.add_argument('--layerwise', dest='layerwise', action='store_true')
parser.add_argument('--no-layerwise', dest='layerwise', action='store_false')
parser.set_defaults(layerwise=True)

parser.add_argument("--seed", type=int, default=0)

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
    "geotopology": "gt"
}


def main():

    # load models
    device = get_device(args.device_id)
    loaded = load_model(args.model1_name, device)
    model1 = loaded.model
    preprocess1 = loaded.preprocess
    model1_num_layers = loaded.num_layers
    model1_hidden_dim = loaded.hidden_dim

    loaded = load_model(args.model2_name, device)
    model2 = loaded.model
    preprocess2 = loaded.preprocess
    model2_num_layers = loaded.num_layers
    model2_hidden_dim = loaded.hidden_dim

    if model1_num_layers != model2_num_layers:
        raise ValueError(f"Models have unequal number of layers: {model1_num_layers} vs {model2_num_layers}")
    
    if model1_hidden_dim != model2_hidden_dim:
        raise ValueError(f"Models have unequal hidden dimensions: {model1_hidden_dim} vs {model2_hidden_dim}")
    
    layer_indices = parse_layer_subset('all', loaded.num_layers)

    print(f"Preprocessing is the same: {preprocess1 == preprocess2}")
    preprocess = preprocess1
    print(f"Model 1 hidden dim: {model1_hidden_dim}, Model 2 hidden dim: {model2_hidden_dim}")



    # load data
    outdir = Path(f'{args.output_dir}/{args.model1_name}_vs_{args.model2_name}/')
    if not outdir.exists():
        outdir.mkdir(parents=True)

    class_index_json = load_imagenet_class_index_json(None, outdir)
    val_ds = ImageFolderWithRemappedTargets(args.val_dir, preprocess, class_index_json)
    eval_idx = choose_indices(len(val_ds), args.eval_samples, seed=args.seed + 1)
    eval_ds = Subset(val_ds, eval_idx)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)


    # get fraction ranks
    fraction_grid = parse_fraction_grid(args.fraction_grid)
    candidate_ranks = build_candidate_ranks(model1_hidden_dim, fraction_grid)


    # --- build projector banks for each compression type to compare ---
    comp_types_to_compare = ["task_sensitive", "pca", "local_geometry", "geotopology"]
    raw_reps_model1 = {}
    raw_reps_model2 = {}

    print(f"Extracting raw representations for all layers before projection ...")
    for li in layer_indices:
        raw_reps_model1[li] = collect_reps(model1, eval_loader, device, li, cls_token=True)
        raw_reps_model2[li] = collect_reps(model2, eval_loader, device, li, cls_token=True)


    cross_align_result_path = f"{outdir}/cross_model_alignment_results.csv"

    projected_reps_model1 = {}
    projected_reps_model2 = {}
    null_reps_model1 = {}
    null_reps_model2 = {}

    for ci, ct in enumerate(comp_types_to_compare):
        print(f"Processing compression type: {ct} ...")
        ct_short = compression_shorthands[ct]
        ct_cache_dir1 = Path(f"vit_results_v2/{args.model1_name}/{ct_short}_cache")
        ct_cache_dir2 = Path(f"vit_results_v2/{args.model2_name}/{ct_short}_cache")
        
        ct_cache_config1 = build_cache_config(args, layer_indices, model1_hidden_dim, fraction_grid, ct, multi_model_index=1)
        ct_cache_config2 = build_cache_config(args, layer_indices, model2_hidden_dim, fraction_grid, ct, multi_model_index=2)
        ct_cached_1 = maybe_load_cached(ct_cache_dir1, ct_cache_config1, ct)
        ct_cached_2 = maybe_load_cached(ct_cache_dir2, ct_cache_config2, ct)

        if ct_cached_1 is None or ct_cached_2 is None:
            raise ValueError(f"No cache found for compression type: {ct} ")
        
        print(f'Cache hit for model 1: {ct_cached_1 is not None}, Cache hit for model 2: {ct_cached_2 is not None}')
        
        _, _, _, ct_eigvecs_1 = ct_cached_1
        _, _, _, ct_eigvecs_2 = ct_cached_2

        ct_projectors_1, ct_null_projectors_1 = build_projector_banks_from_eigenspaces(
            eigvecs_by_layer=ct_eigvecs_1,
            candidate_ranks=candidate_ranks,
            hidden_dim=model1_hidden_dim,
        )
        ct_projectors_2, ct_null_projectors_2 = build_projector_banks_from_eigenspaces(
            eigvecs_by_layer=ct_eigvecs_2,
            candidate_ranks=candidate_ranks,
            hidden_dim=model2_hidden_dim,
        )

        projected_reps_model1[ct] = {}
        projected_reps_model2[ct] = {}
        null_reps_model1[ct] = {}
        null_reps_model2[ct] = {}


        for li in layer_indices:
            projected_reps_model1[ct][li] = {}
            projected_reps_model2[ct][li] = {}
            null_reps_model1[ct][li] = {}
            null_reps_model2[ct][li] = {}

            for r in candidate_ranks:
                P = ct_projectors_1[li][r]
                if isinstance(P, torch.Tensor):
                    P = P.detach().cpu().numpy()
                proj = raw_reps_model1[li] @ P
                norms = np.linalg.norm(proj, axis=1, keepdims=True)
                norms = np.where(norms < 1e-8, 1.0, norms)
                projected_reps_model1[ct][li][r] = proj / norms

                NP = ct_null_projectors_1[li][r]
                if isinstance(NP, torch.Tensor):
                    NP = NP.detach().cpu().numpy()
                null_proj = raw_reps_model1[li] @ NP
                null_norms = np.linalg.norm(null_proj, axis=1, keepdims=True)
                null_norms = np.where(null_norms < 1e-8, 1.0, null_norms)
                null_reps_model1[ct][li][r] = null_proj / null_norms


                P = ct_projectors_2[li][r]
                if isinstance(P, torch.Tensor):
                    P = P.detach().cpu().numpy()
                proj = raw_reps_model2[li] @ P
                norms = np.linalg.norm(proj, axis=1, keepdims=True)
                norms = np.where(norms < 1e-8, 1.0, norms)
                projected_reps_model2[ct][li][r] = proj / norms

                NP = ct_null_projectors_2[li][r]
                if isinstance(NP, torch.Tensor):
                    NP = NP.detach().cpu().numpy()
                null_proj = raw_reps_model2[li] @ NP
                null_norms = np.linalg.norm(null_proj, axis=1, keepdims=True)
                null_norms = np.where(null_norms < 1e-8, 1.0, null_norms)
                null_reps_model2[ct][li][r] = null_proj / null_norms

    print(f"Finished computing projected & null representations")


    

    total = len(comp_types_to_compare) * len(layer_indices) * len(candidate_ranks)
    done = 0

    alignment_rows = []
    for ct in comp_types_to_compare:
        for li in layer_indices:
            for r in candidate_ranks:
                X = projected_reps_model1[ct][li][r]   # [N, r]
                Y = projected_reps_model2[ct][li][r]   # [N, r]

                rsa_val = compute_RSA(X, Y)
                cka_val = compute_CKA(X, Y)

                X_null = null_reps_model1[ct][li][r]
                Y_null = null_reps_model2[ct][li][r]

                rsa_null_val = compute_RSA(X_null, Y_null)
                cka_null_val = compute_CKA(X_null, Y_null)

                row = {
                "comp_type":                 ct,
                "layer_idx":                 li,
                "rank":                      r,
                "retain_fraction_layer":     float(r / model1_hidden_dim),
                "rsa_renorm":                rsa_val,
                "cka_renorm":                cka_val,
                "rsa_null":                  rsa_null_val,
                "cka_null":                  cka_null_val,
                "n_eval_samples":            args.eval_samples,
                }
                alignment_rows.append(row)

                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{total} done")
            
            # compute raw alignment for this layer (no projection)
            X_raw = raw_reps_model1[li]
            Y_raw = raw_reps_model2[li]
            rsa_raw_val = compute_RSA(X_raw, Y_raw)
            cka_raw_val = compute_CKA(X_raw, Y_raw)
            
            row = {
                "comp_type":                 ct,
                "layer_idx":                 li,
                "rank":                      -1,
                "retain_fraction_layer":     1.0,
                "rsa_renorm":                rsa_raw_val,
                "cka_renorm":                cka_raw_val,
                "rsa_null":                  rsa_raw_val,
                "cka_null":                  cka_raw_val,
                "n_eval_samples":            args.eval_samples,
            }
            alignment_rows.append(row)

    alignment_df = pd.DataFrame(alignment_rows)
    alignment_df.to_csv(cross_align_result_path, index=False)
    print(f"Saved cross-model alignment results to: {cross_align_result_path}")
    










                
    
        

if __name__ == "__main__":
    main()