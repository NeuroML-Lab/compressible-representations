import json
import torch
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from utilz.alignment_metrics import compute_CKA, compute_RSA
from utilz.general_utils import get_device, choose_indices, metric_key
from utilz.model_zoo import load_vit, MODEL_REGISTRY
from utilz.parsing import parse_fraction_grid, parse_layer_subset
from utilz.act_compression import build_candidate_ranks, build_projector_banks_from_eigenspaces, nearest_int_budget
from utilz.caching import build_cache_config, maybe_load_cached
from utilz.rank_allocation import solve_allocation_dp
from utilz.model_alignment import collect_reps, compute_subspace_agreement
from utilz.imagenet import load_imagenet_class_index_json, ImageFolderWithRemappedTargets
from torch.utils.data import Subset, DataLoader
from itertools import combinations_with_replacement


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
    
    

    print("Evaluating layerwise alignment for compressed representations ...")

    # --- build projector banks for each compression type to compare ---
    comp_types_to_compare = ["task_sensitive", "pca", "local_geometry", "geotopology", "random"]

    old_layer_indices = layer_indices
    old_candidate_ranks = candidate_ranks


    # DRY RUN
    # comp_types_to_compare = comp_types_to_compare[:2]
    # layer_indices = layer_indices[:2]
    # candidate_ranks = candidate_ranks[:2]




    # load caches and build projectors for each compression type
    all_projected = {}   # comp_type -> layer_idx -> rank -> [N, 768]
    raw_reps = {}
    eigvec_banks = {}

    raw_reps = {li: None for li in layer_indices}
    base_model_reps_path = f"vit_results_v3/{args.model_name}/base_model_reps.pt"

    
    for layer_idx in layer_indices:
        print(f"Collecting base model representations for layer {layer_idx} ...")
        raw_reps[layer_idx] = collect_reps(model=model, loader=eval_loader, device=device, layer_idx=layer_idx, cls_token=True)

    for li in layer_indices:
        raw_reps[li] = collect_reps(model, eval_loader, device, li, cls_token=True)

    
    

    compressed_result_path = f"{outdir}/cross_compression_alignment.csv"

    for ct in comp_types_to_compare:
        print(f"Processing compression type: {ct} ...")
        ct_short = compression_shorthands[ct]
        num_repeats = 2 if ct == "random" else 1

        for ri in range(num_repeats):
            ct_cache_dir = Path(f"vit_results_v3/{args.model_name}/{ct_short}_cache")
            if ct == "random":
                ct_cache_dir = Path(f"vit_results_v3/{args.model_name}/{ct_short}_cache_seed{ri}")

            ct_cache_config = build_cache_config(args, old_layer_indices, loaded.hidden_dim, fraction_grid, ct)
            ct_cached = maybe_load_cached(ct_cache_dir, ct_cache_config, ct)
            if ct_cached is None:
                raise ValueError(f"No cache found for compression type: {ct}")

            _, _, _, ct_eigvecs = ct_cached
            ct_projectors, _ = build_projector_banks_from_eigenspaces(
                eigvecs_by_layer=ct_eigvecs,
                candidate_ranks=old_candidate_ranks,
                hidden_dim=loaded.hidden_dim,
            )
            # store eigvecs for later subspace agreement analysis
            ct_new = f"{ct_short}_seed{ri}" if ct == "random" else ct_short
            eigvec_banks[ct_new] = {}

            # apply projectors
            all_projected[ct_new] = {}
            for li in layer_indices:
                all_projected[ct_new][li] = {}
                eigvec_banks[ct_new][li] = {}
                for r in candidate_ranks:
                    P = ct_projectors[li][r]
                    if isinstance(P, torch.Tensor):
                        P = P.detach().cpu().numpy()
                    proj = raw_reps[li] @ P
                    norms = np.linalg.norm(proj, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-8, 1.0, norms)
                    all_projected[ct_new][li][r] = proj / norms

                    eigvec_banks[ct_new][li][r] = ct_eigvecs[li][:, :r].detach().cpu().numpy()



   
   
   
   
    print("All projected representations computed, now evaluating pairwise alignment across layers and compression types ...")
    # --- build pairwise keys over all (comp_type, layer, rank) combinations ---
    keys = [(ct, li, r) for ct in comp_types_to_compare
                        for li in layer_indices
                        for r in candidate_ranks]


   


    alignment_rows = []
    comp_types_to_compare = list(all_projected.keys())  # update to actual keys used after loading caches (handles random repeats)

    total_pairs = len(comp_types_to_compare) * len(layer_indices) * len(candidate_ranks)
    print(f"Total pairs to compute: ~ {total_pairs}. Yikes..")
    computed_pairs = 0

    for li in layer_indices:
        keys_at_layer = [(ct, r) for ct in comp_types_to_compare for r in candidate_ranks]

        for idx_a, (ct_1, r1) in enumerate(keys_at_layer):
            for idx_b, (ct_2, r2) in enumerate(keys_at_layer):
                if idx_b < idx_a:
                    continue  # upper triangle only

                if ct_1 == ct_2 and r1 == r2:
                    continue  # skip self-pairs

                X = all_projected[ct_1][li][r1]
                Y = all_projected[ct_2][li][r2]

                rsa_val = compute_RSA(X, Y)
                cka_val = compute_CKA(X, Y)

                if r1 == r2:
                    U1 = eigvec_banks[ct_1][li][r1]
                    U2 = eigvec_banks[ct_2][li][r2]
                    sa = compute_subspace_agreement(U1, U2)
                    mean_angle = sa["mean_principal_angle_deg"]
                    grassmann  = sa["grassmann_distance"]
                    mean_cos   = sa["mean_cosine_similarity"]
                else:
                    mean_angle = grassmann = mean_cos = np.nan

                row = {
                    "m1_comp_type": ct_1, "m1_layer_idx": li, "m1_rank": r1,
                    "m1_retain_fraction_layer": float(r1 / loaded.hidden_dim),
                    "m2_comp_type": ct_2, "m2_layer_idx": li, "m2_rank": r2,
                    "m2_retain_fraction_layer": float(r2 / loaded.hidden_dim),
                    "rsa_renorm": rsa_val, "cka_renorm": cka_val,
                    "mean_principal_angle_deg": mean_angle,
                    "grassmann_distance": grassmann,
                    "mean_cosine_similarity": mean_cos,
                    "n_eval_samples": args.eval_samples,
                }
                alignment_rows.append(row)

            computed_pairs += 1
            if computed_pairs % 50 == 0:
                print(f"  {computed_pairs}/{total_pairs} pairs done ...")
            
    alignment_df = pd.DataFrame(alignment_rows)
    alignment_df.to_csv(compressed_result_path, index=False)



                
    
        

if __name__ == "__main__":
    main()