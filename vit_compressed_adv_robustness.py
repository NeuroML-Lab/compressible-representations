import json
import torch
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List
from torch.utils.data import Subset, DataLoader
from utilz.general_utils import get_device, choose_indices, metric_key
from utilz.model_zoo import load_vit, MODEL_REGISTRY
from utilz.model_evals import evaluate_single_layer_projection
from utilz.model_adv_evals import generate_attack_cache_on_fullmodel, evaluate_single_layer_projection_adv, evaluate_multi_layer_projection_adv
from utilz.parsing import parse_fraction_grid, parse_layer_subset
from utilz.imagenet import load_imagenet_class_index_json, load_imagenet_dataloader
from utilz.act_compression import build_candidate_ranks, build_projector_banks_from_eigenspaces, nearest_int_budget
from utilz.caching import build_cache_config, maybe_load_cached
from utilz.rank_allocation import solve_allocation_dp

parser = argparse.ArgumentParser()
parser.add_argument("--train-dir", type=str, required=True)
parser.add_argument("--val-dir", type=str, required=True)

parser.add_argument("--compression-type", type=str, choices=["task_sensitive", "pca", "local_geometry", "geotopology", "random"])
parser.add_argument("--attack-type", type=str, choices=["fgsm", "pgd20", "pgd100", "fgsm", "autoattack"])
parser.add_argument("--attack-target", type=str, choices=["full_model", "compressed_model"])
parser.add_argument('--layerwise', dest='layerwise', action='store_true')
parser.add_argument('--no-layerwise', dest='layerwise', action='store_false')
parser.set_defaults(layerwise=True)
parser.add_argument("--class-index-json", type=str, default=None)
parser.add_argument("--model-name", type=str, default="vit_b_16", choices=list(MODEL_REGISTRY.keys()))
parser.add_argument("--device-id", type=str, default="auto")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--layer-subset", type=str, default="all")
parser.add_argument("--fraction-grid", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
parser.add_argument("--retain-fraction", type=float, default=0.5)
parser.add_argument("--allocation-mode", type=str, default="utility_curve", choices=["utility_curve", "fixed"])
parser.add_argument("--utility-condition", type=str, default="raw", choices=["raw", "renorm"])
parser.add_argument("--utility-metric", type=str, default="top1", choices=["top1", "top5"])
parser.add_argument("--calibration-samples", type=int, default=5000)
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
    layer_indices = parse_layer_subset(args.layer_subset, loaded.num_layers)
    all_layer_indices = parse_layer_subset('all', loaded.num_layers)

    comp = compression_shorthands[args.compression_type]
    seed_str = "" if comp != "random" else f"_seed{args.seed}"
    print(seed_str)
    utility_cache_dir = f"vit_results_v3/{args.model_name}/{comp}_cache{seed_str}"
    cache_dir = Path(utility_cache_dir)
    
    # load ds
    outdir = Path(args.output_dir)
    class_index_json = load_imagenet_class_index_json(args.class_index_json, outdir)

    eval_loader = load_imagenet_dataloader(
        data_dir=args.val_dir,
        class_index_json=class_index_json,
        transform=preprocess,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_size=args.eval_samples,
        balance_classes=True,
        shuffle=False,
        seed=0
    )

    # get fraction ranks
    fraction_grid = parse_fraction_grid(args.fraction_grid)
    candidate_ranks = build_candidate_ranks(loaded.hidden_dim, fraction_grid)
    utility_key = metric_key(args.utility_metric)
    
    # get cached (of all layers)
    cache_config = build_cache_config(args, all_layer_indices, loaded.hidden_dim, fraction_grid, args.compression_type)
    cached = maybe_load_cached(cache_dir, cache_config, args.compression_type)

    comp = compression_shorthands[args.compression_type]
    utility_condition = f"{comp}_{args.utility_condition}"

    if cached:
        utility_df, spectra_df, eigvals_by_layer, eigvecs_by_layer = cached
        projectors_by_layer, null_projectors_by_layer = build_projector_banks_from_eigenspaces(
                eigvecs_by_layer=eigvecs_by_layer,
                candidate_ranks=candidate_ranks,
                hidden_dim=loaded.hidden_dim,
            )
    else:
        raise ValueError("No cache found, need to compute the representations again ...")
    
    
    baseline = evaluate_single_layer_projection(model=model, loader=eval_loader, device=device, layer_idx=layer_indices[0], projector=None, renorm=False)
    print(f"Baseline: {baseline}")
    ### ADV ATTACKS ###
    attack_cache = None
    if args.attack_target == "full_model":
        attack_cache = generate_attack_cache_on_fullmodel(model=model, loader=eval_loader, device=device, attack_type=args.attack_type)
        print(f"Attacks generated using {args.attack_type}")
        
    baseline_adv = evaluate_single_layer_projection_adv(model=model, loader=eval_loader, device=device, attack_type=args.attack_type,
                                                            layer_idx=layer_indices[0], projector=None,
                                                            renorm=False, subspace_label=args.compression_type, adv_images_cache=attack_cache).to_dict()
    print(f"Robust Baseline: {baseline_adv}")
            

    attack_rows = []
    
    attack_result_path = None
    

    if args.layerwise:
        attack_result_path = args.output_dir + f"/{comp}{seed_str}_{args.attack_type}.csv"
        print(f"Saving layerwise attack results to {attack_result_path}")
        for layer_idx in layer_indices:
            print(f"Evaluating attacks for layer {layer_idx} ...")
            for rank in candidate_ranks:

                adv_res_renorm = evaluate_single_layer_projection_adv(model=model, loader=eval_loader, device=device, attack_type=args.attack_type,
                                                                layer_idx=layer_idx, projector=projectors_by_layer[layer_idx][rank],
                                                                renorm=True, subspace_label=args.compression_type, rank=rank, adv_images_cache=attack_cache).to_dict()
                adv_res_null_renorm = evaluate_single_layer_projection_adv(model=model, loader=eval_loader, device=device, attack_type=args.attack_type,
                                                                layer_idx=layer_idx, projector=null_projectors_by_layer[layer_idx][rank],
                                                                renorm=True, subspace_label=args.compression_type, rank=rank, adv_images_cache=attack_cache).to_dict()
                
                attack_rows.append({
                    "layer_index": layer_idx,
                    "hidden_dim": loaded.hidden_dim,
                    "rank": int(rank),
                    "retain_fraction_layer": float(rank / loaded.hidden_dim),
                    "attack": adv_res_renorm['attack_type'],
                    "baseline_top1": float(baseline['top1']),
                    "baseline_top5": float(baseline['top5']),
                    "robust_baseline_top1": float(baseline_adv['robust_top1']),
                    "robust_baseline_top5": float(baseline_adv['robust_top5']),
                    "n_eval": baseline['n_eval'],

                    # compressed clean
                    "top1_renorm": float(adv_res_renorm["clean_top1"]),
                    "top5_renorm": float(adv_res_renorm["clean_top5"]),

                    # compressed robust
                    "robust_top1_renorm": float(adv_res_renorm["robust_top1"]),
                    "robust_top5_renorm": float(adv_res_renorm["robust_top5"]),

                    # compressed robust metrics
                    "top1_drop_renorm": float(adv_res_renorm["top1_accuracy_drop"]),
                    "top5_drop_renorm": float(adv_res_renorm["top5_accuracy_drop"]),
                    
                    
                    # null clean
                    "top1_null_renorm": float(adv_res_null_renorm["clean_top1"]),
                    "top5_null_renorm": float(adv_res_null_renorm["clean_top5"]),

                    # null robust
                    "robust_top1_null_renorm": float(adv_res_null_renorm["robust_top1"]),
                    "robust_top5_null_renorm": float(adv_res_null_renorm["robust_top5"]),

                    # compressed robust metrics
                    "top1_drop_null_renorm": float(adv_res_null_renorm["top1_accuracy_drop"]),
                    "top5_drop_null_renorm": float(adv_res_null_renorm["top5_accuracy_drop"]),
                    
                    "retention_ratio_null_renorm": float(adv_res_null_renorm["robustness_retention_ratio"])
                    })
        
        attack_df = pd.DataFrame(attack_rows)
        attack_df.to_csv(attack_result_path)

    else:
        # global projections
        attack_result_path = args.output_dir + f"/{comp}{seed_str}_{args.attack_type}_global.csv"
        print(f"Saving global attack results to {attack_result_path}")
        for r_frac in fraction_grid:
            alloc_path = args.output_dir + f"/allocation/{comp}_{r_frac}.csv"
            Path(alloc_path).parent.mkdir(parents=True, exist_ok=True)
            total_dims = loaded.hidden_dim * len(layer_indices)

            if args.allocation_mode == "fixed":
                allocation_rows = []
            
                for layer_idx in layer_indices:
                    k_l = max(1, round(loaded.hidden_dim * r_frac))
                    retain = k_l / loaded.hidden_dim
                    allocation_rows.append({
                        "layer_index": layer_idx,
                        "hidden_dim": loaded.hidden_dim,
                        "k_l": k_l,
                        "retain_fraction_layer": retain,
                        "allocation_mode": args.allocation_mode,
                        "utility_score_used_for_allocation": float("nan"),
                    })
                
            elif args.allocation_mode == "utility_curve":
                if utility_df.empty:
                    raise RuntimeError("utility_curve allocation requested, but cached utility curves are empty.")
                layer_candidates: Dict[int, List[Dict[str, float]]] = {}
                for layer_idx in layer_indices:
                    sub = utility_df[utility_df["layer_index"] == layer_idx].sort_values("rank")
                    cand_list = []
                    for _, row in sub.iterrows():
                        util = float(row[f"{utility_key}_{utility_condition}"])
                        cand_list.append({
                            "rank": int(row["rank"]),
                            "utility": util,
                            "retain_fraction_layer": float(row["retain_fraction_layer"]),
                        })
                    layer_candidates[layer_idx] = cand_list

                min_required = sum(min(int(c["rank"]) for c in layer_candidates[l]) for l in layer_indices)
                budget_dims = nearest_int_budget(total_dims, r_frac)
                print(f"total_dims: {total_dims}")
                print(f"retain_fraction: {r_frac}")
                print(f"budget_dims: {budget_dims}")
                if budget_dims < min_required:
                    print(f"Mismatch in budget and min dims required: {(min_required, budget_dims)} for layer {layer_idx} r_frac {r_frac}")
                budget_dims = max(budget_dims, min_required)
                
                # print("layer_candidates:", layer_candidates)
                chosen = solve_allocation_dp(layer_candidates, budget_dims=budget_dims, utility_metric="utility")
                allocation_rows = []
                for layer_idx in layer_indices:
                    c = chosen[layer_idx]
                    allocation_rows.append({
                        "layer_index": layer_idx,
                        "hidden_dim": loaded.hidden_dim,
                        "k_l": int(c["rank"]),
                        "retain_fraction_layer": float(c["retain_fraction_layer"]),
                        "allocation_mode": args.allocation_mode,
                        "utility_score_used_for_allocation": float(c["utility"]),
                    })
            allocation_df = pd.DataFrame(allocation_rows).sort_values("layer_index").reset_index(drop=True)
            allocation_df.to_csv(alloc_path)
            print("Allocated k_l by layer:", allocation_df["k_l"].tolist())
            
            projectors: Dict[int, torch.Tensor] = {}
            null_projectors: Dict[int, torch.Tensor] = {}
            for _, row in allocation_df.iterrows():
                layer_idx = int(row["layer_index"])
                k = int(row["k_l"])
                projectors[layer_idx] = projectors_by_layer[layer_idx][k]
                null_projectors[layer_idx] = null_projectors_by_layer[layer_idx][k]

            adv_res_renorm = evaluate_multi_layer_projection_adv(model=model, loader=eval_loader, device=device, attack_type=args.attack_type,
                                                            projectors=projectors,
                                                            renorm=True, subspace_label=args.compression_type, adv_images_cache=attack_cache).to_dict()
            adv_res_null_renorm = evaluate_multi_layer_projection_adv(model=model, loader=eval_loader, device=device, attack_type=args.attack_type,
                                                            projectors=null_projectors,
                                                            renorm=True, subspace_label=args.compression_type, adv_images_cache=attack_cache).to_dict()
            
            attack_rows.append({
                "hidden_dim": loaded.hidden_dim,
                "retain_fraction_total": r_frac,
                "attack": adv_res_renorm['attack_type'],
                "baseline_top1": float(baseline['top1']),
                "baseline_top5": float(baseline['top5']),
                "robust_baseline_top1": float(baseline_adv['robust_top1']),
                "robust_baseline_top5": float(baseline_adv['robust_top5']),
                "n_eval": baseline['n_eval'],

                # compressed clean
                "top1_renorm": float(adv_res_renorm["clean_top1"]),
                "top5_renorm": float(adv_res_renorm["clean_top5"]),

                # compressed robust
                "robust_top1_renorm": float(adv_res_renorm["robust_top1"]),
                "robust_top5_renorm": float(adv_res_renorm["robust_top5"]),

                # compressed robust metrics
                "top1_drop_renorm": float(adv_res_renorm["top1_accuracy_drop"]),
                "top5_drop_renorm": float(adv_res_renorm["top5_accuracy_drop"]),
                
            
                
                # null clean
                "top1_null_renorm": float(adv_res_null_renorm["clean_top1"]),
                "top5_null_renorm": float(adv_res_null_renorm["clean_top5"]),

                # null robust
                "robust_top1_null_renorm": float(adv_res_null_renorm["robust_top1"]),
                "robust_top5_null_renorm": float(adv_res_null_renorm["robust_top5"]),

                # compressed robust metrics
                "top1_drop_null_renorm": float(adv_res_null_renorm["top1_accuracy_drop"]),
                "top5_drop_null_renorm": float(adv_res_null_renorm["top5_accuracy_drop"]),
                
                })
        

        
        attack_df = pd.DataFrame(attack_rows)
        attack_df.to_csv(attack_result_path)
        

if __name__ == "__main__":
    main()