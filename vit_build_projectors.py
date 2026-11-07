# main driver function that builds projectors and evaluates them for layerwise and global compression

import torch
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List
from utilz.general_utils import get_device, choose_indices, metric_key, set_seed, make_dir
from utilz.model_zoo import load_model, MODEL_REGISTRY
from utilz.model_evals import evaluate_multi_layer_projection, evaluate_single_layer_projection
from utilz.parsing import parse_fraction_grid, parse_layer_subset
from utilz.imagenet import load_imagenet_class_index_json, load_imagenet_dataloader
from utilz.act_compression import build_candidate_ranks, build_projector_banks_from_eigenspaces, estimate_projector, nearest_int_budget
from utilz.caching import build_cache_config, maybe_load_cached, save_compression_cache
from utilz.rank_allocation import solve_allocation_dp


parser = argparse.ArgumentParser()
parser.add_argument("--compression-type", type=str, choices=["task_sensitive", "pca", "local_geometry", "geotopology", "random"])
parser.add_argument("--output-dir", type=str, default="")
parser.add_argument("--utility-cache-dir", type=str, default=None)

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device-id", type=str, default="auto")
parser.add_argument("--model-name", type=str, default="vit_b_16", choices=list(MODEL_REGISTRY.keys()))
parser.add_argument('--layerwise', dest='layerwise', action='store_true')
parser.add_argument('--no-layerwise', dest='layerwise', action='store_false')
parser.set_defaults(layerwise=True)

parser.add_argument("--train-dir", type=str, required=True)
parser.add_argument("--val-dir", type=str, required=True)
parser.add_argument("--calibration-samples", type=int, default=512)
parser.add_argument("--eval-samples", type=int, default=2000)
parser.add_argument("--class-index-json", type=str, default=None)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--max-calibration-batches", type=int, default=None) 
parser.add_argument("--pin-memory", action="store_true")

parser.add_argument("--fraction-grid", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")

parser.add_argument("--retain-fraction", type=float, default=0.5) # global
parser.add_argument("--allocation-mode", type=str, default="utility_curve", choices=["utility_curve", "fixed"])
parser.add_argument("--utility-condition", type=str, default="raw", choices=["raw", "renorm"])
parser.add_argument("--utility-metric", type=str, default="top1", choices=["top1", "top5"])
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
    "random": "random",
}


def main():
    # Set up
    set_seed(args.seed)
    device = get_device(args.device_id)
    outdir = Path(f'{args.output_dir}/{args.model_name}/')
    if not outdir.exists():
        outdir.mkdir(parents=True)
    # cache_dir = make_dir(args.utility_cache_dir) if args.utility_cache_dir is not None else outdir
    comp = compression_shorthands[args.compression_type]
    cache_path = f"{outdir}/{comp}_cache_seed{args.seed}/" if  comp == "random" else f"{outdir}/{comp}_cache/"
    cache_dir = make_dir(cache_path)

    # load model
    loaded = load_model(args.model_name, device)
    
    
    model = loaded.model
    preprocess = loaded.preprocess
    layer_indices = parse_layer_subset('all', loaded.num_layers)

    # load params
    fraction_grid = parse_fraction_grid(args.fraction_grid)
    candidate_ranks = build_candidate_ranks(loaded.hidden_dim, fraction_grid)
    utility_key = metric_key(args.utility_metric)
    print(f"===Evaluating compression type {args.compression_type} with model {args.model_name} on device {device}===")
    # print(f"Candidate retained fractions: {fraction_grid}")
    # print(f"Candidate ranks: {candidate_ranks}")


    # load data
    class_index_json = load_imagenet_class_index_json(args.class_index_json, outdir)
    calib_loader = load_imagenet_dataloader(
        data_dir=args.train_dir,
        class_index_json=class_index_json,
        transform=preprocess,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_size=args.calibration_samples,
        balance_classes=True,
        shuffle=True,
        seed=0,
    )

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

    

    cache_config = build_cache_config(args, layer_indices, loaded.hidden_dim, fraction_grid, args.compression_type)
    print(f"Building Projectors for Cache config: {cache_config}")

    baseline = evaluate_single_layer_projection(model=model, loader=eval_loader, device=device, layer_idx=layer_indices[0], projector=None, renorm=False)
    print(f"Baseline | top1={baseline['top1']:.4f} | top5={baseline['top5']:.4f}")

    cached =  maybe_load_cached(cache_dir, cache_config, args.compression_type)
    print(f"Cache hit: {cached is not None}" )
    if cached is None:
        if args.layerwise:
            # ==== build layerwise projectors ====
            eigvals_by_layer: Dict[int, torch.Tensor] = {}
            eigvecs_by_layer: Dict[int, torch.Tensor] = {}
            projectors_by_layer: Dict[int, Dict[int, torch.Tensor]] = {}
            null_projectors_by_layer: Dict[int, Dict[int, torch.Tensor]] = {}
            P_stats_by_layer: Dict[int, Dict[str, float]] = {}

            curves_rows: List[Dict[str, float | int | str]] = []
            spectra_rows: List[Dict[str, float | int]] = []

            for layer_idx in layer_indices:
                print(f"Estimating Projector for layer {layer_idx} / {loaded.num_layers}")

                projector_res  = estimate_projector(args.compression_type, model, calib_loader, layer_idx, device, candidate_ranks, 
                                                    loaded.hidden_dim,args.max_calibration_batches, args=args, rand_seed=args.seed)
                
                projectors_by_layer[layer_idx] = projector_res["projectors"]
                null_projectors_by_layer[layer_idx] = projector_res["null_projectors"]
                eigvals_by_layer[layer_idx] = projector_res["eigvals"]
                eigvecs_by_layer[layer_idx] = projector_res["eigvecs"]
                P_stats_by_layer[layer_idx] = projector_res["P_stats"]

                # save spectra rows
                total_mass = float(eigvals_by_layer[layer_idx].sum().item())
                for i, lam in enumerate(eigvals_by_layer[layer_idx].tolist()):
                    spectra_rows.append({
                        "layer_index": layer_idx,
                        "eig_rank": i + 1,
                        "eigenvalue": float(lam),
                        "normalized_eigenvalue": float(lam / total_mass) if total_mass > 1e-12 else 0.0,
                        "cumulative_mass_fraction": float(eigvals_by_layer[layer_idx][: i + 1].sum().item() / max(total_mass, 1e-12)),
                    })

        
                # ==== evaluate layerwise projectors + utility curves ====
                print(f"Evaluating single-layer utility curve for layer {layer_idx}")
                for rank in candidate_ranks:
                    Pi = projectors_by_layer[layer_idx][rank]
                    Pi_null = null_projectors_by_layer[layer_idx][rank]

                    m_raw = evaluate_single_layer_projection(model=model, loader=eval_loader, device=device, layer_idx=layer_idx, projector=Pi, renorm=False)
                    m_renorm = evaluate_single_layer_projection(model=model, loader=eval_loader, device=device, layer_idx=layer_idx, projector=Pi, renorm=True)
                    m_null_raw = evaluate_single_layer_projection(model=model, loader=eval_loader, device=device, layer_idx=layer_idx, projector=Pi_null, renorm=False)
                    m_null_renorm = evaluate_single_layer_projection(model=model, loader=eval_loader, device=device, layer_idx=layer_idx, projector=Pi_null, renorm=True)

                    curves_rows.append({
                        "layer_index": layer_idx,
                        "hidden_dim": loaded.hidden_dim,
                        "rank": int(rank),
                        "retain_fraction_layer": float(rank / loaded.hidden_dim),
                        "top1_raw": float(m_raw["top1"]),
                        "top5_raw": float(m_raw["top5"]),
                        "top1_renorm": float(m_renorm["top1"]),
                        "top5_renorm": float(m_renorm["top5"]),
                        "top1_null_raw": float(m_null_raw["top1"]),
                        "top5_null_raw": float(m_null_raw["top5"]),
                        "top1_null_renorm": float(m_null_renorm["top1"]),
                        "top5_null_renorm": float(m_null_renorm["top5"]),
                        "delta_top1_raw_vs_baseline": float(m_raw["top1"] - baseline["top1"]),
                        "delta_top5_raw_vs_baseline": float(m_raw["top5"] - baseline["top5"]),
                        "delta_top1_renorm_vs_baseline": float(m_renorm["top1"] - baseline["top1"]),
                        "delta_top5_renorm_vs_baseline": float(m_renorm["top5"] - baseline["top5"]),
                        "delta_top1_null_raw_vs_baseline": float(m_null_raw["top1"] - baseline["top1"]),
                        "delta_top5_null_raw_vs_baseline": float(m_null_raw["top5"] - baseline["top5"]),
                        "delta_top1_null_renorm_vs_baseline": float(m_null_renorm["top1"] - baseline["top1"]),
                        "delta_top5_null_renorm_vs_baseline": float(m_null_renorm["top5"] - baseline["top5"]),
                        **{f"Proj_{k}": v for k, v in P_stats_by_layer[layer_idx].items()},
                    })
                    print(f"rank={rank:>3d} frac={rank/loaded.hidden_dim:.3f} | task_raw top1={m_raw['top1']:.4f} | task_renorm top1={m_renorm['top1']:.4f}")
                    
            utility_df = pd.DataFrame(curves_rows)
            spectra_df = pd.DataFrame(spectra_rows)

            save_compression_cache(
                cache_dir=cache_dir,
                cache_config=cache_config,
                utility_df=utility_df,
                spectra_df=spectra_df,
                compression_type=args.compression_type,
                eigvals_by_layer=eigvals_by_layer,
                eigvecs_by_layer=eigvecs_by_layer,
            )
            
            print(f"Saved compression cache to {cache_dir} with utility, spectra and eigenspaces.")
        else:
            raise NotImplementedError("Multi-layer projections only work with cached results for now.")   
   
   
   
    # ==== build multi-layer projectors with fixed allocation ====
    if not args.layerwise and cached:
        utility_df, spectra_df, eigvals_by_layer, eigvecs_by_layer = cached
        print(f"Loaded utility curves, spectra and eigenspaces from cache dir {cache_dir}. Building multi-layer projectors and evaluating...")
        
        total_dims = loaded.hidden_dim * len(layer_indices)
        alloc_path = f"{outdir}/allocation/"
        if not Path(alloc_path).exists():
            make_dir(alloc_path)

        projectors_by_layer, null_projectors_by_layer = build_projector_banks_from_eigenspaces(
            eigvecs_by_layer=eigvecs_by_layer,
            candidate_ranks=candidate_ranks,
            hidden_dim=loaded.hidden_dim,
        )
        final_rows: List[Dict[str, float | str]] = [{"condition": "baseline", **baseline}]

        for r_fraction in fraction_grid:
            print(f"Allocating ranks for global retain fraction {r_fraction} with allocation mode {args.allocation_mode}")

            # allocate fixed, uniform rank to each layer
            if args.allocation_mode == "fixed":
                allocation_rows = []
            
                for layer_idx in layer_indices:
                    k_l = max(1, round(loaded.hidden_dim * r_fraction))
                    retain = k_l / loaded.hidden_dim
                    allocation_rows.append({
                        "layer_index": layer_idx,
                        "hidden_dim": loaded.hidden_dim,
                        "k_l": k_l,
                        "retain_fraction_layer": retain,
                        "allocation_mode": args.allocation_mode,
                        "utility_score_used_for_allocation": float("nan"),
                    })

            # allocate with DP utility curve optimization
            elif args.allocation_mode == "utility_curve":
                if utility_df.empty:
                    raise RuntimeError("utility_curve allocation requested, but cached utility curves are empty.")
                layer_candidates: Dict[int, List[Dict[str, float]]] = {}
                for layer_idx in layer_indices:
                    sub = utility_df[utility_df["layer_index"] == layer_idx].sort_values("rank")
                    cand_list = []
                    for _, row in sub.iterrows():
                        util = float(row[f"{utility_key}_{args.utility_condition}"])
                        cand_list.append({
                            "rank": int(row["rank"]),
                            "utility": util,
                            "retain_fraction_layer": float(row["retain_fraction_layer"]),
                        })
                    layer_candidates[layer_idx] = cand_list
                min_required = sum(min(int(c["rank"]) for c in layer_candidates[l]) for l in layer_indices)
                budget_dims = nearest_int_budget(total_dims, r_fraction)
                if budget_dims < min_required:
                    print(f"Mismatch in budget and min dims required: {(budget_dims, min_required)} for layer {layer_idx} r_frac {r_fraction}")
                budget_dims = max(budget_dims, min_required)

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
            seed_str = f"_seed{args.seed}" if comp == "random"  else ''
            allocation_df.to_csv(f"{alloc_path}/{comp}_{args.allocation_mode}_{r_fraction}{seed_str}.csv")

            projectors: Dict[int, torch.Tensor] = {}
            null_projectors: Dict[int, torch.Tensor] = {}
            for _, row in allocation_df.iterrows():
                layer_idx = int(row["layer_index"])
                k = int(row["k_l"])
                projectors[layer_idx] = projectors_by_layer[layer_idx][k]
                null_projectors[layer_idx] = null_projectors_by_layer[layer_idx][k]

            print("Evaluating baseline / allocated task-pruned / allocated null-pruned conditions...")
            

            for cond_name, projectors, renorm in [
                ("compressed_raw", projectors, False),
                ("null_raw", null_projectors, False),
                ("compressed_renorm", projectors, True),
                ("null_renorm", null_projectors, True),
            ]:
                metrics = evaluate_multi_layer_projection(model=model, loader=eval_loader, device=device, projectors=projectors, renorm=renorm)
                final_rows.append({
                    "retain_fraction_global": r_fraction,
                    "condition": cond_name,
                    **metrics,
                    "delta_top1_vs_baseline": float(metrics["top1"] - baseline["top1"]),
                    "delta_top5_vs_baseline": float(metrics["top5"] - baseline["top5"]),
                })
                print(f"{cond_name:>16s} | top1={metrics['top1']:.4f} | top5={metrics['top5']:.4f}")

        final_df = pd.DataFrame(final_rows)
        final_results_path = f"{outdir}/{compression_shorthands[args.compression_type]}_global_results{seed_str}/"
        if not Path(final_results_path).exists():
            make_dir(final_results_path)
        final_df.to_csv(f"{final_results_path}/{args.allocation_mode}.csv", index=False)







if __name__ == "__main__":
    main()