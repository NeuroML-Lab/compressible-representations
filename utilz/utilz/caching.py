import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from utilz.general_utils import load_json, safe_json

def build_cache_config(args, layer_indices: List[int], hidden_dim: int, fraction_grid: List[float], compression_type: str, multi_model_index: int = None) -> dict:
    if multi_model_index is not None:
        model_name = args.model1_name if multi_model_index == 1 else args.model2_name
    else:
        model_name = args.model_name


    config =  {
        "model_name": model_name,
        "train_dir": str(args.train_dir),
        "val_dir": str(args.val_dir),
        "layer_indices": list(layer_indices),
        "hidden_dim": int(hidden_dim),
        "fraction_grid": list(fraction_grid),
        "calibration_samples": int(args.calibration_samples),
        "eval_samples": int(args.eval_samples),
        "seed": int(args.seed),
        "max_calibration_batches": None if args.max_calibration_batches is None else int(args.max_calibration_batches),
    }
    # print(f"Loading cache with config: {config}")
    if compression_type == 'local_geometry':
        config["k_neighbors"] = int(args.k_neighbors)
        config["neighbor_metric"] = str(args.neighbor_metric)
        config["weight_mode"] = str(args.weight_mode)
        config["sigma"] = float(args.sigma) if args.sigma is not None else None

    if compression_type == 'geotopology':
        config["distance_metric"] = str(args.distance_metric)
        config["gt_lower_quantile"] = float(args.gt_lower_quantile)
        config["gt_upper_quantile"] = float(args.gt_upper_quantile)
        config["gt_weight_mode"] = str(args.gt_weight_mode)
    return config

def cache_matches(expected: dict, observed: dict) -> bool:
    # compare all keys except train_dir and val_dir
    ignore = {"train_dir", "val_dir", "eval_samples", "seed", "calibration_samples"}
    return {k: v for k, v in expected.items() if k not in ignore} == \
           {k: v for k, v in observed.items() if k not in ignore}


def get_cache_paths(cache_dir: Path, compression_type: str) -> dict:
    dict_to_return =  {
        "meta": cache_dir / "utility_cache_meta.json",
        "utility": cache_dir / f"single_layer_{compression_type}_utility_curves.csv",
        "spectra": cache_dir / f"layer_{compression_type}_spectra.csv",
        "eigenspaces": cache_dir / f"{compression_type}_eigenspaces.pt"}
    return dict_to_return


def maybe_load_cached(
    cache_dir: Path,
    expected_cache_config: dict,
    compression_type: str,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]]:
    paths = get_cache_paths(cache_dir, compression_type)
    if not all(p.exists() for p in paths.values()):
        print("Cache config mismatch!")
        missing_files = [str(p) for p in paths.values() if not p.exists()]
        print(f"Missing files: {missing_files}")
        return None
    observed = load_json(paths["meta"])
    if not cache_matches(expected_cache_config, observed):
        print("Cache config mismatch!")
        print(f"Expected cache config: {expected_cache_config}\n\n")
        print(f"Observed cache config: {observed}")
        mismatch_keys = [k for k in expected_cache_config if expected_cache_config[k] != observed.get(k)]
        print(f"Mismatched keys: {mismatch_keys}")
        return None

    utility_df = pd.read_csv(paths["utility"])
    spectra_df = pd.read_csv(paths["spectra"])
    saved = torch.load(paths["eigenspaces"], map_location="cpu", weights_only=False)
    eigvals_by_layer = {int(k): v for k, v in saved["eigvals_by_layer"].items()}
    eigvecs_by_layer = {int(k): v for k, v in saved["eigvecs_by_layer"].items()}
    return utility_df, spectra_df, eigvals_by_layer, eigvecs_by_layer


def save_compression_cache(
    cache_dir: Path,
    cache_config: dict,
    utility_df: pd.DataFrame,
    spectra_df: pd.DataFrame,
    compression_type: str,
    eigvals_by_layer: Dict[int, torch.Tensor],
    eigvecs_by_layer: Dict[int, torch.Tensor],
) -> None:
    paths = get_cache_paths(cache_dir, compression_type=compression_type)
    safe_json(cache_config, paths["meta"])
    utility_df.to_csv(paths["utility"], index=False)
    spectra_df.to_csv(paths["spectra"], index=False)
    torch.save(
        {
            "eigvals_by_layer": {int(k): v.cpu() for k, v in eigvals_by_layer.items()},
            "eigvecs_by_layer": {int(k): v.cpu() for k, v in eigvecs_by_layer.items()},
        },
        paths["eigenspaces"])