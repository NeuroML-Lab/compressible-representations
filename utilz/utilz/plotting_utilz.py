import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


legal_metrics = ['clean_acc', 'robust_top1', 'robust_top5', 'rsa', 'cka',
                      'top1', 'top5', 'delta_top1', 'delta_top5',
                      'norm_rep_distance']



def extract_layer_rank_grid(df, renorm=False, null_proj=False, metric='clean_acc', avg_repeats=False):
    assert renorm in [True, False, None], "renorm should be a boolean or None"
    assert null_proj in [True, False], "null_proj should be a boolean"
    assert metric in legal_metrics, f"metric should be one of {legal_metrics}"  

    return_col = ''
    metric_str = f'{metric}'
    proj_str = 'null_' if null_proj else ''
    
    if renorm is None:
        rep_str = ''
    else:   
        rep_str = 'renorm' if renorm else 'raw'

    delta_str = '_vs_baseline' if 'delta' in metric else ''

    return_col = f'{metric_str}_{proj_str}{rep_str}{delta_str}'
    if return_col not in df.columns:
        return_col = f'{proj_str}{metric_str}_{rep_str}'  #

    # print(return_col) # remove after debugging

    layer_ind_col = 'layer_idx' if 'layer_idx' in df.columns else 'layer_index'
    cols_keep = [layer_ind_col, 'retain_fraction_layer', return_col]
    lr_df = df[cols_keep].copy()
    if avg_repeats:
        lr_df = lr_df.groupby([layer_ind_col, 'retain_fraction_layer'], as_index=False).mean()
    lr_df = lr_df.pivot(index=layer_ind_col, columns='retain_fraction_layer', values=return_col)
    lr_grid = lr_df.values

    return lr_grid


def extract_layer_inds_from_data(df):
    # check if col name is layer_idx or layer_index
    if 'layer_idx' in df.columns:
        _, layer_order_inds = np.unique(df.layer_idx, return_index=True)
        layer_inds = df.layer_idx.values[layer_order_inds]
    else:
        _, layer_order_inds = np.unique(df.layer_index, return_index=True)
        layer_inds = df.layer_index.values[layer_order_inds]
    
    return layer_inds

def extract_rank_fracs_from_data(df, global_flag=False):
    if global_flag:
        _, rank_order_inds = np.unique(df.retain_fraction_global, return_index=True)
        rank_fracs = df.retain_fraction_global.values[rank_order_inds].round(2)
    else:
        _, rank_order_inds = np.unique(df.retain_fraction_layer, return_index=True)
        rank_fracs = df.retain_fraction_layer.values[rank_order_inds].round(2)
    return rank_fracs

proj_to_title = {
    "task": "Task sensitivity",
    "pca": "PCA",
    "local": "Local",
    "gt": "Geotopology",
    'task_sensitive': "Task sensitivity",
    "local_geometry": "Local",
    "geotopology": "Geotopology",
    "random": "Random"
}

proj_to_shorthand = {
    "task_sensitive": "task",
    "local_geometry": "local",
    "geotopology": "gt",
    "pca": "pca",
    "random": "random"
}

proj_to_filename = {
    "task": "task_sensitive",
    "local": "local_geometry",
    "gt": "geotopology",
    "pca": "pca",
    "random": "random"
}

model_to_title = {
    "vit_b_16": "ViT-B/16",
    "vit_b_32": "ViT-B/32",
    "vit_l_16": "ViT-L/16",
    "vit_l_32": "ViT-L/32",
    "deit3_b_16": "DeiT3-B/16",
    "vit_b_16_in21k": "ViT-B/16 (IN21K)",
    "vit_s_16": "ViT-S/16",
    "vit_ti_16": "ViT-Ti/16"
}

metric_to_title = {
    "clean_acc": "Clean Accuracy",
    "robust_top1": "Robust Top-1 Accuracy",
    "robust_top5": "Robust Top-5 Accuracy",
    "delta_top1": "Change in Top-1 Accuracy",
    "delta_top5": "Change in Top-5 Accuracy",
    "top1": "Top-1 Accuracy",
    "top5": "Top-5 Accuracy",
    "rsa": "RSA",
    "cka": "CKA"
}

proj_types = ['task', 'pca', 'local', 'gt']
proj_types_with_random = proj_types + ['random']
proj_colors = sns.color_palette("husl", len(proj_types))

