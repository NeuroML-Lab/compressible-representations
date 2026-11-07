import torch
from typing import Dict, List, Tuple




def solve_allocation_dp(
    layer_candidates: Dict[int, List[Dict[str, float]]],
    budget_dims: int,
    utility_metric: str,
) -> Dict[int, Dict[str, float]]:
    layers = sorted(layer_candidates)
    dp: Dict[int, Dict[int, float]] = {0: {0: 0.0}}
    back: Dict[Tuple[int, int], Tuple[int, int, int]] = {}

    for i, layer in enumerate(layers, start=1):
        dp[i] = {}
        cand_list = layer_candidates[layer]
        for prev_b, prev_val in dp[i - 1].items():
            for j, cand in enumerate(cand_list):
                b = prev_b + int(cand["rank"])
                if b > budget_dims:
                    continue
                val = prev_val + float(cand[utility_metric])
                if (b not in dp[i]) or (val > dp[i][b]):
                    dp[i][b] = val
                    back[(i, b)] = (prev_b, layer, j)

    if not dp[len(layers)]:
        raise RuntimeError(
            f"No feasible allocation found under budget {budget_dims}. "
            f"Minimum required budget is at least {sum(min(int(c['rank']) for c in layer_candidates[l]) for l in layers)}."
        )

    final_b, _ = max(dp[len(layers)].items(), key=lambda kv: kv[1])
    chosen: Dict[int, Dict[str, float]] = {}
    cur_b = final_b
    for i in range(len(layers), 0, -1):
        prev_b, layer, j = back[(i, cur_b)]
        chosen[layer] = layer_candidates[layer][j]
        cur_b = prev_b
    return chosen


def allocate_by_normalized_spectral_mass(
    layer_eigvals: Dict[int, torch.Tensor],
    budget_dims: int,
) -> Dict[int, int]:
    entries: List[Tuple[float, int, int]] = []
    for layer_idx, eigvals in layer_eigvals.items():
        total = float(eigvals.sum().item())
        denom = total if total > 1e-12 else 1.0
        normed = (eigvals / denom).tolist()
        for i, score in enumerate(normed, start=1):
            entries.append((float(score), int(layer_idx), int(i)))

    entries.sort(key=lambda x: x[0], reverse=True)
    keep_counts = {layer_idx: 0 for layer_idx in layer_eigvals.keys()}
    for _, layer_idx, _ in entries[:budget_dims]:
        keep_counts[layer_idx] += 1

    for layer_idx, eigvals in layer_eigvals.items():
        d = eigvals.numel()
        keep_counts[layer_idx] = max(1, min(d - 1, keep_counts[layer_idx]))

    current = sum(keep_counts.values())
    if current > budget_dims:
        removable: List[Tuple[float, int]] = []
        for layer_idx, eigvals in layer_eigvals.items():
            total = float(eigvals.sum().item())
            denom = total if total > 1e-12 else 1.0
            normed = (eigvals / denom).tolist()
            k = keep_counts[layer_idx]
            for i in range(k, 0, -1):
                removable.append((float(normed[i - 1]), int(layer_idx)))
        removable.sort(key=lambda x: x[0])
        idx = 0
        while current > budget_dims and idx < len(removable):
            _, layer_idx = removable[idx]
            if keep_counts[layer_idx] > 1:
                keep_counts[layer_idx] -= 1
                current -= 1
            idx += 1

    current = sum(keep_counts.values())
    if current < budget_dims:
        addable: List[Tuple[float, int]] = []
        for layer_idx, eigvals in layer_eigvals.items():
            total = float(eigvals.sum().item())
            denom = total if total > 1e-12 else 1.0
            normed = (eigvals / denom).tolist()
            k = keep_counts[layer_idx]
            for i in range(k + 1, len(normed) + 1):
                addable.append((float(normed[i - 1]), int(layer_idx)))
        addable.sort(key=lambda x: x[0], reverse=True)
        idx = 0
        while current < budget_dims and idx < len(addable):
            _, layer_idx = addable[idx]
            d = layer_eigvals[layer_idx].numel()
            if keep_counts[layer_idx] < d - 1:
                keep_counts[layer_idx] += 1
                current += 1
            idx += 1

    return keep_counts