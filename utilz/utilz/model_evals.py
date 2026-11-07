import torch
from typing import Dict, Optional, Tuple
from utilz.act_compression import SingleCLSProjectionHook, MultiCLSProjectionHook
from torch import nn
from torch.utils.data import DataLoader


def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> Dict[int, float]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    out: Dict[int, float] = {}
    for k in topk:
        out[k] = float(correct[:k].any(dim=0).float().sum().item())
    return out



def evaluate_single_layer_projection(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    layer_idx: int,
    projector: Optional[torch.Tensor],
    renorm: bool,
) -> Dict[str, float]:
    model.eval()
    hook = None
    
    # registers projector hook on layer of choice
    if projector is not None:
        hook = model.encoder.layers[layer_idx].register_forward_hook(SingleCLSProjectionHook(projector.to(device), renorm=renorm))

    top1 = 0.0
    top5 = 0.0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            accs = accuracy_topk(logits, labels, topk=(1, 5))
            bs = labels.shape[0]
            top1 += accs[1]
            top5 += accs[5]
            total += bs

    if hook is not None:
        hook.remove()

    return {"top1": top1 / max(total, 1), "top5": top5 / max(total, 1), "n_eval": float(total)}



def evaluate_multi_layer_projection(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    projectors: Dict[int, torch.Tensor],
    renorm: bool,
) -> Dict[str, float]:
    model.eval()
    handles = []

    # create hook factory for multiple layer projectors and register
    hook_factory = MultiCLSProjectionHook({k: v.to(device) for k, v in projectors.items()}, renorm=renorm)
    for layer_idx in sorted(projectors):
        handles.append(model.encoder.layers[layer_idx].register_forward_hook(hook_factory.make_hook(layer_idx)))

    top1 = 0.0
    top5 = 0.0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            accs = accuracy_topk(logits, labels, topk=(1, 5))
            bs = labels.shape[0]
            top1 += accs[1]
            top5 += accs[5]
            total += bs

    for h in handles:
        h.remove()

    return {"top1": top1 / max(total, 1), "top5": top5 / max(total, 1), "n_eval": float(total)}