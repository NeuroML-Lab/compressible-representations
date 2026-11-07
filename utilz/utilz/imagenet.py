
from typing import Callable, Dict, Optional, Tuple
from pathlib import Path
import urllib.request
import json
from torch.utils.data import Dataset, Subset
from torchvision import datasets
from utilz.general_utils import make_dir
import torch
import numpy as np

DEFAULT_CLASS_INDEX_URLS = [
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json",
    "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
]


def load_imagenet_class_index_json(class_index_json: Optional[str], output_dir: Path) -> Dict[str, Tuple[str, str]]:
    if class_index_json is not None:
        with open(class_index_json, "r") as f:
            return json.load(f)

    cache_dir = make_dir(output_dir / "imagenet_meta")
    cache_file = cache_dir / "imagenet_class_index.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)

    last_err = None
    for url in DEFAULT_CLASS_INDEX_URLS:
        try:
            urllib.request.urlretrieve(url, cache_file)
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception as e:
            last_err = e
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except Exception:
                    pass
            continue

    raise RuntimeError(
        "Could not load imagenet_class_index.json automatically. "
        "Pass --class-index-json PATH to a local copy. "
        f"Last error: {last_err}"
    )


class ImageFolderWithRemappedTargets(Dataset):
    def __init__(self, root: str, transform: Callable, class_index_json: Dict[str, Tuple[str, str]]):
        self.base = datasets.ImageFolder(root=root, transform=transform)

        wnid_to_imagenet_idx: Dict[str, int] = {}
        for idx_str, pair in class_index_json.items():
            idx = int(idx_str)
            wnid = pair[0]
            wnid_to_imagenet_idx[wnid] = idx

        missing = [wnid for wnid in self.base.classes if wnid not in wnid_to_imagenet_idx]
        if missing:
            raise ValueError(
                "Some folder names are not valid ImageNet wnids or missing from class_index_json. "
                f"First few missing: {missing[:10]}"
            )

        self.old_to_new = {
            old_idx: wnid_to_imagenet_idx[wnid]
            for wnid, old_idx in self.base.class_to_idx.items()
        }

        self.classes = self.base.classes
        self.class_to_idx = self.base.class_to_idx
        self.targets = [self.old_to_new[t] for t in self.base.targets]
        self.samples = self.base.samples

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, old_target = self.base[idx]
        return img, self.old_to_new[old_target]


def load_imagenet_dataloader(
    data_dir: str,
    class_index_json: Dict[str, Tuple[str, str]],
    transform: Callable,
    batch_size: int,
    num_workers: int,
    subset_size: int,
    balance_classes: bool = False,
    shuffle: bool = False,
    seed: Optional[int] = 0,
) -> Dataset:
    
    dataset = ImageFolderWithRemappedTargets(root=data_dir, transform=transform, class_index_json=class_index_json)
    # indices = torch.randperm(len(dataset))[:subset_num] # random subset (not class balanced)
    targets = np.array(dataset.targets)          
    classes = np.unique(targets)
    samples_per_class = subset_size // len(classes)

    # set seed
    np.random.seed(seed)
    torch.manual_seed(seed)


    if samples_per_class > 0 and balance_classes:
        indices = np.concatenate([
            np.random.choice(np.where(targets == c)[0], samples_per_class, replace=False)
            for c in classes
        ]).tolist()
    else:
        indices = torch.randperm(len(dataset))[:subset_size].tolist() # if subset_size is smaller than number of classes, just take random subset (not class balanced)

    ds = Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader