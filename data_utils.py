import os
import random
from typing import List, Tuple

import h5py
import torch
import torch.nn.functional as F


SEED = 42


def _find_h5_files(root: str) -> List[str]:
    root = os.path.abspath(os.path.expanduser(root))
    print(f"[DEBUG] Searching for H5 files under: {root}")

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Directory not found: {root}")

    candidates = []
    for sub in [root, os.path.join(root, "patches")]:
        if os.path.isdir(sub):
            for fn in os.listdir(sub):
                if fn.lower().endswith(".h5") or fn.lower().endswith(".hdf5"):
                    candidates.append(os.path.join(sub, fn))

    candidates = sorted(set(candidates))

    if not candidates:
        raise FileNotFoundError(
            f"No .h5 files found under:\n  {root}\n  {os.path.join(root, 'patches')}"
        )

    print(f"[DEBUG] Found {len(candidates)} H5 files")
    print(f"[DEBUG] First file: {candidates[0]}")
    return candidates


def _read_embeddings_from_h5(path: str, feature_name: str) -> torch.Tensor:
    with h5py.File(path, "r") as f:
        arr = f["features"][feature_name][:]
    return torch.tensor(arr, dtype=torch.float32)


def _infer_label_from_filename(path: str) -> int:
    base = os.path.basename(path).lower()
    if "tumor" in base:
        return 1
    if "normal" in base:
        return 0
    raise ValueError(
        f"Cannot infer label from filename: {base}. "
        f"Expected 'tumor' or 'normal' in filename."
    )


def _count_labels(files: List[str]) -> Tuple[int, int]:
    normal = 0
    tumor = 0
    for path in files:
        y = _infer_label_from_filename(path)
        if y == 1:
            tumor += 1
        else:
            normal += 1
    return normal, tumor

# Created a stratified train/validation split such that both classes (tumor and normal)
#  appear in both splits while keeping roughly the same class proportions.
def _stratified_split(
    all_files: List[str],
    train_ratio: float = 0.9,
    seed: int = SEED
) -> Tuple[List[str], List[str]]:
    tumor_files = [p for p in all_files if _infer_label_from_filename(p) == 1]
    normal_files = [p for p in all_files if _infer_label_from_filename(p) == 0]

    # Ensure both classes exist so stratified splitting is possible
    if len(tumor_files) == 0 or len(normal_files) == 0:
        raise ValueError(
            "Stratified split requires at least one tumor file and one normal file."
        )

    rng = random.Random(seed)

    # Shuffle each class separately before splitting
    rng.shuffle(tumor_files)
    rng.shuffle(normal_files)

    # Helper function to split a single class into train/validation subsets
    def split_class(files: List[str]) -> Tuple[List[str], List[str]]:
        if len(files) == 1:
            raise ValueError(
                "A class has only 1 file; cannot create both train and val splits."
            )

        # Compute the training split size
        train_len = int(train_ratio * len(files))

        # Ensure at least one file in train and one in validation
        train_len = max(1, train_len)
        train_len = min(train_len, len(files) - 1)

        # Perform the split
        train_part = files[:train_len]
        val_part = files[train_len:]

        if len(val_part) == 0:
            train_part = files[:-1]
            val_part = files[-1:]

        return train_part, val_part

    # Perform stratified split for tumor and normal classes
    tumor_train, tumor_val = split_class(tumor_files)
    normal_train, normal_val = split_class(normal_files)

    # Combine the class-specific splits
    train_files = tumor_train + normal_train
    val_files = tumor_val + normal_val

    # Shuffle final splits to mix class ordering
    rng.shuffle(train_files)
    rng.shuffle(val_files)

    # Return training and validation file lists
    return train_files, val_files


def instance_dropout(bag: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    if bag.dim() != 2:
        raise ValueError(f"Expected bag shape (N, D). Got {tuple(bag.shape)}")

    n = bag.size(0)
    if n <= 1:
        return bag

    keep = (torch.rand(n) > p)
    if keep.sum().item() == 0:
        keep[random.randrange(n)] = True

    return bag[keep]


def embedding_noise(bag: torch.Tensor, w_max: float = 0.05) -> torch.Tensor:
    if bag.dim() != 2:
        raise ValueError(f"Expected bag shape (N, D). Got {tuple(bag.shape)}")
    w = torch.rand(1).item() * w_max
    return bag + torch.randn_like(bag) * w


def normalize_bag(bag: torch.Tensor) -> torch.Tensor:
    bag = torch.nan_to_num(bag, nan=0.0, posinf=1e6, neginf=-1e6)
    bag = F.normalize(bag, p=2, dim=1)
    return bag


def collate_keep_list(batch):
    bags, ys = zip(*batch)
    return list(bags), torch.stack(list(ys), dim=0)