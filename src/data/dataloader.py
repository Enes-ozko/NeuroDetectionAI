import torch
import numpy as np
from collections import Counter

from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold

from .dataset import BrainTumorDataset


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_stratified_folds(dataset, n_splits=5, seed=42):
    labels = np.array(dataset.labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        folds.append((train_idx.tolist(), val_idx.tolist()))

    return folds


def _build_sampler(dataset):
    labels = dataset.labels
    class_counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = torch.tensor([class_weights[l] for l in labels], dtype=torch.float)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def build_dataloaders(root, train_idx, val_idx, train_tf, val_tf, cfg, device):
    max_pc = cfg.get("max_per_class")
    pin    = device.type == "cuda"
    bs     = cfg["batch_size"]
    nw     = cfg["num_workers"]

    train_ds = BrainTumorDataset(root, indices=train_idx, transform=train_tf, max_per_class=max_pc)
    val_ds   = BrainTumorDataset(root, indices=val_idx,   transform=val_tf,   max_per_class=max_pc)

    sampler = _build_sampler(train_ds)

    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=nw, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,   num_workers=nw, pin_memory=pin)

    return train_loader, val_loader