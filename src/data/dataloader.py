import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold

from .dataset import BrainTumorDataset
from .transforms import get_transforms


def make_weighted_sampler(labels: list):
    labels_arr    = np.array(labels)
    class_counts  = np.bincount(labels_arr)
    sample_weights = (1.0 / class_counts)[labels_arr]
    return WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(sample_weights))


def build_folds(paths, labels, cfg):
    skf   = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["seed"])
    folds = []

    paths_arr  = np.array(paths)
    labels_arr = np.array(labels)

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths_arr, labels_arr)):
        train_paths  = paths_arr[train_idx].tolist()
        train_labels = labels_arr[train_idx].tolist()
        val_paths    = paths_arr[val_idx].tolist()
        val_labels   = labels_arr[val_idx].tolist()

        train_ds = BrainTumorDataset(train_paths, train_labels, get_transforms("train", cfg["image_size"]))
        val_ds   = BrainTumorDataset(val_paths,   val_labels,   get_transforms("val",   cfg["image_size"]))

        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                                  sampler=make_weighted_sampler(train_labels),
                                  num_workers=cfg["num_workers"], pin_memory = torch.cuda.is_available())
        val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"],
                                  shuffle=False,
                                  num_workers=cfg["num_workers"], pin_memory = torch.cuda.is_available())

        folds.append({
            "fold":         fold + 1,
            "train_loader": train_loader,
            "val_loader":   val_loader,
            "train_labels": train_labels,
            "val_labels":   val_labels,
        })

    return folds