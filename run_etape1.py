import os
import sys
import random
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data.dataset import BrainTumorDataset
from data.transforms import build_transforms
from data.dataloader import get_device, get_stratified_folds, build_dataloaders


def load_config(path="config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.yaml introuvable : {os.path.abspath(path)}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    cfg    = load_config()
    set_seed(cfg["seed"])
    device = get_device()

    train_tf, val_tf = build_transforms(cfg)

    full_ds = BrainTumorDataset(
        root=cfg["data_root"],
        max_per_class=cfg.get("max_per_class"),
        seed=cfg["seed"],
    )

    folds = get_stratified_folds(full_ds, cfg["n_splits"], cfg["seed"])

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"fold {fold_idx + 1}/{cfg['n_splits']}")

        train_loader, val_loader = build_dataloaders(
            root=cfg["data_root"],
            train_idx=train_idx,
            val_idx=val_idx,
            train_tf=train_tf,
            val_tf=val_tf,
            cfg=cfg,
            device=device,
        )

        imgs, labels = next(iter(train_loader))
        imgs   = imgs.to(device)
        labels = labels.to(device)

        print(f"  shape  : {list(imgs.shape)}")
        print(f"  labels : {labels[:8].tolist()}")
        print(f"  range  : [{imgs.min():.3f}, {imgs.max():.3f}]")
        break


if __name__ == "__main__":
    main()