import warnings
warnings.filterwarnings("ignore")

import random
import yaml
import numpy as np
import torch

from src.data import collect_data, build_folds, plot_fold_distribution


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device()

    paths, labels = collect_data(
        cfg["dataset_root"],
        cfg["classes"],
        samples_per_class=cfg["samples_per_class"],
        seed=cfg["seed"]
    )

    folds = build_folds(paths, labels, cfg)
    plot_fold_distribution(folds, cfg["classes"])

    print("Classes :", {i: c for i, c in enumerate(cfg["classes"])})
    for fold in folds:
        imgs, lbls = next(iter(fold["train_loader"]))
        print(f"Fold {fold['fold']} | train={len(fold['train_labels'])} val={len(fold['val_labels'])} | batch={list(imgs.shape)} | {lbls[:8].tolist()}")