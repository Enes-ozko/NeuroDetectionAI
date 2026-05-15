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

    # Vérification du mapping et du mélange
    print("Classes :", {i: c for i, c in enumerate(cfg["classes"])})
    print("Labels (30 premiers):", labels[:30])

    folds = build_folds(paths, labels, cfg)

    # Vérification des folds
    for fold in folds:
        tl = fold["train_labels"]
        vl = fold["val_labels"]
        print(f"\nFold {fold['fold']}")
        print(f"  Train : glioma={tl.count(0)} meningioma={tl.count(1)} notumor={tl.count(2)} pituitary={tl.count(3)}")
        print(f"  Val   : glioma={vl.count(0)} meningioma={vl.count(1)} notumor={vl.count(2)} pituitary={vl.count(3)}")
        imgs, lbls = next(iter(fold["train_loader"]))
        print(f"  Batch train : shape={list(imgs.shape)} labels={lbls[:8].tolist()}")
        imgs, lbls = next(iter(fold["val_loader"]))
        print(f"  Batch val   : shape={list(imgs.shape)} labels={lbls[:8].tolist()}")

    plot_fold_distribution(folds, cfg["classes"])