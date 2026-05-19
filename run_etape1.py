import warnings
warnings.filterwarnings("ignore")

import os
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

    cfg["task"] = "multiclass"

    set_seed(cfg["seed"])
    device = get_device()
    os.makedirs("outputs", exist_ok=True)

    print(f"Device : {device} | Task : {cfg['task']} | Seed : {cfg['seed']}")

    paths, labels = collect_data(
        cfg["dataset_root"],
        cfg["classes"],
        samples_per_class=cfg["samples_per_class"],
        seed=cfg["seed"]
    )

    print(f"Dataset : {len(paths)} images")
    for i, cls in enumerate(cfg["classes"]):
        print(f"  {cls} : {labels.count(i)}")

    folds = build_folds(paths, labels, cfg)

    for fold in folds:
        tl = fold["train_labels"]
        vl = fold["val_labels"]
        print(f"Fold {fold['fold']}/{cfg['n_folds']} — "
              f"train glioma={tl.count(0)} meningioma={tl.count(1)} notumor={tl.count(2)} pituitary={tl.count(3)} | "
              f"val glioma={vl.count(0)} meningioma={vl.count(1)} notumor={vl.count(2)} pituitary={vl.count(3)}")

    plot_fold_distribution(folds, cfg["classes"], save_path="outputs/fold_distribution_multiclass.png")