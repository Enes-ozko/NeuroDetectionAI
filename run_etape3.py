import warnings
warnings.filterwarnings("ignore")

import os
import random
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset    import collect_data, BrainTumorDataset
from src.data.dataloader import build_folds
from src.data.transforms import get_transforms
from src.etape3.model    import build_model
from src.etape3.train    import train_etape3
from src.etape3.evaluate import evaluate_etape3


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


LABEL_REMAP = {0: 0, 1: 1, 3: 2}


def filter_and_remap(paths, labels):
    filtered_paths  = []
    filtered_labels = []
    for p, l in zip(paths, labels):
        if l == 2:
            continue
        filtered_paths.append(p)
        filtered_labels.append(LABEL_REMAP[l])
    return filtered_paths, filtered_labels


if __name__ == "__main__":

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device : {device}")

    CLASSES_4 = ["glioma", "meningioma", "notumor", "pituitary"]

    all_train_paths, all_train_labels = collect_data(
        root=cfg["train_root"],
        classes=CLASSES_4,
        samples_per_class=cfg.get("samples_per_class", None),
        seed=cfg["seed"],
    )
    train_paths, train_labels = filter_and_remap(all_train_paths, all_train_labels)

    print(f"Training filtre : {len(train_paths)} images")
    for name, idx in [("glioma", 0), ("meningioma", 1), ("pituitary", 2)]:
        print(f"  {name:<12} : {train_labels.count(idx)}")

    all_test_paths, all_test_labels = collect_data(
        root=cfg["test_root"],
        classes=CLASSES_4,
        samples_per_class=None,
        seed=cfg["seed"],
    )
    test_paths, test_labels = filter_and_remap(all_test_paths, all_test_labels)

    print(f"Testing filtre  : {len(test_paths)} images")
    for name, idx in [("glioma", 0), ("meningioma", 1), ("pituitary", 2)]:
        print(f"  {name:<12} : {test_labels.count(idx)}")

    cfg_etape3 = {**cfg, "task": "multiclass"}

    print(f"\nConstruction des {cfg['n_folds']} folds stratifies...")
    folds = build_folds(train_paths, train_labels, cfg_etape3)

    print("\nEntrainement Etape 3")
    results = train_etape3(
        get_model_fn=lambda: build_model(
            num_classes=3,
            dropout=cfg.get("dropout_p_e3", cfg.get("dropout_p", 0.3)),
        ),
        folds=folds,
        cfg=cfg_etape3,
        device=device,
    )

    best = max(results, key=lambda r: r["best_val_acc"])
    print(f"\nMeilleur fold : Fold {best['fold']} (val_acc={best['best_val_acc']:.3f})")

    torch.save(best["model"].state_dict(), "outputs/model_etape3.pth")
    print("Modele sauvegarde -> outputs/model_etape3.pth")

    evaluate_etape3(
        model=best["model"],
        val_loader=best["val_loader"],
        device=device,
        save_path="outputs/etape3_evaluation_val.png",
    )

    test_ds = BrainTumorDataset(
        test_paths, test_labels,
        get_transforms("val", cfg["image_size"]),
        task="multiclass",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    metrics = evaluate_etape3(
        model=best["model"],
        val_loader=test_loader,
        device=device,
        save_path="outputs/etape3_evaluation_test.png",
    )

    print(f"\nVal acc (test) : {metrics['val_acc']:.3f}")
    print(f"AUC moy (test) : {metrics['mean_auc']:.4f}")