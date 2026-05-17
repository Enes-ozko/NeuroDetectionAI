import torch
import yaml
import numpy as np
import random

from src.data.dataset import collect_data, BrainTumorDataset
from src.data.dataloader import build_folds

from src.etape3.model import build_model
from src.etape3.train import train_etape3
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
        if l == 2:      # notumor → exclu
            continue
        filtered_paths.append(p)
        filtered_labels.append(LABEL_REMAP[l])
    return filtered_paths, filtered_labels


if __name__ == "__main__":

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device : {device}")

    CLASSES_4 = ["glioma", "meningioma", "notumor", "pituitary"]
    all_paths, all_labels = collect_data(
        root=cfg["dataset_root"],
        classes=CLASSES_4,
        samples_per_class=cfg.get("samples_per_class", None),
        seed=cfg["seed"],
    )
    print(f"Dataset brut : {len(all_paths)} images")

    paths, labels = filter_and_remap(all_paths, all_labels)
    print(f"Dataset filtré (sans notumor) : {len(paths)} images")
    for name, idx in [("glioma", 0), ("meningioma", 1), ("pituitary", 2)]:
        print(f"  {name:<12} : {labels.count(idx)}")

    cfg_etape3 = {**cfg, "task": "multiclass"}

    print(f"\nConstruction des {cfg['n_folds']} folds stratifiés...")
    folds = build_folds(paths, labels, cfg_etape3)

    print("\nEntraînement Étape 3")
    results = train_etape3(
        get_model_fn=lambda: build_model(num_classes=3, dropout=cfg.get("dropout_p", 0.3)),
        folds=folds,
        cfg=cfg_etape3,
        device=device,
    )

    best = max(results, key=lambda r: r["best_val_acc"])
    print(f"\n Évaluation du meilleur fold (Fold {best['fold']}) ")
    metrics = evaluate_etape3(
        model=best["model"],
        val_loader=best["val_loader"],
        device=device,
        save_path="outputs/etape3_evaluation.png",
    )

    torch.save(best["model"].state_dict(), "outputs/model_etape3.pth")
    print("Modèle sauvegardé → outputs/model_etape3.pth")

    print(f"\nVal acc : {metrics['val_acc']:.3f}")
    print(f"AUC moy : {metrics['mean_auc']:.4f}")