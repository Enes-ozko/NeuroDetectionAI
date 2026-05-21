import os
import yaml
import torch
import random
import numpy as np

from src.data.dataset import collect_data
from src.data.dataloader import build_folds

from src.etape3.model import build_model
from src.etape3.train import train_etape3
from src.etape3.evaluate import evaluate_etape3

def set_seed(seed: int):
    """Fixe la graine aléatoire pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#on enlève 'notumor'
LABEL_REMAP = {0: 0, 1: 1, 3: 2}

def filter_and_remap(paths, labels):
    """Filtre les images saines et re-mappe les étiquettes de 0 à 2."""
    filtered_paths  = []
    filtered_labels = []
    for p, l in zip(paths, labels):
        if l == 2:  # On ignore notumor
            continue
        filtered_paths.append(p)
        filtered_labels.append(LABEL_REMAP[l])
    return filtered_paths, filtered_labels

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Appareil (Device) utilisé : {device}")

    #Collecte des données brutes via dataset.py
    print("\nCollecte des données")
    all_paths, all_labels = collect_data(
        root=cfg["dataset_root"],
        classes=cfg["classes"],
        samples_per_class=cfg.get("samples_per_class", None),
        seed=cfg["seed"],
    )
    print(f"Dataset brut (4 classes) : {len(all_paths)} images")

    #3 types de tumeurs
    paths, labels = filter_and_remap(all_paths, all_labels)
    print(f"Dataset filtré (3 tumeurs)  : {len(paths)} images")

    cfg_etape3 = {**cfg, "task": "binary"}

    #Construction des K-Folds
    print("\nCréation des folds")
    folds = build_folds(paths, labels, cfg_etape3)

    #Entraînement
    print("\nEntraînement")
    results = train_etape3(
        get_model_fn=lambda: build_model(num_classes=3, dropout=cfg.get("dropout_p", 0.3)),
        folds=folds,
        cfg=cfg_etape3,
        device=device,
    )

    #Évaluation du meilleur modèle 
    best = max(results, key=lambda r: r["best_val_acc"])
    
    print(f"\nÉvaluation finale du meilleur modèle (Fold {best['fold']})")
    val_acc_finale = evaluate_etape3(
        model=best["model"],
        val_loader=best["val_loader"],
        device=device,
        save_path="outputs/etape3_evaluation.png",
    )

    chemin_sauvegarde = "outputs/model_etape3.pth"
    torch.save(best["model"].state_dict(), chemin_sauvegarde)
    print(f"Précision finale de validation (Val Acc) : {val_acc_finale * 100:.2f}%")