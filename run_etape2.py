import sys
import yaml
import torch

sys.path.insert(0, ".")
from src.data import collect_data, build_folds
from src.etape2 import get_binary_model, train_etape2, evaluate_etape2


def main():
    # Configuration commune
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg["task"] = "binary"
    
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained = cfg.get("pretrained", True)

    # Données 
    paths, labels = collect_data(
        root=cfg["dataset_root"],
        classes=cfg["classes"],
        samples_per_class=cfg["samples_per_class"],
        seed=cfg["seed"],
    )
    folds = build_folds(paths, labels, cfg)

    # Entraînement sur tous les folds
    results = train_etape2(
        get_model_fn=lambda: get_binary_model(dropout_p=cfg["dropout_p"], pretrained=pretrained),
        folds=folds,
        cfg=cfg,
        device=device,
    )

    # Évaluation du meilleur fold
    best = max(results, key=lambda r: r["best_val_acc"])
    print(f"\nÉvaluation du meilleur fold (Fold {best['fold']})")

    evaluate_etape2(
        model=best["model"],
        val_loader=best["val_loader"],
        cfg=cfg,
        device=device,
        save_path="outputs/etape2_evaluation.png",
    )


if __name__ == "__main__":
    main()