import os
import sys
import yaml

sys.path.insert(0, ".")
from src.data import collect_data, build_folds, plot_fold_distribution


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg["task"] = "binary"

    # Collecter les données
    paths, labels = collect_data(
        root=cfg["dataset_root"],
        classes=cfg["classes"],
        samples_per_class=cfg["samples_per_class"],
        seed=cfg["seed"],
    )
    print(f"\nTotal images collectées : {len(paths)}")
    for i, cls in enumerate(cfg["classes"]):
        count = labels.count(i)
        print(f"{cls:12s} : {count} images (label {i})")

    # Construire les folds
    folds = build_folds(paths, labels, cfg)
    print(f"\nFolds construits : {len(folds)}")
    for fold in folds:
        n_train = len(fold["train_labels"])
        n_val   = len(fold["val_labels"])
        print(f"Fold {fold['fold']} : {n_train} train, {n_val} val")

    
    os.makedirs("outputs", exist_ok=True)
    plot_fold_distribution(folds, cfg["classes"], save_path="outputs/fold_distribution.png")

    return folds, cfg


if __name__ == "__main__":
    main()