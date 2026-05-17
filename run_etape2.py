import warnings
warnings.filterwarnings("ignore")

import os
import sys
import random
import argparse
import numpy as np
import torch
import yaml

from src.data.dataset    import collect_data
from src.data.dataloader import build_folds
from src.data.visualize  import plot_fold_distribution
from src.etape2.model    import get_binary_model
from src.etape2.train    import train_etape2
from src.etape2.evaluate import evaluate_etape2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["task"] = "binary"

    set_seed(cfg["seed"])
    device = get_device()
    os.makedirs("outputs", exist_ok=True)

    print(f"Device : {device} | Task : {cfg['task']} | Seed : {cfg['seed']}")

    paths, labels = collect_data(
        root=cfg["dataset_root"],
        classes=cfg["classes"],
        samples_per_class=cfg.get("samples_per_class"),
        seed=cfg["seed"],
    )

    print(f"Dataset : {len(paths)} images")
    for i, cls in enumerate(cfg["classes"]):
        print(f"  {cls} : {labels.count(i)}")

    folds = build_folds(paths, labels, cfg)

    plot_fold_distribution(
        folds,
        classes=["Sain", "Tumeur"],
        save_path="outputs/fold_distribution_binary.png",
    )

    if args.fold is not None:
        folds = [f for f in folds if f["fold"] == args.fold]
        if not folds:
            sys.exit(f"Fold {args.fold} introuvable.")
        print(f"Mode fold unique : {args.fold}")

    results = train_etape2(
        get_model_fn=lambda: get_binary_model(dropout_p=cfg.get("dropout_p", 0.5)),
        folds=folds,
        cfg=cfg,
        device=device,
    )

    best_result = max(results, key=lambda r: r["best_val_acc"])
    best_fold   = next(f for f in folds if f["fold"] == best_result["fold"])

    print(f"Meilleur fold : {best_result['fold']} (val_acc={best_result['best_val_acc']:.3f})")

    save_path_model = "outputs/model_etape2.pth"
    torch.save(best_result["model"].state_dict(), save_path_model)

    metrics = evaluate_etape2(
        model=best_result["model"],
        val_loader=best_result["val_loader"],
        cfg=cfg,
        device=device,
        val_dataset=best_fold["val_dataset"],
        train_acc=best_result["best_val_acc"],
        save_path="outputs/etape2_evaluation.png",
    )

    accs = [r["best_val_acc"] for r in results]
    print(f"Val acc : {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"AUC     : {metrics['roc_auc']:.4f}")
    print(f"Ambigus : {metrics['n_ambig']}/{len(metrics['all_probs'])} ({100*metrics['n_ambig']/len(metrics['all_probs']):.1f}%)")