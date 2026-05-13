import numpy as np
import matplotlib.pyplot as plt


def plot_fold_distribution(folds, classes, save_path="outputs/fold_distribution.png"):
    n_folds   = len(folds)
    n_classes = len(classes)
    x         = np.arange(n_classes)
    width     = 0.35

    fig, axes = plt.subplots(1, n_folds, figsize=(4 * n_folds, 4), sharey=True)
    fig.suptitle("Distribution des classes par fold", fontweight="bold")

    for ax, fold in zip(axes, folds):
        train = [fold["train_labels"].count(i) for i in range(n_classes)]
        val   = [fold["val_labels"].count(i)   for i in range(n_classes)]

        ax.bar(x - width / 2, train, width, label="Train")
        ax.bar(x + width / 2, val,   width, label="Val", alpha=0.8)
        ax.set_title(f"Fold {fold['fold']}")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=20, ha="right", fontsize=8)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()