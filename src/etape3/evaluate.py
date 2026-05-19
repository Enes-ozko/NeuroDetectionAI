import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize


CLASSES = ["glioma", "meningioma", "pituitary"]


def evaluate_etape3(model, val_loader, device, save_path="outputs/etape3_evaluation.png"):
    model.eval()
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs  = imgs.to(device)
            probs = F.softmax(model(imgs), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.numpy())

    all_probs  = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    all_preds  = all_probs.argmax(axis=1)

    print("\n  Rapport de classification :")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))

    cm         = confusion_matrix(all_labels, all_preds)
    labels_bin = label_binarize(all_labels, classes=[0, 1, 2])

    aucs = {}
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        aucs[cls]   = auc(fpr, tpr)
        print(f"  AUC {cls:<12} = {aucs[cls]:.4f}")

    mean_auc = np.mean(list(aucs.values()))
    print(f"  AUC moyenne    = {mean_auc:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Étape 3 — Évaluation multiclasse", fontsize=14, fontweight="bold")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES,
        ax=axes[0],
    )
    axes[0].set_title("Matrice de confusion")
    axes[0].set_xlabel("Prédit")
    axes[0].set_ylabel("Réel")

    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, (cls, color) in enumerate(zip(CLASSES, colors)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        axes[1].plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={aucs[cls]:.3f})")

    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_title("Courbes ROC — One vs Rest")
    axes[1].set_xlabel("Taux faux positifs")
    axes[1].set_ylabel("Taux vrais positifs")
    axes[1].legend(loc="lower right")
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nGraphe sauvegardé:{save_path}")

    return {
        "val_acc":  float((all_preds == all_labels).mean()),
        "aucs":     aucs,
        "mean_auc": mean_auc,
        "cm":       cm,
    }