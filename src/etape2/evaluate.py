import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report


def evaluate_etape2(model, val_loader, cfg, device,
                    train_acc: float = None,
                    save_path: str = "outputs/etape2_evaluation.png"):

    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs  = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).squeeze(1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    preds   = (all_probs > 0.5).astype(int)
    val_acc = (preds == all_labels).mean()

    cm          = confusion_matrix(all_labels, preds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc     = auc(fpr, tpr)

    print(f"\nVal Acc = {val_acc:.4f}")
    print(f"AUC     = {roc_auc:.4f}")

    if train_acc is not None:
        gap  = train_acc - val_acc
        flag = "POSSIBLE SURAPPRENTISSAGE" if gap > 0.08 else "OK"
        print(f"Train/Val gap = {gap:+.4f}  [{flag}]")

    print("\nRapport de classification :")
    print(classification_report(all_labels, preds,
                                target_names=["Sain", "Tumeur"], digits=4))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Étape 2 - Évaluation binaire tumeur/sain", fontweight="bold")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Sain", "Tumeur"],
                yticklabels=["Sain", "Tumeur"])
    axes[0].set_title("Matrice de confusion")
    axes[0].set_ylabel("Réel")
    axes[0].set_xlabel("Prédit")

    axes[1].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].fill_between(fpr, tpr, alpha=0.1, color="steelblue")
    axes[1].set_title("Courbe ROC")
    axes[1].set_xlabel("FPR (Faux Positifs)")
    axes[1].set_ylabel("TPR (Vrais Positifs)")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "val_acc":    val_acc,
        "roc_auc":    roc_auc,
        "all_probs":  all_probs,
        "all_labels": all_labels,
    }