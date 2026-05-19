
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

from src.data.transforms import get_transforms



def _predict_single(model, val_loader, device):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs  = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).squeeze(1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

    return np.array(all_probs), np.array(all_labels)



def _predict_tta(model, val_dataset, cfg, device, n_augments: int = 5):

    from PIL import Image
    tta_transform = get_transforms("tta", cfg["image_size"])

    model.eval()
    all_probs, all_labels = [], []

    for path, label in zip(val_dataset.paths, val_dataset.labels):
        img      = Image.open(path).convert("RGB")
        pil_imgs = [img] * n_augments
        batch    = torch.stack([tta_transform(i) for i in pil_imgs]).to(device)

        with torch.no_grad():
            probs = torch.sigmoid(model(batch)).squeeze(1).cpu().numpy()

        all_probs.append(probs.mean())   
        all_labels.append(label)

    return np.array(all_probs), np.array(all_labels)


# Évaluation principale 
def evaluate_etape2(model, val_loader, cfg, device, val_dataset=None,
                    train_acc: float = None, save_path: str = "outputs/etape2_evaluation.png"):
   
    n_tta = cfg.get("tta_n_augments", 0)

    if n_tta > 0 and val_dataset is not None:
        all_probs, all_labels = _predict_tta(model, val_dataset, cfg, device, n_augments=n_tta)
    else:
        all_probs, all_labels = _predict_single(model, val_loader, device)

    preds = np.where(
        all_probs > cfg["threshold_high"], 1,
        np.where(all_probs < cfg["threshold_low"], 0, -1)
    )
    mask_clear = preds != -1
    n_ambig    = np.sum(preds == -1)

    # Métriques 
    preds_binary = (all_probs > 0.5).astype(int)
    val_acc      = (preds_binary == all_labels).mean()

    cm          = confusion_matrix(all_labels[mask_clear], preds[mask_clear])
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc     = auc(fpr, tpr)

    print(f"\nAUC            = {roc_auc:.4f}")
    print(f"Val Acc (0.5)  = {val_acc:.4f}")
    print(f"Ambigus        = {n_ambig}/{len(preds)} "
          f"({100*n_ambig/len(preds):.1f}%)")


    if train_acc is not None:
        gap = train_acc - val_acc
        flag = "POSSIBLE OVERFITTING" if gap > 0.08 else "OK"
        print(f"Train/Val gap  = {gap:+.4f}  {flag}")

    print("\nRapport de classification (seuil 0.5) :")
    print(classification_report(all_labels, preds_binary,
                                target_names=["Sain", "Tumeur"], digits=4))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Étape 2 - Évaluation binaire tumeur/sain", fontweight="bold")

    # Matrice de confusion 
    if cm.size > 0:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                    xticklabels=["Sain", "Tumeur"],
                    yticklabels=["Sain", "Tumeur"])
    axes[0].set_title(f"Matrice de confusion\n(prédictions nettes : {mask_clear.sum()})")
    axes[0].set_ylabel("Réel")
    axes[0].set_xlabel("Prédit")

    # Courbe ROC
    axes[1].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].fill_between(fpr, tpr, alpha=0.1, color="steelblue")
    axes[1].set_title("Courbe ROC")
    axes[1].set_xlabel("FPR (Faux Positifs)")
    axes[1].set_ylabel("TPR (Vrais Positifs)")
    axes[1].legend(loc="lower right")

    # Confiance slice par slice avec zones de décision
    colors = np.where(all_probs > cfg["threshold_high"], "red",
             np.where(all_probs < cfg["threshold_low"], "green", "orange"))
    axes[2].scatter(range(len(all_probs)), all_probs, c=colors, s=4, alpha=0.6)
    axes[2].axhline(cfg["threshold_high"], color="red",   linestyle="--", lw=1.2,
                    label=f"Seuil haut ({cfg['threshold_high']})")
    axes[2].axhline(cfg["threshold_low"],  color="green", linestyle="--", lw=1.2,
                    label=f"Seuil bas ({cfg['threshold_low']})")
    axes[2].axhspan(cfg["threshold_low"], cfg["threshold_high"],
                    alpha=0.08, color="orange", label="Zone ambiguë")
    axes[2].set_title(f"Confiance slice par slice\n(TTA × {n_tta})" if n_tta > 0
                      else "Confiance slice par slice")
    axes[2].set_xlabel("Index slice")
    axes[2].set_ylabel("Probabilité d'anomalie")
    axes[2].legend(fontsize=8)
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nGraphe sauvegardé:{save_path}")

    return {
        "val_acc":  val_acc,
        "roc_auc":  roc_auc,
        "n_ambig":  n_ambig,
        "all_probs": all_probs,
        "all_labels": all_labels,
    }