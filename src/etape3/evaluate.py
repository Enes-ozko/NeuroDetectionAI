import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# On fixe les 3 classes cibles
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

    print("\nRapport de classification :")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    # Binarisation sur 3 classes (0, 1, 2)
    labels_bin = label_binarize(all_labels, classes=[0, 1, 2])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Évaluation Multiclasse (3 Tumeurs)", fontsize=14, fontweight="bold")

    # 1. Matrice de confusion
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0])
    axes[0].set_title("Matrice de confusion")
    axes[0].set_xlabel("Prédictions")
    axes[0].set_ylabel("Vérité terrain")

    # 2. Courbes ROC 
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, (cls, color) in enumerate(zip(CLASSES, colors)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={roc_auc:.3f})")

    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_title("Courbes ROC")
    axes[1].set_xlabel("Taux faux positifs")
    axes[1].set_ylabel("Taux vrais positifs")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return float((all_preds == all_labels).mean())