import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize


CLASSES_FR = ["Gliome", "Méningiome", "Pituitaire"]
COLORS     = ["#e74c3c", "#2ecc71", "#3498db"]


def plot_inference(img_annotated, result, save_path=None):
    probs       = result["probs"]
    scenario    = result["scenario"]
    message     = result["message"]
    pred_idx    = result["pred_idx"]
    title_color = {"A": "#27ae60", "B": "#e74c3c", "C": "#f39c12"}.get(scenario, "black")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Scénario {scenario} - {message}", fontsize=12, fontweight="bold",
                 color=title_color, y=1.01)

    axes[0].imshow(img_annotated)
    axes[0].axis("off")
    axes[0].set_title("Zone d'activation (Grad-CAM++)", fontsize=11)

    bars = axes[1].bar(
        CLASSES_FR, probs,
        color=[COLORS[i] if i == pred_idx else "#bdc3c7" for i in range(3)],
        edgecolor="white", linewidth=0.8,
    )
    for bar, prob in zip(bars, probs):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{prob * 100:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel("Probabilité softmax", fontsize=10)
    axes[1].set_title("Répartition par type de tumeur", fontsize=11)
    axes[1].axhline(y=0.55, color="gray", linestyle="--", linewidth=0.8, label="Seuil de confiance")
    axes[1].text(0.98, 0.97, f"Entropie : {result['entropy']:.3f} / {np.log(3):.3f}",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=8, color="gray")
    axes[1].legend(fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée  {save_path}")

    return fig


def plot_calibration(all_probs, all_labels, save_path=None, n_bins=10):
    labels_bin = label_binarize(all_labels, classes=[0, 1, 2])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Calibration des probabilités - One vs Rest", fontsize=13, fontweight="bold")

    for i, (cls_fr, color, ax) in enumerate(zip(CLASSES_FR, COLORS, axes)):
        fraction_pos, mean_pred = calibration_curve(
            labels_bin[:, i], all_probs[:, i],
            n_bins=n_bins, strategy="uniform",
        )
        ece = float(np.mean(np.abs(fraction_pos - mean_pred)))

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Calibration parfaite")
        ax.plot(mean_pred, fraction_pos, "o-", color=color, linewidth=2, markersize=6, label=cls_fr)
        ax.set_title(f"{cls_fr}\nECE = {ece:.3f}", fontsize=11)
        ax.set_xlabel("Probabilité prédite")
        ax.set_ylabel("Fraction de positifs")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Calibration sauvegardée:{save_path}")

    return fig