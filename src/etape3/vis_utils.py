import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES_FR = ["Gliome", "Méningiome", "Pituitaire"]
COLORS     = ["#e74c3c", "#2ecc71", "#3498db"]

def plot_simple_inference(img_rgb, probs, save_path=None):
    pred_idx = np.argmax(probs)
    pred_label = CLASSES_FR[pred_idx]
    confidence = probs[pred_idx] * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Prédiction du modèle : {pred_label} ({confidence:.1f}%)", fontsize=13, fontweight="bold")

    axes[0].imshow(img_rgb)
    axes[0].axis("off")
    axes[0].set_title("Image IRM analysée", fontsize=11)

    bars = axes[1].bar(
        CLASSES_FR, probs, 
        color=[COLORS[i] if i == pred_idx else "gray" for i in range(3)]
    )
    
    # Affichage des pourcentages sur les barres
    for bar, prob in zip(bars, probs):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{prob * 100:.1f}%", ha="center", va="bottom", fontweight="bold")

    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel("Probabilité")
    axes[1].set_title("Répartition par classe")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {save_path}")
    
    return fig