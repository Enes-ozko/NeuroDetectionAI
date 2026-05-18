import argparse
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.etape2.model     import get_binary_model
from src.etape3.model     import build_model
from src.etape3.ood_logic import classify_prediction
from src.etape3.gradcam   import GradCAMPlusPlus, annotate_image


MODEL_E2_PATH  = "outputs/model_etape2.pth"
MODEL_E3_PATH  = "outputs/model_etape3.pth"
IMG_SIZE       = 224
THRESHOLD_LOW  = 0.3
THRESHOLD_HIGH = 0.7
CLASSES_FR     = ["Gliome", "Méningiome", "Pituitaire"]
COLORS         = ["#e74c3c", "#2ecc71", "#3498db"]

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()           else
    torch.device("mps")  if torch.backends.mps.is_available()   else
    torch.device("cpu")
)

PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_etape2(path):
    model = get_binary_model(dropout_p=0.0)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_etape3(path):
    model = build_model(num_classes=3, dropout=0.0)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def make_figure_sain(img_rgb, proba, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"Aucune anomalie détectée  —  score tumeur : {proba * 100:.1f}%",
        fontsize=12, fontweight="bold", color="#27ae60",
    )
    axes[0].imshow(img_rgb)
    axes[0].axis("off")
    axes[0].set_title("Image analysée")

    axes[1].barh(["Tumeur", "Sain"], [proba, 1 - proba], color=["#e74c3c", "#27ae60"])
    axes[1].set_xlim(0, 1)
    axes[1].axvline(x=THRESHOLD_LOW,  color="gray", linestyle="--", lw=1,
                    label=f"Seuil bas ({THRESHOLD_LOW})")
    axes[1].axvline(x=THRESHOLD_HIGH, color="gray", linestyle=":",  lw=1,
                    label=f"Seuil haut ({THRESHOLD_HIGH})")
    axes[1].set_title("Score de détection")
    axes[1].legend(fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure sauvegardée → {save_path}")


def make_figure_ambigu(img_rgb, proba, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"Signal détecté mais non concluant  —  score tumeur : {proba * 100:.1f}%  —  relecture recommandée",
        fontsize=10, fontweight="bold", color="#f39c12",
    )
    axes[0].imshow(img_rgb)
    axes[0].axis("off")
    axes[0].set_title("Image analysée")

    axes[1].barh(["Tumeur", "Sain"], [proba, 1 - proba], color=["#e67e22", "#f39c12"])
    axes[1].set_xlim(0, 1)
    axes[1].axvspan(THRESHOLD_LOW, THRESHOLD_HIGH, alpha=0.15, color="orange",
                    label="Zone de doute")
    axes[1].axvline(x=THRESHOLD_LOW,  color="gray", linestyle="--", lw=1)
    axes[1].axvline(x=THRESHOLD_HIGH, color="gray", linestyle=":",  lw=1)
    axes[1].set_title("Score de détection")
    axes[1].legend(fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure sauvegardée → {save_path}")


def make_figure_tumeur(img_annotated, proba_tumeur, result_e3, bbox, save_path):
    scenario    = result_e3["scenario"]
    message     = result_e3["message"]
    probs_e3    = result_e3["probs"]
    pred_idx    = result_e3["pred_idx"]
    entropy     = result_e3["entropy"]
    title_color = {"A": "#27ae60", "B": "#e74c3c", "C": "#f39c12"}.get(scenario, "black")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Tumeur détectée ({proba_tumeur * 100:.1f}%)   {message}",
        fontsize=11, fontweight="bold", color=title_color, y=1.02,
    )

    axes[0].imshow(img_annotated)
    axes[0].axis("off")
    title_cam = "Zone d'activation (Grad-CAM++)"
    if bbox:
        x, y, w, h = bbox
        title_cam += f"\nrégion : x={x}, y={y}, {w}×{h} px"
    axes[0].set_title(title_cam, fontsize=10)

    bar_colors = [COLORS[i] if i == pred_idx else "#bdc3c7" for i in range(3)]
    bars = axes[1].bar(CLASSES_FR, probs_e3, color=bar_colors, edgecolor="white", lw=0.8)
    for bar, prob in zip(bars, probs_e3):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{prob * 100:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    axes[1].set_ylim(0, 1.15)
    axes[1].set_ylabel("Probabilité softmax")
    axes[1].set_title("Répartition par type de tumeur")
    axes[1].axhline(y=0.55, color="gray", linestyle="--", lw=0.8, label="Seuil de confiance")
    axes[1].text(0.98, 0.97, f"Entropie : {entropy:.3f}", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=8, color="gray")
    axes[1].legend(fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)

    axes[2].barh(["Tumeur", "Sain"], [proba_tumeur, 1 - proba_tumeur],
                 color=["#e74c3c", "#27ae60"])
    axes[2].set_xlim(0, 1)
    axes[2].axvline(x=THRESHOLD_HIGH, color="gray", linestyle="--", lw=1,
                    label=f"Seuil de détection ({THRESHOLD_HIGH})")
    axes[2].set_title("Score de détection initial")
    axes[2].legend(fontsize=8)
    axes[2].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_inference(image_path, save_path="outputs/inference_result.png"):
    os.makedirs("outputs", exist_ok=True)
    print(f"Image  : {image_path}")
    print(f"Device : {DEVICE}")

    pil_img    = Image.open(image_path).convert("RGB")
    img_rgb    = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    img_tensor = PREPROCESS(pil_img).unsqueeze(0).to(DEVICE)

    model_e2 = load_etape2(MODEL_E2_PATH)

    with torch.no_grad():
        proba_tumeur = torch.sigmoid(model_e2(img_tensor)).item()

    print(f"Score tumeur : {proba_tumeur * 100:.1f}%")

    if proba_tumeur < THRESHOLD_LOW:
        print(f"\nAucune anomalie détectée ({proba_tumeur * 100:.1f}%)")
        make_figure_sain(img_rgb, proba_tumeur, save_path)
        return

    if proba_tumeur <= THRESHOLD_HIGH:
        print(f"\nSignal ambigu ({proba_tumeur * 100:.1f}%) — le modèle hésite, relecture recommandée")
        make_figure_ambigu(img_rgb, proba_tumeur, save_path)
        return

    print(f"Tumeur probable ({proba_tumeur * 100:.1f}%), lancement de la classification")

    model_e3 = load_etape3(MODEL_E3_PATH)

    with torch.no_grad():
        logits_e3 = model_e3(img_tensor)

    result_e3 = classify_prediction(logits_e3)
    print(f"\n{result_e3['message']}")
    print(f"Probabilités  : { {c: round(float(p), 3) for c, p in zip(['glioma', 'meningioma', 'pituitary'], result_e3['probs'])} }")
    print(f"Entropie      : {result_e3['entropy']:.4f} (normalisée : {result_e3['entropy_norm']:.3f})")

    print("\nLocalisation de la zone tumorale")
    model_grad      = load_etape3(MODEL_E3_PATH)
    cam_extractor   = GradCAMPlusPlus(model_grad)
    img_tensor_grad = PREPROCESS(pil_img).unsqueeze(0).to(DEVICE)
    cam             = cam_extractor.generate(img_tensor_grad, class_idx=result_e3["pred_idx"])

    img_annotated, bbox = annotate_image(
        img_rgb    = img_rgb,
        cam        = cam,
        label      = result_e3["pred_label"],
        confidence = result_e3["confidence"],
    )

    if bbox:
        x, y, w, h = bbox
        print(f"Région détectée : x={x}, y={y}, largeur={w}, hauteur={h}")
    else:
        print("Aucune région principale isolée")

    make_figure_tumeur(img_annotated, proba_tumeur, result_e3, bbox, save_path)

    print(f"\nType           : {result_e3['pred_label']} — scénario {result_e3['scenario']}")
    print(f"Conclusion     : {result_e3['message']}")
    if bbox:
        print(f"Zone           : x={x}, y={y}, {w}×{h} px")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  type=str, default="test3M.jpg")
    parser.add_argument("--output", type=str, default="outputs/inference_result.png")
    args = parser.parse_args()

    run_inference(image_path=args.image, save_path=args.output)