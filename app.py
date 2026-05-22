import os
import io
import sys
import yaml
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, ".")
from src.data.transforms import get_transforms
from src.etape2.model import get_binary_model
from src.etape3.model import build_model as get_multiclass_model

app = Flask(__name__, static_folder="static")
CORS(app)

# ── Chargement config ──────────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# ── Device ─────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Device : {device}")

# ── Chargement des modèles (une seule fois au démarrage) ───────────────────────
CLASSES = ["Gliome", "Méningiome", "Pituitaire"]

model_bin = get_binary_model(dropout_p=0.0, pretrained=False)
model_bin.load_state_dict(
    torch.load("outputs/mobilenet_binaire.pth", map_location=device)
)
model_bin.to(device).eval()

model_multi = get_multiclass_model(num_classes=3, dropout=0.0)
model_multi.load_state_dict(
    torch.load("outputs/model_etape3.pth", map_location=device)
)
model_multi.to(device).eval()

transform = get_transforms("val", cfg["image_size"])

print("Modèles chargés.")

# ── Route principale : page HTML ───────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

# ── Route prédiction ───────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Impossible de lire l'image"}), 400

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Étape 1 : détection binaire tumeur / sain
        logits_bin = model_bin(img_tensor)
        if logits_bin.shape[1] == 1:
            score_tumeur = torch.sigmoid(logits_bin).item()
        else:
            score_tumeur = F.softmax(logits_bin, dim=1)[0][1].item()

        if score_tumeur < 0.5:
            return jsonify({
                "has_tumor":   False,
                "bin_score":   round(score_tumeur, 4),
                "sain_conf":   round((1.0 - score_tumeur) * 100, 1),
                "tumor_class": None,
                "probs":       None,
            })

        # Étape 2 : classification multiclasse
        probs_multi = F.softmax(model_multi(img_tensor), dim=1)[0].tolist()
        pred_idx    = int(torch.tensor(probs_multi).argmax())

        return jsonify({
            "has_tumor":   True,
            "bin_score":   round(score_tumeur, 4),
            "tumor_conf":  round(score_tumeur * 100, 1),
            "tumor_class": CLASSES[pred_idx],
            "pred_idx":    pred_idx,
            "probs":       [round(p * 100, 1) for p in probs_multi],
            "classes":     CLASSES,
        })


if __name__ == "__main__":
    app.run(debug=True, port=5000)