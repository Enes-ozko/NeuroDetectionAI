import io
import os
import sys
import json
import base64
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.etape2.model     import get_binary_model
from src.etape3.model     import build_model
from src.etape3.ood_logic import classify_prediction
from src.etape3.gradcam   import GradCAMPlusPlus, annotate_image

MODEL_E2_PATH  = "outputs/model_etape2.pth"
MODEL_E3_PATH  = "outputs/model_etape3.pth"
SESSION_FILE   = "outputs/session.json"
IMAGES_DIR     = "outputs/images"
IMG_SIZE       = 224
THRESHOLD_LOW  = 0.3
THRESHOLD_HIGH = 0.7

os.makedirs(IMAGES_DIR, exist_ok=True)

DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available()  else
    torch.device("cuda") if torch.cuda.is_available()           else
    torch.device("cpu")
)

PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_e2 = None
model_e3 = None


def load_models():
    global model_e2, model_e3
    if os.path.exists(MODEL_E2_PATH):
        m = get_binary_model(dropout_p=0.0)
        m.load_state_dict(torch.load(MODEL_E2_PATH, map_location=DEVICE))
        m.to(DEVICE).eval()
        model_e2 = m
    if os.path.exists(MODEL_E3_PATH):
        m = build_model(num_classes=3, dropout=0.0)
        m.load_state_dict(torch.load(MODEL_E3_PATH, map_location=DEVICE))
        m.to(DEVICE).eval()
        model_e3 = m


load_models()


def _load_session():
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r") as f:
            return json.load(f)
    return []


def _save_session(data):
    with open(SESSION_FILE, "w") as f:
        json.dump(data, f)


def _safe_stem(filename):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in filename)


def save_image_to_disk(img_bgr, filename, suffix=""):
    stem = os.path.splitext(_safe_stem(filename))[0]
    name = f"{stem}{suffix}.jpg"
    path = os.path.join(IMAGES_DIR, name)
    cv2.imwrite(path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return f"/images/{name}"


app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/static", StaticFiles(directory="interface"), name="static")


@app.get("/")
def root():
    return FileResponse("interface/index.html")


@app.get("/stats")
def stats_page():
    return FileResponse("interface/stats.html")


@app.get("/status")
def status():
    return {
        "device": str(DEVICE),
        "e2": model_e2 is not None,
        "e3": model_e3 is not None,
    }


@app.get("/session")
def get_session():
    return _load_session()


@app.post("/session")
def post_session(new_results: list = Body(...)):
    existing = _load_session()
    existing.extend(new_results)
    _save_session(existing)
    return {"saved": len(new_results), "total": len(existing)}


@app.delete("/session")
def delete_session():
    _save_session([])
    return {"cleared": True}


@app.delete("/session/{filename}")
def delete_session_entry(filename: str):
    existing = _load_session()
    filtered = [r for r in existing if r.get("filename") != filename]
    _save_session(filtered)
    stem = os.path.splitext(_safe_stem(filename))[0]
    for suffix in ["", "_gradcam"]:
        p = os.path.join(IMAGES_DIR, f"{stem}{suffix}.jpg")
        if os.path.exists(p):
            os.remove(p)
    return {"deleted": len(existing) - len(filtered), "total": len(filtered)}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data    = await file.read()
    pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    img_rgb = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    src_url = save_image_to_disk(img_bgr, file.filename, suffix="")

    if model_e2 is None:
        return {"error": "model_e2 not loaded"}

    tensor = PREPROCESS(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        proba = torch.sigmoid(model_e2(tensor)).item()

    if proba < THRESHOLD_LOW:
        return {"scenario": "sain", "proba": proba, "src_url": src_url}

    if proba <= THRESHOLD_HIGH:
        return {"scenario": "ambig", "proba": proba, "src_url": src_url}

    if model_e3 is None:
        return {"scenario": "tumeur_no_e3", "proba": proba, "src_url": src_url}

    with torch.no_grad():
        logits = model_e3(tensor)

    r = classify_prediction(logits)

    cam_extractor      = GradCAMPlusPlus(model_e3)
    cam                = cam_extractor.generate(tensor.clone().requires_grad_(True), class_idx=r["pred_idx"])
    annotated, bbox    = annotate_image(img_rgb, cam, r["pred_label"], r["confidence"])
    annotated_bgr      = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    annotated_url      = save_image_to_disk(annotated_bgr, file.filename, suffix="_gradcam")

    classes = ["Gliome", "Meningiome", "Pituitaire"]

    return {
        "scenario":      "tumeur",
        "proba":         proba,
        "type":          classes[r["pred_idx"]],
        "probs":         [float(p) for p in r["probs"]],
        "predIdx":       r["pred_idx"],
        "confidence":    float(r["confidence"]),
        "entropy":       float(r["entropy"]),
        "entropy_norm":  float(r["entropy_norm"]),
        "ood":           r["scenario"] == "B",
        "bbox":          {"x": int(bbox[0]), "y": int(bbox[1]), "w": int(bbox[2]), "h": int(bbox[3])} if bbox else None,
        "src_url":       src_url,
        "annotated_url": annotated_url,
    }