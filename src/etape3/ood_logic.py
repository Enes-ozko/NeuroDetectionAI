import numpy as np
import torch
import torch.nn.functional as F


CLASSES   = ["glioma", "meningioma", "pituitary"]
LABELS_FR = {"glioma": "Gliome", "meningioma": "Méningiome", "pituitary": "Pituitaire"}
H_MAX     = np.log(len(CLASSES))


def shannon_entropy(probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def classify_prediction(logits, ood_ratio=0.75, confidence_threshold=0.55):
    probs        = F.softmax(logits.squeeze(), dim=-1).detach().cpu().numpy()
    pred_idx     = int(np.argmax(probs))
    confidence   = float(probs[pred_idx])
    entropy      = shannon_entropy(probs)
    entropy_norm = entropy / H_MAX
    pred_class   = CLASSES[pred_idx]
    pred_label   = LABELS_FR[pred_class]

    if entropy > ood_ratio * H_MAX:
        scenario = "B"
        message  = "Tumeur confirmée mais type biologique inconnu — consultation humaine requise"
    elif confidence >= confidence_threshold:
        scenario = "A"
        message  = f"{pred_label} confirmé ({confidence * 100:.1f}%)"
    else:
        scenario = "C"
        message  = f"Type probable : {pred_label} ({confidence * 100:.1f}%) — confiance insuffisante"

    return {
        "probs":        probs,
        "pred_idx":     pred_idx,
        "pred_class":   pred_class,
        "pred_label":   pred_label,
        "confidence":   confidence,
        "entropy":      entropy,
        "entropy_norm": entropy_norm,
        "scenario":     scenario,
        "message":      message,
    }