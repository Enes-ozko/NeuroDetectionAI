import numpy as np
import torch
import torch.nn.functional as F
import cv2


class GradCAMPlusPlus:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None

        target = model.features[-1]
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, img_tensor, class_idx):
        self.model.eval()
        self.gradients   = None
        self.activations = None

        img_tensor = img_tensor.requires_grad_(True)
        logits     = self.model(img_tensor)

        self.model.zero_grad()
        logits[0, class_idx].backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Les hooks Grad-CAM n'ont pas capturé les gradients.")

        grads    = self.gradients.detach()
        acts     = self.activations.detach()
        grads_sq = grads ** 2
        grads_cu = grads ** 3
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)

        denom   = 2 * grads_sq + sum_acts * grads_cu
        denom   = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha   = grads_sq / denom
        weights = (alpha * F.relu(grads)).mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * acts).sum(dim=1, keepdim=True)).squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam


def apply_heatmap(img_rgb, cam, alpha=0.45):
    H, W        = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (W, H))
    heatmap     = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (alpha * heatmap_rgb + (1 - alpha) * img_rgb).astype(np.uint8)


def extract_bounding_box(cam, img_size, threshold=0.4):
    W, H        = img_size
    cam_resized = cv2.resize(cam, (W, H))
    binary      = (cam_resized > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    return cv2.boundingRect(max(contours, key=cv2.contourArea))


def annotate_image(img_rgb, cam, label, confidence, threshold=0.4, alpha=0.45):
    H, W    = img_rgb.shape[:2]
    overlay = apply_heatmap(img_rgb, cam, alpha)
    bbox    = extract_bounding_box(cam, img_size=(W, H), threshold=threshold)

    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    text = f"{label} {confidence * 100:.0f}%"
    cv2.putText(overlay, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 3)
    cv2.putText(overlay, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    return overlay, bbox