import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Importation de ton modèle
from src.etape3.model import build_model

# 1. NOTRE FILTRE OPENCV (Directement intégré pour le Grad-CAM)
class RemoveSkullAndFace(object):
    def __call__(self, img):
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img 
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        img_cleaned = cv2.bitwise_and(img_np, img_np, mask=mask)
        return Image.fromarray(img_cleaned)

def run_gradcam(img_path="test.jpg", save_path="outputs/gradcam_result.png"):
    device = torch.device("cpu")
    model = build_model(num_classes=3, dropout=0.0)
    
    # Chargement du nouveau modèle Kaggle
    model.load_state_dict(torch.load("outputs/model_etape3.pth", map_location=device))
    
    for param in model.parameters():
        param.requires_grad = True
    model.eval()

    activations, gradients = None, None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    target_layer = model.features[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    # 2. PRÉPARATION DE L'IMAGE AVEC SKULL STRIPPING
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        RemoveSkullAndFace(), # <-- Le filtre est appliqué ici !
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # On charge l'image de base
    raw_pil = Image.open(img_path).convert("RGB")
    
    # On génère une version "nettoyée" pour l'affichage visuel à la fin
    clean_pil = RemoveSkullAndFace()(raw_pil)
    clean_img_cv2 = np.array(clean_pil)
    
    # On génère le tenseur pour le modèle
    input_tensor = transform(raw_pil).unsqueeze(0).to(device)

    # 3. PASSAGE DANS LE RÉSEAU
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    
    model.zero_grad()
    output[0, pred_class].backward()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 4. SUPERPOSITION OPENCV (Sur le cerveau nettoyé)
    heatmap_resized = cv2.resize(heatmap, (clean_img_cv2.shape[1], clean_img_cv2.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # On applique le masque pour ne pas colorer le fond noir
    gray_clean = cv2.cvtColor(clean_img_cv2, cv2.COLOR_RGB2GRAY)
    _, brain_mask = cv2.threshold(gray_clean, 1, 255, cv2.THRESH_BINARY)
    heatmap_colored = cv2.bitwise_and(heatmap_colored, heatmap_colored, mask=brain_mask)

    superimposed_img = heatmap_colored * 0.4 + clean_img_cv2 * 0.6

    # 5. SAUVEGARDE
    plt.figure(figsize=(8, 8))
    plt.imshow(superimposed_img.astype(np.uint8))
    plt.title(f"Grad-CAM (Classe prédite : {pred_class}) - Filtre OpenCV Actif")
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Explicabilité générée dans : {save_path}")

if __name__ == "__main__":
    run_gradcam()