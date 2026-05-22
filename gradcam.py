import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.data.transforms import get_transforms
from src.etape3.model import build_model

def run_gradcam(img_path="test.jpg", save_path="outputs/gradcam_result.png"):
    # 1. Chargement du modèle (sur CPU pour éviter les soucis de mémoire)
    device = torch.device("cpu")
    model = build_model(num_classes=3, dropout=0.0)
    model.load_state_dict(torch.load("outputs/model_etape3.pth", map_location=device))

    for param in model.parameters():
        param.requires_grad = True
        
    model.eval()

    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    target_layer = model.features[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    #Préparation de l'image
    original_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    transform = get_transforms("val", 224)
    input_tensor = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

    #passage dans le réseau
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    
    model.zero_grad()
    output[0, pred_class].backward()

    #Grad-CAM
    # On fait la moyenne des gradients
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    # On pondère chaque carte d'activation par son importance
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    # On fusionne toutes les cartes en une seule Heatmap (Carte de chaleur)
    heatmap = torch.mean(activations, dim=1).squeeze().detach().numpy()
    heatmap = np.maximum(heatmap, 0) # ReLU : on garde que les valeurs positives
    heatmap /= np.max(heatmap)       # Normalisation entre 0 et 1

    # 6. Superposition Visuelle (OpenCV)
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = heatmap_colored * 0.4 + original_img * 0.6 # Mélange à 40/60

    # Sauvegarde
    plt.figure(figsize=(8, 8))
    plt.imshow(superimposed_img.astype(np.uint8))
    plt.title(f"Grad-CAM (Classe prédite : {pred_class})")
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')

if __name__ == "__main__":
    run_gradcam()