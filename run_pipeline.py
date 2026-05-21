import os
import sys
import yaml
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, ".")
from src.data.transforms import get_transforms
from src.etape2.model import get_binary_model
from src.etape3.model import build_model as get_multiclass_model

CLASSES_ETAPE3 = ["Gliome", "Méningiome", "Pituitaire"]

def load_models(cfg, device):
    print("Chargement des modèles")
    
    model_bin = get_binary_model(dropout_p=0.0, pretrained=False)
    model_bin.load_state_dict(torch.load("outputs/mobilenet_binaire.pth", map_location=device))
    model_bin.to(device)
    model_bin.eval()

    # Modèle 2 : Multiclasse (EfficientNet)
    model_multi = get_multiclass_model(num_classes=3, dropout=0.0)
    model_multi.load_state_dict(torch.load("outputs/model_etape3.pth", map_location=device))
    model_multi.to(device)
    model_multi.eval()

    return model_bin, model_multi

def analyze_mri(img_path, model_bin, model_multi, transform, device):
    print(f"\nAnalyse de l'image : {img_path}")
    
    # Préparation de l'image
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device) 

    with torch.no_grad():
        
        logits_bin = model_bin(img_tensor)

        # Vérification intelligente de la dimension de sortie du modèle
        if logits_bin.shape[1] == 1:
            score_tumeur = torch.sigmoid(logits_bin).item()
            confiance_sain = (1.0 - score_tumeur) * 100
        else:
            probs_bin = F.softmax(logits_bin, dim=1)[0]
            score_tumeur = probs_bin[1].item()
            confiance_sain = probs_bin[0].item() * 100

        # Si le modèle est sûr à moins de 50% que c'est une tumeur
        if score_tumeur < 0.5:
            print(f"Résultat : Cerveau SAIN (Confiance : {confiance_sain:.1f}%)")
            return "Sain"

        print(f"Anomalie détectée (Confiance Tumeur : {score_tumeur * 100:.1f}%)")
        print("Transfert au modèle multiclasse")
        
        logits_multi = model_multi(img_tensor)
        probs_multi = F.softmax(logits_multi, dim=1)[0]
        
        pred_idx = torch.argmax(probs_multi).item()
        score_type = probs_multi[pred_idx].item() * 100
        type_tumeur = CLASSES_ETAPE3[pred_idx]

        print(f"Diagnostic final : {type_tumeur} (Confiance : {score_type:.1f}%)")
        return type_tumeur

def main():
    # Configuration
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Optimisation matérielle Mac M1 (MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Transformations (mode "val" pour ne pas déformer l'image)
    transform = get_transforms("val", cfg["image_size"])

    # Chargement
    try:
        model_bin, model_multi = load_models(cfg, device)
    except FileNotFoundError:
        return

   
    img_test = "test.jpg"  

    if os.path.exists(img_test):
        analyze_mri(img_test, model_bin, model_multi, transform, device)
    else:
        print(f"\nFichier introuvable : '{img_test}'")

if __name__ == "__main__":
    main()