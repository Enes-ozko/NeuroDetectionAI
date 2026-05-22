import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, ".")
from src.data.dataset import collect_data, BrainTumorDataset
from src.data.transforms import get_transforms
from src.etape2.model import get_binary_model
from src.etape3.model import build_model as get_multiclass_model

# Configuration des classes
CLASSES_BIN   = ["Sain", "Tumeur"]
CLASSES_MULTI = ["Gliome", "Méningiome", "Pituitaire"]
REMAP         = {0: 0, 1: 1, 3: 2} # 0:glioma, 1:meningioma, 3:pituitary

if __name__ == "__main__":
    OUTPUTS = "outputs"
    os.makedirs(OUTPUTS, exist_ok=True)
    
    possible_paths = ["dataset/Testing", "../dataset/Testing", "Testing"]
    TEST_ROOT = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if not TEST_ROOT:
        print(f"ERREUR : Dossier de test introuvable.")
        sys.exit(1)
        
    print(f"Dossier Testing trouvé : {os.path.abspath(TEST_ROOT)}")

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Appareil
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Chargement modèles
    model_bin = get_binary_model(dropout_p=0.0, pretrained=False)
    model_bin.load_state_dict(torch.load(f"{OUTPUTS}/mobilenet_binaire.pth", map_location=device))
    model_bin.to(device).eval()

    model_multi = get_multiclass_model(num_classes=3, dropout=0.0)
    model_multi.load_state_dict(torch.load(f"{OUTPUTS}/model_etape3.pth", map_location=device))
    model_multi.to(device).eval()

    # Chargement Data
    paths, labels = collect_data(root=TEST_ROOT, classes=cfg["classes"], samples_per_class=None, seed=cfg["seed"])
    print(f"Images chargées : {len(paths)}")
    
    test_ds = BrainTumorDataset(paths, labels, transform=get_transforms("val", cfg["image_size"]), task="multiclass")
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    # Inférence
    bin_preds, bin_labels = [], []
    multi_preds, multi_labels = [], []

    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            logits_bin = model_bin(imgs)
            scores_bin = torch.sigmoid(logits_bin).squeeze(1) if logits_bin.shape[1] == 1 else F.softmax(logits_bin, dim=1)[:, 1]
            
            for i in range(len(imgs)):
                lbl_val = lbls[i].item()
                score = scores_bin[i].item()
                is_tumor = score >= 0.5
                
                # Binaire
                bin_preds.append(int(is_tumor))
                bin_labels.append(0 if lbl_val == 2 else 1)
                
                # Multiclasse 
                if is_tumor and lbl_val in REMAP:
                    logits_multi = model_multi(imgs[i].unsqueeze(0))
                    pred = torch.argmax(F.softmax(logits_multi, dim=1)).item()
                    multi_preds.append(pred)
                    multi_labels.append(REMAP[lbl_val])

    # 6. Rapports
    print("Binaire :")
    print(classification_report(bin_labels, bin_preds, target_names=CLASSES_BIN))
    print("Multiclasse :")
    print(classification_report(multi_labels, multi_preds, target_names=CLASSES_MULTI))