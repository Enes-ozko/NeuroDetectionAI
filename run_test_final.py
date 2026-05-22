import os
import sys
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

sys.path.insert(0, ".")
from src.data.dataset import collect_data, BrainTumorDataset
from src.data.transforms import get_transforms
from src.etape2.model import get_binary_model
from src.etape3.model import build_model as get_multiclass_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_ROOT = os.path.join(BASE_DIR, "..", "dataset", "Testing")
OUTPUTS = os.path.join(BASE_DIR, "outputs")

CLASSES_BIN   = ["Sain", "Tumeur"]
CLASSES_MULTI = ["Gliome", "Méningiome", "Pituitaire"]
REMAP         = {0: 0, 1: 1, 3: 2}

if __name__ == "__main__":
    
    if not os.path.exists(TEST_ROOT):
        print(f"ERREUR : Le dossier {TEST_ROOT} n'existe pas.")
        sys.exit(1)

    with open(os.path.join(BASE_DIR, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Chargement modèles
    model_bin = get_binary_model(dropout_p=0.0, pretrained=False)
    model_bin.load_state_dict(torch.load(os.path.join(OUTPUTS, "mobilenet_binaire.pth"), map_location=device))
    model_bin.to(device).eval()

    model_multi = get_multiclass_model(num_classes=3, dropout=0.0)
    model_multi.load_state_dict(torch.load(os.path.join(OUTPUTS, "model_etape3.pth"), map_location=device))
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
                is_tumor = scores_bin[i].item() >= 0.5
                
                bin_preds.append(int(is_tumor))
                bin_labels.append(0 if lbl_val == 2 else 1)
                
                if is_tumor and lbl_val in REMAP:
                    logits_multi = model_multi(imgs[i].unsqueeze(0))
                    pred = torch.argmax(F.softmax(logits_multi, dim=1)).item()
                    multi_preds.append(pred)
                    multi_labels.append(REMAP[lbl_val])

    print(classification_report(bin_labels, bin_preds, target_names=CLASSES_BIN))
    print(classification_report(multi_labels, multi_preds, target_names=CLASSES_MULTI))