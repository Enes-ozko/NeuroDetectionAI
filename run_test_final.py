import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
)

sys.path.insert(0, ".")
from src.data.dataset import collect_data, BrainTumorDataset
from src.data.transforms import get_transforms
from src.etape2.model import get_binary_model
from src.etape3.model import build_model as get_multiclass_model

CLASSES_4     = ["glioma", "meningioma", "notumor", "pituitary"]
CLASSES_BIN   = ["Sain", "Tumeur"]
CLASSES_MULTI = ["Gliome", "Méningiome", "Pituitaire"]
REMAP         = {0: 0, 1: 1, 3: 2}

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Appareil : {device}")

    # Chargement des modèles 
    print("\nChargement des modèles")
    model_bin = get_binary_model(dropout_p=0.0, pretrained=False)
    model_bin.load_state_dict(torch.load("outputs/mobilenet_binaire.pth", map_location=device))
    model_bin.to(device).eval()

    model_multi = get_multiclass_model(num_classes=3, dropout=0.0)
    model_multi.load_state_dict(torch.load("outputs/model_etape3.pth", map_location=device))
    model_multi.to(device).eval()
    print("Modèles chargés")

    # Chargement du dossier Testing 
    test_root = cfg["dataset_root"].replace("Training", "Testing")
    print(f"\nDossier Testing : {test_root}")

    paths, labels = collect_data(
        root=test_root,
        classes=cfg["classes"],
        samples_per_class=None,
        seed=cfg["seed"],
    )
    print(f"Total images : {len(paths)}")
    for i, cls in enumerate(CLASSES_4):
        print(f"  {cls:12s} : {labels.count(i)} images")

    transform   = get_transforms("val", cfg["image_size"])
    test_ds     = BrainTumorDataset(paths, labels, transform=transform, task="multiclass")
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    # Évaluation binaire 
    bin_preds, bin_labels, bin_scores = [], [], []

    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs   = imgs.to(device)
            logits = model_bin(imgs)
            if logits.shape[1] == 1:
                scores = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                preds  = (scores >= 0.5).astype(int)
            else:
                probs  = F.softmax(logits, dim=1).cpu().numpy()
                scores = probs[:, 1]
                preds  = probs.argmax(axis=1)
            bin_preds.extend(preds)
            bin_labels.extend([0 if l == 2 else 1 for l in lbls.numpy()])
            bin_scores.extend(scores)

    acc2 = np.mean(np.array(bin_preds) == np.array(bin_labels))
    auc2 = roc_auc_score(bin_labels, bin_scores)
    print(f"Accuracy : {acc2:.4f} | AUC : {auc2:.4f}")
    print(classification_report(bin_labels, bin_preds, target_names=CLASSES_BIN))

    cm2 = confusion_matrix(bin_labels, bin_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm2, display_labels=CLASSES_BIN).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Binaire - Acc={acc2:.3f}  AUC={auc2:.3f}")
    plt.tight_layout()
    plt.savefig("outputs/test_etape2_confusion.png", dpi=150)
    plt.close()
    print("outputs/test_etape2_confusion.png")

    # Évaluation multiclasse 
    multi_preds, multi_labels = [], []

    with torch.no_grad():
        for imgs, lbls in test_loader:
            for i in range(len(lbls)):
                lbl = lbls[i].item()
                if lbl == 2:          
                    continue
                img_t  = imgs[i].unsqueeze(0).to(device)
                logits = model_bin(img_t)
                score  = (torch.sigmoid(logits).item()
                          if logits.shape[1] == 1
                          else F.softmax(logits, dim=1)[0][1].item())
                if score >= 0.5:
                    pred = torch.argmax(F.softmax(model_multi(img_t), dim=1)).item()
                else:
                    pred = 0          
                multi_preds.append(pred)
                multi_labels.append(REMAP[lbl])

    acc3 = np.mean(np.array(multi_preds) == np.array(multi_labels))
    print(f"Accuracy : {acc3:.4f}")
    print(classification_report(multi_labels, multi_preds, target_names=CLASSES_MULTI))

    cm3 = confusion_matrix(multi_labels, multi_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm3, display_labels=CLASSES_MULTI).plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title(f"Multiclasse - Acc={acc3:.3f}")
    plt.tight_layout()
    plt.savefig("outputs/test_etape3_confusion.png", dpi=150)
    plt.close()
    print("outputs/test_etape3_confusion.png")

