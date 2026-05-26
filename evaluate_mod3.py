import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix

# Import de la fonction pour construire l'architecture de ton modèle
from src.etape3.model import build_model

# 1. Configuration
TEST_DIR = "dataset/Testing"
MODEL_WEIGHTS = "outputs/model_etape3.pth"
# On exclut volontairement "notumor" pour cette évaluation
CLASSES = ["glioma", "meningioma", "pituitary"] 
BATCH_SIZE = 32

# 2. Le filtre OpenCV (Indispensable pour que le modèle retrouve ses repères)
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

# 3. Pipeline de transformation
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
test_transform = transforms.Compose([
    RemoveSkullAndFace(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# 4. Dataset sur-mesure pour le Test Set
class TestDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.paths = []
        self.labels = []
        self.transform = transform
        
        for label_idx, cls_name in enumerate(classes):
            folder = Path(root_dir) / cls_name
            if not folder.exists():
                print(f"⚠️ Attention: Le dossier {folder} est introuvable.")
                continue
            
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in folder.glob(ext):
                    self.paths.append(str(img_path))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Évaluation sur l'appareil : {device}")

    # Chargement des données
    test_dataset = TestDataset(TEST_DIR, CLASSES, transform=test_transform)
    if len(test_dataset) == 0:
        print("❌ Aucune image trouvée. Vérifie le chemin 'dataset/Testing'.")
        return
        
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"✅ Données de test chargées : {len(test_dataset)} images (3 classes).")

    # Chargement du modèle
    model = build_model(num_classes=3, dropout=0.0)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("🧠 Analyse des images en cours...")
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Calcul des métriques
    print("\n📊 RAPPORT DE CLASSIFICATION (SUR DONNÉES INCONNUES) :")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))

    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Matrice de Confusion - Test Set Inconnu", fontweight="bold")
    plt.ylabel("Vérité Terrain")
    plt.xlabel("Prédictions du Modèle")
    
    os.makedirs("outputs", exist_ok=True)
    save_path = "outputs/matrice_test_final.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n💾 Matrice de confusion sauvegardée dans : {save_path}")

if __name__ == "__main__":
    main()