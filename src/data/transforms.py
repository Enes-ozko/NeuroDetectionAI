import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

class RemoveSkullAndFace(object):
    """
    Transform sur-mesure qui utilise OpenCV pour isoler le cerveau.
    Il trouve la plus grosse masse, et peint tout le reste en noir.
    Aucun zoom n'est appliqué, aucune information centrale/périphérique n'est perdue.
    """
    def __call__(self, img):
        # 1. Convertir l'image PIL en tableau pour OpenCV
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # 2. Seuillage pour différencier le fond du patient
        # (Tout ce qui est plus lumineux que 15/255 devient blanc)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        # 3. Trouver tous les contours des objets dans l'image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sécurité : si l'image est toute noire, on ne fait rien
        if not contours:
            return img

        # 4. Garder uniquement le contour le plus vaste (le cerveau)
        largest_contour = max(contours, key=cv2.contourArea)

        # 5. Créer un masque noir, et dessiner le cerveau en blanc dessus
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # 6. Appliquer le masque : le cerveau reste, le nez/crâne disparaît dans le noir
        img_cleaned = cv2.bitwise_and(img_np, img_np, mask=mask)

        return Image.fromarray(img_cleaned)

def get_transforms(mode: str, image_size: int = 224):
    """
    Retourne les transformations avec notre nouveau filtre OpenCV intégré.
    """
    if mode == "train":
        return transforms.Compose([
            RemoveSkullAndFace(), # <-- Notre arme secrète anti-biais !
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    return transforms.Compose([
        RemoveSkullAndFace(), # Doit aussi être appliqué pour l'évaluation !
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])