from torchvision import transforms

# Moyenne et écart-type ImageNet (obligatoire pour les modèles pré-entraînés)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_transforms(mode: str, image_size: int = 224):
    """
    Retourne les transformations selon le mode :
      - "train" : augmentations aléatoires pour éviter le surapprentissage
      - autre   : redimensionnement + normalisation uniquement (val/test)
    """
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])