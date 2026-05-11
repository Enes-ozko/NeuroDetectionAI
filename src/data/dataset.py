import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class BrainTumorDataset(Dataset):

    def __init__(self, root, indices=None, transform=None, max_per_class=None, seed=42):
        self.root = Path(root)
        self.transform = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(CLASS_NAMES)}
        self.samples = []

        rng = random.Random(seed)

        for cls_name in CLASS_NAMES:
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue

            imgs = sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])
            rng.shuffle(imgs)

            if max_per_class is not None:
                imgs = imgs[:max_per_class]

            for path in imgs:
                self.samples.append((path, self.class_to_idx[cls_name]))

        if not self.samples:
            raise RuntimeError(f"aucune image trouvée dans {self.root.resolve()}")

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def labels(self):
        return [label for _, label in self.samples]