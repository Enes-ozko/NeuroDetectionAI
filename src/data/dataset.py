import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class BrainTumorDataset(Dataset):

    def __init__(self, paths: list, labels: list, transform=None, task="multiclass"):
        self.paths     = paths
        self.transform = transform
        self.task      = task

        if task == "binary":
            self.labels = [0 if l == 2 else 1 for l in labels]
        else:
            self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def collect_data(root: str, classes: list, samples_per_class: int = None, seed: int = 42):

    paths, labels = [], []

    for i, cls in enumerate(classes):
        folder = Path(root) / cls
        class_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            class_files.extend(folder.glob(ext))

        class_files = sorted(class_files)

        if samples_per_class is not None:
            random.seed(seed)
            class_files = random.sample(class_files, min(samples_per_class, len(class_files)))

        for p in class_files:
            paths.append(str(p))
            labels.append(i)

    combined = list(zip(paths, labels))
    random.seed(seed)
    random.shuffle(combined)
    paths, labels = zip(*combined)

    return list(paths), list(labels)