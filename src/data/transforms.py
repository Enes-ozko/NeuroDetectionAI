import torch
from torchvision import transforms


def _gaussian_noise(x, std=0.01):
    return x + std * torch.randn_like(x)


def build_transforms(cfg):
    size = cfg["img_size"]
    mean = cfg["imagenet_mean"]
    std  = cfg["imagenet_std"]

    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Lambda(_gaussian_noise),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_tf, val_tf