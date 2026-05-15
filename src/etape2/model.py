
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def get_binary_model(dropout_p: float = 0.5):
   
    model  = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    layers = list(model.features.children())

    for param in model.parameters():
        param.requires_grad = False

    for layer in layers[-4:]:
        for param in layer.parameters():
            param.requires_grad = True

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.BatchNorm1d(in_features),   
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, 1),
    )

    return model


def freeze_bn(model: nn.Module):
  
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()