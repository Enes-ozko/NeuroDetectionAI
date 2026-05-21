import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def get_binary_model(dropout_p: float = 0.5, pretrained: bool = True):
 
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model   = mobilenet_v3_small(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, 1),
    )

    return model