"""
src/model.py

Provides a small factory for a ResNet18-based binary classifier.
Function: get_model(num_classes=2, pretrained=True)
Returns a torch.nn.Module ready for training.
"""
from typing import Optional
import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes: int = 2, pretrained: bool = True, dropout: Optional[float] = None) -> nn.Module:
    """
    Create a ResNet18 model with the final layer replaced for `num_classes`.

    Args:
        num_classes: number of output classes (2 for real/fake).
        pretrained: if True, load ImageNet pretrained weights.
        dropout: optional dropout probability inserted before the final fc.
    Returns:
        model (nn.Module)
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) if pretrained else models.resnet18(weights=None)

    # grab number of features from the original fc
    in_features = model.fc.in_features

    modules = []
    if dropout is not None and dropout > 0.0:
        modules.append(nn.Dropout(p=dropout))
    modules.append(nn.Linear(in_features, num_classes))

    model.fc = nn.Sequential(*modules)

    return model


if __name__ == "__main__":
    # quick smoke test
    m = get_model()
    x = torch.randn(2, 3, 224, 224)
    logits = m(x)   
    print("Model output shape:", logits.shape)  # expect: (2, 2)
