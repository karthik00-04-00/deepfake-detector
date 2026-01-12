# DEBUG_TOUCH_CONVNEXT

"""
src/model.py

Model factory for image-based deepfake classification.
Supports:
- ResNet18
- EfficientNet-B0
- ConvNeXt-Tiny

Backward-compatible with Phase 2 training code.
"""

import torch.nn as nn
from torchvision import models


# -------------------------
# ResNet18
# -------------------------
def get_resnet18(
    num_classes=2,
    pretrained=True,
    finetune_layer4=False,
):
    model = models.resnet18(pretrained=pretrained)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    for p in model.parameters():
        p.requires_grad = False

    for p in model.fc.parameters():
        p.requires_grad = True

    if finetune_layer4:
        for p in model.layer4.parameters():
            p.requires_grad = True

    return model


# -------------------------
# EfficientNet-B0
# -------------------------
def get_efficientnet_b0(
    num_classes=2,
    pretrained=True,
    finetune_top=False,
):
    model = models.efficientnet_b0(pretrained=pretrained)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    for p in model.parameters():
        p.requires_grad = False

    for p in model.classifier.parameters():
        p.requires_grad = True

    if finetune_top:
        # last meaningful MBConv block
        for p in model.features[-2].parameters():
            p.requires_grad = True

    return model


# -------------------------
# ConvNeXt-Tiny
# -------------------------
def get_convnext_tiny(
    num_classes=2,
    pretrained=True,
    finetune_stage=False,
):
    model = models.convnext_tiny(pretrained=pretrained)

    # Replace classifier
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Train classifier
    for p in model.classifier.parameters():
        p.requires_grad = True

    # Optional: fine-tune last ConvNeXt stage
    if finetune_stage:
        for p in model.features[-1].parameters():
            p.requires_grad = True

    return model


# -------------------------
# Model factory
# -------------------------
def get_model(cfg=None, **kwargs):
    """
    Backward-compatible model factory.

    Phase 2:
        get_model(num_classes=..., pretrained=..., finetune_layer4=...)

    Phase 3+:
        get_model(cfg)
    """

    # Phase 2 legacy API
    if cfg is None:
        return get_resnet18(
            num_classes=kwargs.get("num_classes", 2),
            pretrained=kwargs.get("pretrained", True),
            finetune_layer4=kwargs.get("finetune_layer4", False),
        )

    model_name = cfg["model"]["name"]

    if model_name == "resnet18":
        return get_resnet18(
            num_classes=cfg["model"].get("num_classes", 2),
            pretrained=cfg["model"].get("pretrained", True),
            finetune_layer4=cfg["model"].get("finetune_layer4", False),
        )

    if model_name == "efficientnet_b0":
        return get_efficientnet_b0(
            num_classes=cfg["model"].get("num_classes", 2),
            pretrained=cfg["model"].get("pretrained", True),
            finetune_top=cfg["model"].get("finetune_top", False),
        )

    if model_name == "convnext_tiny":
        return get_convnext_tiny(
            num_classes=cfg["model"].get("num_classes", 2),
            pretrained=cfg["model"].get("pretrained", True),
            finetune_stage=cfg["model"].get("finetune_stage", False),
        )

    raise ValueError(f"Unknown model name: {model_name}")
