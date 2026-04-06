import torch.nn as nn
from torchvision import models


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
    elif name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
    elif name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
    elif name == "squeezenet1_0":
        weights = models.SqueezeNet1_0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.squeezenet1_0(weights=weights)
    elif name == "shufflenet_v2_x1_0":
        weights = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.shufflenet_v2_x1_0(weights=weights)
    elif name == "mnasnet1_0":
        weights = models.MNASNet1_0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mnasnet1_0(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {name}")
    model.eval()
    return model


def is_supported_activation_parent(module: nn.Module) -> bool:
    return isinstance(module, (nn.Conv2d, nn.Linear))
