import timm
import torch.nn as nn

__all__ = [
    "vit16",
]


def vit16(pretrained: bool = False, num_classes: int = 1000) -> nn.Module:
    """ViT-Base (ViT-B/16) from original paper.

    This function just call function in timm repo https://github.com/rwightman/pytorch-image-models (Apache License 2.0).

    Note:
        The main params to create Vision Transformer is shown in below.
        - patch_size = 16
        - embed_dim = 768
        - depth = 12
        - num_heads = 12

    Args:
        pretrained (bool, optional): If pretrained is True, this function downloads and loads pretrained model of ImageNet-1k.
        num_classes (int, optional): The number of class.

    Returns:
        nn.Module: Vision Transformer model.

    """
    return timm.create_model(
        "vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes
    )
