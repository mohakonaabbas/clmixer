import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18,resnet152,resnet50
import torch.nn as nn
import json

from segment_anything import sam_model_registry, SamPredictor

from torchvision import transforms
from typing import Sequence


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)
    


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)

class Dinov2:
    """
    Load the dinov2 backbone
    """
    def __init__(backbone_name,
                 transformations):
        """
        backbone_name : The kind of backbone we want to load
        transform : The king of tranform to apply to the data
        """
        local_path="./datasets/models/sam/sam_vit_b_01ec64.pth"
        self.backbone= torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        transforms_list=[
            MaybeToTensor(),
            make_normalize_transform(),
        ]
        
        self.transformations=transforms.Compose(transforms_list)

        dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

        pass





    def predict():


        pass
    
    