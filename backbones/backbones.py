import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import json

from segment_anything import sam_model_registry, SamPredictor

from torchvision import transforms,models
from torchvision.transforms.functional import to_pil_image
from typing import Sequence
import os
import numpy as np
import PIL


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
DEFAULT_BACKBONES_PATH="./backbones/pretrained/"
FACEBOOK_REPO="facebookresearch/dinov2"
PYTORCH_VISION_REPO="pytorch/vision"

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

class MaybeToPIL():
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
        if isinstance(pic, PIL.Image.Image) or isinstance(pic, torch.Tensor):
            return pic
        return to_pil_image(pic)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)

def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        MaybeToPIL(),
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)


class baseBackbone:
    def __init__(self,
                 backone_name,
                 device):
        raise NotImplementedError
    
    def predict(self,X):
        raise NotImplementedError
class Dinov2(baseBackbone):
    """
    Load the dinov2 backbone
    """
    def __init__(self,
                 backbone_name : str ="dinov2_vits14",
                 device="cpu"
                 ):
        """
        backbone_name : The kind of backbone we want to load
        transform : The king of tranform to apply to the data
        """
        local_path=os.path.join(DEFAULT_BACKBONES_PATH,"dinov2",backbone_name)+".pth"

        # if os.path.exists(local_path):
        #     self.backbone=torch.load(local_path)
        # else:
        try:
            name=backbone_name
            self.backbone= torch.hub.load(FACEBOOK_REPO, name)
        except :
            raise Exception('The model type does not exist ! ') 
        self.transformations=make_classification_eval_transform()
        
        

        self.backbone.eval
        self.backbone.to(device)


    def predict(self,
                X: np.array):
        """
        Predict the image
        """
        x=self.transformations(X)
        x=x.to(self.device)
        with torch.no_grad():
            x=self.backbone(x)
        
        return x
    

class Resnet(baseBackbone):
    def __init__(self,
                 backbone_name : str ="dinov2_vits14",
                 device="cpu"
                 ):
        """
        backbone_name : The kind of backbone we want to load
        transform : The king of tranform to apply to the data

        resnet101'
        61:
        'resnet152'
        62:
        'resnet18'
        63:
        'resnet34'
        64:
        'resnet50'
        """
        self.device=device
        # if os.path.exists(local_path):
        #     self.backbone=torch.load(local_path)
        # else:
        try:
            name=backbone_name
            self.backbone= torch.hub.load(PYTORCH_VISION_REPO, name)
        except :
            raise Exception('The model type does not exist ! ')
        
        self.backbone.fc=Identity()


        transformations=[MaybeToPIL(), ResNet18_Weights.DEFAULT.transforms()]
        self.transformations=transforms.Compose(transformations)


        self.backbone.eval
        self.backbone.to(device)


    def predict(self,
                X: np.array):
        """
        Predict the image
        """
        x=self.transformations(X)
        x=x.to(self.device)
        with torch.no_grad():
            x=self.backbone(x)
        
        return x

    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


if __name__=="__main__":
    backbone=Resnet(backbone_name="resnet18",device="cuda")
    im=torch.rand(size=(2,3,224,224))
    pred=backbone.predict(im)
    pred




        
    
    