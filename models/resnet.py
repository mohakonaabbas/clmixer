from torch import nn
import torch
from typing import Union,List,Tuple
FACEBOOK_REPO="facebookresearch/RESNET"
PYTORCH_VISION_REPO="pytorch/vision"

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
class RESNET(nn.Module):

    def __init__(self,
                 name : str,
                 input_dim : Tuple[int]=(3,224,224),
                 out_dimension : int = 512):
        super().__init__()

        self.out_dim=out_dimension
        self.input_dim=input_dim
        if input_dim != (3,224,224) : 
            raise Exception("Input_dimension should be (3,224,224)")
        
        self.backbone= torch.hub.load(PYTORCH_VISION_REPO, name)
        self.backbone.fc=Identity()
       



    def forward(self, x):
        x_i=self.backbone(x) # Linear layer
        return {"attention": x_i}

class resnet18(RESNET):
    def __init__(self,
                 name : str ="resnet18",
                 input_dim : Tuple[int]=(3,224,224),
                 out_dimension : int = 512):
        
        if out_dimension != 512 : 
            raise Exception("Out_dimension should be 512")

        
        super().__init__(name=name,input_dim=input_dim,out_dimension=out_dimension)


if __name__=="__main__":
    x=torch.rand(17,3,244,244,requires_grad=True)
    

    from torchvision import transforms
    from torchvision.transforms.functional import to_pil_image
    from typing import Sequence
    import os
    import numpy as np
    import PIL
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
        
    class MaybeAddNewAxis():
        """
        Check the size of a tensor. If only 3 dimensions, add one empty  dimension in front
        """

        def __call__(self, pic):
            """
            Args:
                pic (torch.tensor): Image to be converted.
            Returns:
                Tensor: Converted image.
            """
            size=pic.size()

            if len(size)==4:
                return pic
            
            return pic[None,:,:,:]


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
            MaybeAddNewAxis()
        ]
        return transforms.Compose(transforms_list)



        

    t=make_classification_eval_transform()
    conv=resnet18()
    x=x.to('cuda')
    conv=conv.to("cuda")
    result=conv(t(x))