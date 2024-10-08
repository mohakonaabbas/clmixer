import torch
from torch import nn
from torch import optim

from models import resnet18, dinov2_vitb14,dinov2_vitl14,dinov2_vits14,MLP,Identity,MLPLora, MLPLoraSubspace,MLPSubspace
# from my_resnet import resnet_rebuffi
# from mlp import MLP
# from mlp_sam import MLP_SAM


def get_optimizer(params, optimizer, lr, weight_decay=0.0):
    if optimizer == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    elif optimizer == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.0)
    else:
        raise NotImplementedError


def get_net(net_type, **kwargs):
    if net_type == "resnet18":
        return resnet18()
    elif net_type == "mlp":
        return MLP(**kwargs)
    elif net_type == "mlp_lora":
        return MLPLora(**kwargs)
    elif net_type =="mlp_subspace_lora":
        return MLPLoraSubspace(**kwargs)
    elif net_type =="mlp_subspace":
        return MLPSubspace(**kwargs)
    elif net_type == "dinov2_vitb14":
        return dinov2_vitb14()
    elif net_type == "dinov2_vitl14":
        return dinov2_vitl14()
    elif net_type == "dinov2_vits14":
        return dinov2_vits14()
    elif net_type == "identity":
        return Identity(**kwargs)




def set_device(cfg):
    device_type = cfg["device"]

    if device_type == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(device_type))

    cfg["device"] = device
    return device

