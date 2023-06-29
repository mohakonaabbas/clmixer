import torch
from torch import nn
from torch import optim


from my_resnet import resnet_rebuffi
from mlp import MLP
from mlp_sam import MLP_SAM


def get_optimizer(params, optimizer, lr, weight_decay=0.0):
    if optimizer == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    elif optimizer == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise NotImplementedError


def get_net(net_type, **kwargs):
    if net_type == "convnet":
        return resnet_rebuffi(**kwargs)
    elif net_type == "mlp_sam":
        return MLP_SAM(**kwargs)
    elif net_type == "mlp":
        return MLP(**kwargs)




def set_device(cfg):
    device_type = cfg["device"]

    if device_type == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(device_type))

    cfg["device"] = device
    return device
