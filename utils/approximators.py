import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Union,List
from tqdm import tqdm
from torch.utils import data
from sklearn.model_selection import KFold


def Epanechnikov_kernel(u:torch.tensor):
    """
    https://fr.wikipedia.org/wiki/Noyau_(statistiques)
    """
    mask=u<1
    K=3/4*(1-u**2)
    K=K*mask
    return K

def IdentityKernel(u:torch.tensor):
    return u

def TriangleKernel(u:torch.tensor):
    return 1-torch.abs(u)

class BasicKernelAprroximator:
    def __init__(self,
                theta_refs : torch.tensor,
                theta_refs_losses : torch.tensor):
        """
        Args:
            - theta_refs : Represent the sampled parameters.
            
            - theta_refs_losses : The losses
            - h : the neighboorhood parameter
        """

        
        self.anchors=theta_refs
        self.anchors_losses=torch.squeeze(theta_refs_losses)
        
        # distances_map=torch.cdist(self.anchors,self.anchors,p=2)
        # mean=torch.mean(distances_map)
        # self.h=2*mean
        # std=torch.std(torch.flatten(distances_map))


    def evaluate(self, theta,h,kernel=Epanechnikov_kernel):
        """
        This functions compute a smoothed loss function for theta
        """
        #Compute the norm of theta-anchors

        norms=torch.norm(self.anchors-theta,dim=1)/h
        K_i=kernel(norms)
        K_y_i=self.anchors_losses*K_i
        loss=torch.sum(K_y_i)/torch.sum(K_i)
        return loss

    



