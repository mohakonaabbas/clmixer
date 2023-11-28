import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Union,List,Dict,Callable
from tqdm import tqdm
from torch.utils import data
from sklearn.model_selection import KFold
import copy
from sklearn.neighbors import BallTree

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

class BasicKernelAprroximator(nn.Module):
    def __init__(self,
                theta_refs : torch.tensor,
                theta_refs_losses : torch.tensor):
        """
        Args:
            - theta_refs : Represent the sampled parameters.
            
            - theta_refs_losses : The losses
            - h : the neighboorhood parameter
        """
        super().__init__()

        
        self.anchors=theta_refs
        self.anchors_losses=torch.squeeze(theta_refs_losses)
        self.anchors_losses=torch.tensor(self.anchors_losses)
        
        # distances_map=torch.cdist(self.anchors,self.anchors,p=2)
        # mean=torch.mean(distances_map)
        # self.h=2*mean
        # std=torch.std(torch.flatten(distances_map))


    def forward(self, theta,h=100,kernel=Epanechnikov_kernel):
        """
        This functions compute a smoothed loss function for theta
        """
        #Compute the norm of theta-anchors
        device=theta.get_device()
        if device>0:
            device="cuda:0"
        else: 
            device ="cpu"
        anchors=copy.deepcopy(self.anchors).to(device)
        anchors_losses=copy.deepcopy(self.anchors_losses).to(device)

        norms=torch.norm(anchors-theta,dim=1)/h
        K_i=kernel(norms)
        K_y_i=anchors_losses*K_i
        loss=torch.sum(K_y_i)/torch.sum(K_i)
        return loss
    

    def calibrate_h(self,
                    calibration_samples : torch.Tensor,
                    calibration_targets: torch.Tensor,
                    kernel : Callable,
                    method :str ="knn",
                    method_hyperparameters : Dict= {"min_nbrs_neigh":10},
                    search_resolution : int =100):
        """
        Compute h

        calibration_samples : Samples to uses for calibration
        calibration_targets : targets of samples
        methods : knn method  or RMSE methods
        method_hyperparameters : method hyperparameter
        
        """

        # Sample some data
        assert isinstance(calibration_samples,torch.Tensor)
        #
        distances_map=torch.cdist(self.anchors,self.anchors,p=2)
        h_max = torch.max(distances_map)

        h_range=np.linspace(1e-10,h_max,search_resolution)

        best_h=-1.0
        
        if method == "knn":
            # We want explicity N_target_neighboors to compute the loss
            # Do a ball tree 
            tree = BallTree(self.anchors, leaf_size=2) 
            N_target_neighboors=method_hyperparameters["min_nbrs_neigh"]
            avg_neighboors=[]
            for h in h_range:
                counts=tree.query_radius(calibration_targets.detach().numpy(),r=h,count_only=True)
                avg_neighboors.append(np.mean(counts))
            
            
            idx_best=np.argmax(np.abs(N_target_neighboors-np.array(avg_neighboors)))
            best_h=h_range[idx_best]


        elif method == "rmse":

            param=test_data[i,:]
            loss_approx=approximator(param,h=h,kernel=Epanechnikov_kernel)
            loss=get_parameters_loss(parameter=param,model=model,dataloader=dataloader)
            loss, _= rescaling_func(loss,hyperparameters=rescaling_func_hyperparameters)
            l_x.append(loss.item())
            l_x_hat.append(loss_approx.item())
            MAP+=torch.abs(loss_approx-loss)
            

        else:
            raise ValueError


        return best_h

    



class BasicMLPAprroximator:
    def __init__(self,
                 network,
                 epochs,
                 bs,
                 lr,
                 loader):
        """
        Args :
            epochs : Epochs
            bs: batch size
            lr:learning rate
            X: weights to regress. Size =  encoder output x Classifier outputs
            y : loss value to regress to
            criterion : regression criterion
        
        """
        
        
        loss_criterion = F.mse_loss
        network.train()
        network=network.to('cuda:0')
        loss=0.0
        optimizer = torch.optim.SGD(network.parameters(), lr=lr)
        pbar=tqdm(range(epochs))

        for epoch in pbar:
            
            for inputs,targets in loader:

                outputs=network(inputs)
                loss = loss_criterion(outputs, targets)
                pbar.set_description("%s  " % loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        self.network=network


class LandScapeModel(nn.Module):
    """
    A simple MLP wich maps all parameters from the classifier to the loss function on a certain dataset X
    """
    def __init__(self,
                input_dim : int,
                out_dimension : int = 1):
        super().__init__()

        self.input_dim=input_dim
        self.out_dim=out_dimension
        self.model=nn.Sequential(nn.Linear(self.input_dim,self.out_dim,bias=True),
                        nn.BatchNorm1d(self.out_dim),
                        nn.ReLU(),
                        nn.Dropout(p=0.1))


    def forward(self, x):
        return self.model(x)
        
