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
from .sampling import EfficientSampler, identity, box_cox, normal , minmax

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
    mask=np.abs(u)<1
    K=1-u
    K=K*mask

    return K

class BasicKernelAprroximator(nn.Module):
    def __init__(self,
                theta_refs : torch.Tensor,
                theta_refs_raw_losses : torch.Tensor,
                theta_refs_losses : torch.Tensor):
        """
        Args:
            - theta_refs : Represent the sampled parameters.
            
            - theta_refs_losses : The losses
            - h : the neighboorhood parameter
        """
        super().__init__()

        
        self.anchors=theta_refs # X

        self.anchors_losses=torch.squeeze(theta_refs_losses) # Y
        self.anchors_losses=torch.tensor(self.anchors_losses)

        self.anchors_raw_losses=torch.squeeze(theta_refs_raw_losses)
        self.anchors_raw_losses=torch.tensor(self.anchors_raw_losses)

        self.h=None
        self.rescaling_parameters=None
        
        self.anchors_distances_map=torch.cdist(self.anchors,self.anchors,p=2.0)

        # Get distances normalisation 
        mask_distance=self.anchors_distances_map >1e-2
        min_distance=torch.min(self.anchors_distances_map[mask_distance]).item()
        max_distance=torch.max(self.anchors_distances_map[mask_distance]).item()

        def scaler(x):
            """
            Scale the normss sto be close to the real topography of the space

            """

            if not isinstance(x,torch.Tensor):
                x=torch.tensor(x,dtype=torch.float32)

            mask_min=torch.abs(x)<=min_distance
            mask_max=torch.abs(x)>=max_distance
            mask_middle=torch.bitwise_not(mask_min+mask_max)


            x[mask_min]=0.0
            x[mask_max]=1.0
            x[mask_middle]=(x[mask_middle]-min_distance)/(max_distance-min_distance)
            return x


        
        self.distance_scaler=scaler


        # self.h=2*mean
        # std=torch.std(torch.flatten(distances_map))


    def forward(self, theta,h=-1,kernel=Epanechnikov_kernel):
        """
        This functions compute a smoothed loss function for theta
        """
        if h<0:
            assert self.h is not None
            assert self.rescaling_parameters is not None
        h_=h if h>0 else self.h
        #Compute the norm of theta-anchors
        device=theta.get_device()
        if device>0:
            device="cuda:0"
        else: 
            device ="cpu"
        anchors=copy.deepcopy(self.anchors).to(device)
        anchors_losses=copy.deepcopy(self.anchors_losses).to(device)

        w_shape=theta.shape

        w_theta=torch.reshape(theta,(w_shape[0],1,w_shape[1]))

        norms=torch.norm(anchors-w_theta,dim=2)

        norms=self.distance_scaler(norms)/self.distance_scaler(h_)

        # norms=self.distance_scaler(norms)
        K_i=kernel(norms)
        K_y_i=anchors_losses.reshape(1,-1)*K_i
        loss=torch.sum(K_y_i,dim=1)/(torch.sum(K_i,dim=1))
        loss=torch.nan_to_num(loss,nan=float("inf"))
        # loss=torch.nan_to_num(loss,nan=3.0)
        return loss
    

    def calibrate_h(self,
                    calibration_samples : torch.Tensor,
                    calibration_targets: torch.Tensor,
                    method :str ="knn",
                    method_hyperparameters : Dict= {"min_nbrs_neigh":10,"kernel":Epanechnikov_kernel},
                    search_resolution : int =100):
        """
        Compute an optimal h

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

        if method == "knn":
            # We want explicity N_target_neighboors to compute the loss
            # Do a ball tree 
            tree = BallTree(self.anchors, leaf_size=2) 
            N_target_neighboors=method_hyperparameters["min_nbrs_neigh"]
            avg_neighboors=[]
            for h in h_range:
                counts=tree.query_radius(calibration_samples.detach().numpy(),r=h,count_only=True)
                avg_neighboors.append(np.min(counts))
            
            
            idx_best=np.argmin(np.abs(N_target_neighboors-np.array(avg_neighboors)))
            best_h=h_range[idx_best]


        elif method == "rmse":
            kernel=method_hyperparameters["kernel"]
            avg_rmse=[]

            for h in h_range:
                pred=self.forward(calibration_samples,h,kernel=kernel)
                errrors=(pred-calibration_targets)**2
                mse=torch.median(errrors)
                rmse=torch.sqrt(mse)
                avg_rmse.append(rmse)
            idx_best=torch.argmin(torch.tensor(avg_rmse)).item()
            best_h=h_range[idx_best]

        else:
            raise ValueError

        self.h=best_h

        return best_h
    

    def active_augment_anchors(self,
                               maybe_anchors : torch.Tensor ,
                               maybe_anchors_losses: torch.Tensor,
                               maybe_anchors_preds : torch.Tensor,
                               maybe_anchors_raw_losses : torch.Tensor):
        """
        This function augments the anchors with the anchors that has been computed
        """
        print(f"Old shape of anchors : {self.anchors.shape}")

        indices=torch.argwhere(maybe_anchors_preds==float("inf"))
        indices=torch.squeeze(indices)
        
        if len(indices.size())>0:
            to_add_anchors=maybe_anchors[indices,:]
            to_add_anchors_losses=torch.squeeze(maybe_anchors_losses[indices,:])
            to_add_anchors_raw_losses=torch.squeeze(maybe_anchors_raw_losses[indices,:])

            self.anchors=torch.concatenate([self.anchors,to_add_anchors])
            self.anchors_losses=torch.concatenate([self.anchors_losses,to_add_anchors_losses])
            self.anchors_raw_losses=torch.concatenate([self.anchors_raw_losses,to_add_anchors_raw_losses])

            self.anchors=torch.tensor(self.anchors)
            self.anchors_losses=torch.tensor(self.anchors_losses)
            self.anchors_raw_losses=torch.tensor(self.anchors_raw_losses)

        print(f"New shape of anchors : {self.anchors.shape}")

        return indices
    
    def set_rescaling_parameters(self,callable:Callable,parameters:Dict):
        self.rescaling_parameters={"func":callable,"params":parameters}
    

def active_anchors_choice(approximator : BasicKernelAprroximator,
                          approximator_hyperparameters : Dict,
                          sampler :EfficientSampler,
                          sampler_hyperparameters : Dict,
                          n_anchors_max : int = 10000,
                          n_rounds_improve : int=10,
                          n_targets_neighboors : int = 5) -> BasicKernelAprroximator:
    """
    THis function implement an active learnng strategy to sample useful anchor points in an efficient way
    """

    n_qanta=n_anchors_max//n_rounds_improve
    n_h=100

    # Init phase


    # Define the sampler callback
    random_sampler_callback = sampler_hyperparameters["random_sampler_callback"]
    rescaling_func= sampler_hyperparameters["rescaling_func"]
    rescaling_func_hyperparameters=sampler_hyperparameters["rescaling_func_hyperparameters"]
    kernel=approximator_hyperparameters["kernel"]



    for i in range(n_rounds_improve):
        # Sample data to test

        

        results=sampler.sample(n_qanta,callback=random_sampler_callback,
                            rescaling_func= rescaling_func,
                            rescaling_func_hyperparameters=rescaling_func_hyperparameters)

        
        test_samples=results["parameters"]
        test_targets=results["rescaled_losses"]
        true_losses=results["losses"]

        test_predictions= approximator(test_samples,kernel=kernel)
        # Insert the one that had no neighboors

        indices=approximator.active_augment_anchors(maybe_anchors=test_samples,
                                            maybe_anchors_losses=test_targets,
                                            maybe_anchors_preds=test_predictions,
                                            maybe_anchors_raw_losses=true_losses)
        

        

        # Recompute the metrics for prediction
        _,rescaling_func_hyperparameters=rescaling_func(approximator.anchors_raw_losses,{"mean":None,"std":None,"lambda":None,"min":None,"max":None})
        


        #Calibrate this kernel h
        
        results=sampler.sample(n_h,callback=random_sampler_callback,
                            rescaling_func= rescaling_func,
                            rescaling_func_hyperparameters=rescaling_func_hyperparameters)

        calibration_samples=results["parameters"]
        calibration_targets=results["rescaled_losses"]


        h=approximator.calibrate_h(calibration_samples= calibration_samples , 
                                calibration_targets= calibration_targets, 
                                method= "knn", 
                                method_hyperparameters={"min_nbrs_neigh":n_targets_neighboors,"kernel":kernel})
        
    approximator.set_rescaling_parameters(rescaling_func,parameters=rescaling_func_hyperparameters)

    return approximator
        



    



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
        
