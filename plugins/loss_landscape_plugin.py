
from .base_plugin import Operation
from torch.nn import functional as F
import copy
import torch

from typing import List, Tuple
from sklearn.utils import resample

import numpy as np
from datasets import base
from utils import sampling,approximators
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
global counter_writer
counter_writer =0
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from torch import nn
from scipy.stats import qmc
from typing import Dict, List
from tqdm import tqdm

class Projector(nn.Module):
    def __init__(self, 
                 reference_point : torch.Tensor,
                d_hyperplane : torch.Tensor,
                mean : torch.Tensor,
                std : torch.Tensor) -> None:
        """
        reference_point : The centered point in full space D
        d_hyperplane : Plane of projection in Dxd space d<<D

        
        """
        super().__init__()
        self.reference_point=nn.parameter.Parameter(reference_point,requires_grad=False)
        self.mean=nn.parameter.Parameter(mean,requires_grad=False)
        self.std=nn.parameter.Parameter(std,requires_grad=False)
        self.hyperplane=nn.parameter.Parameter(d_hyperplane,requires_grad=False)
        self.d=self.hyperplane.shape[1]

    def forward(self,parameter:torch.Tensor):
        """
        1. Project a parameter in full space D to small d space
        """
        d_parameter= parameter-self.reference_point
        d_parameter-=self.mean
        d_parameter/=self.std
        projected=torch.matmul(d_parameter,self.hyperplane).squeeze()
        return projected
    
    def unproject(self,parameter:torch.Tensor):
        """
        1. Project a parameter in d space D to  D space
        """

        reprojected=self.mean+self.std*(torch.matmul(parameter,self.hyperplane.T).squeeze())+self.reference_point
        return reprojected

    
class subspaceClassifierModel(nn.Module):
    def __init__(self,
                 projector: Projector,
                current_model_checkpoint: nn.Module) -> None:
        """
        reference_point : The centered point in full space D
        d_hyperplane : Plane of projection in Dxd space d<<D
        current_model_checkpoint : true neural network architecture
        
        """
        super().__init__()
        #Get the current model architecture
        self.model_arch=copy.deepcopy(current_model_checkpoint).freeze()
        parameter=self.flatten_architecture().reshape(1,-1)
        
        self.projector=projector
        weight_init=self.projector(parameter).squeeze()
        self.weight=nn.parameter.Parameter(weight_init,requires_grad=True)


    def forward(self,input:torch.Tensor):
        """
        1. Rebuild the actual network
        2. Querry the model
        """

        #Iinsert the parameters in the model
        
        parameter=self.projector.unproject(self.weight)
        shape=(int(parameter.shape[-1]/input.shape[1]),input.shape[1])
        parameter=parameter.reshape(shape)
        out = F.linear(input, parameter)
        # output=torch.matmul(input,parameter.T)
        # parameter=nn.Parameter(parameter.clone(),requires_grad=True)
        # self.rebuild_architecture(parameter=parameter)
        # output=self.model_arch(x)
        return out

    def flatten_architecture(self):
        parameter=[]
        start=0
        with torch.no_grad():
            for name,param in self.model_arch.named_parameters():
                if ("weight" in name) or ("bias"  in name):
                    parameter.append(torch.flatten(param))
            parameter=torch.cat(parameter)
        return parameter.data
    
    def rebuild_architecture(self,parameter:torch.Tensor):
        """
        This function does not allow to backpropagate the gradient
        We need to manually create the network to run the true forward
        /!\ DO NOT USE IT
        """
        start=0
        for name,param in self.model_arch.named_parameters():
            if ("weight" in name) or ("bias" in name):
                shape=param.shape
                flat_shape=np.prod(param.shape)
                end=start+flat_shape
                param.data=parameter[:,start:start+flat_shape].reshape(shape)
                param.requires_grad = True
                start=end



class LossLandscapeOperation(Operation):
# class LossLandscapeOperation():
    def __init__(self,
                entry_point =["before_backward","before_training_exp","after_training_exp","after_training_epoch"],
                inputs={},
            callback=(lambda x:x), 
            paper_ref="",
                is_loss=True):
        super().__init__(entry_point, inputs, callback, paper_ref,is_loss)

        self.set_callback(self.LossApprox_callback)
        self.set_config_template({
      "name": self.__class__.__name__,
      "hyperparameters": {
        "approximators": [],
        "current_task_loss_dataset":{
                "weights": [],
                "losses":[]
            },
        "n":10**4,
        "d":2
        # "bs":32,
        # "lr":1e-3

      },
      "function": "knowledge_retention"
    })
        self.device=None
        self.past_anchors=None
        self.past_anchors_raw_losses=None
        self.old_dataloaders=[]

        
        #Dimensionality reduction

        # Projector
        self.projector=None
        # History dictionary
        self.History={"initial_trajectory":{"parameters":[],"losses":[]},
                      "dataloaders":[],
                      "approximators":[],
                      "current_trajectory":{"parameters":[],"losses":[]}}


        #Classifier
        self.classifier=None

    def compute_projector(self,
                          d : int,
                          training_trajectory: Dict):
        """
        This methods create a projection from the original dimension of network parameters to d dimension
        Using PLS, we restrict to the d first axis the projection.
        Args
        - d : int ,the target dimension
        - training_trajectory, a Dict = {"parameters" : parameters : torch.Tensor} representing the trajectory of training
        
        """

        pls = PLSRegression(n_components=d)
        parameters=torch.concatenate(training_trajectory["parameters"]).cpu().numpy()
        losses=torch.stack(training_trajectory["losses"]).cpu().numpy()
        reference_point=training_trajectory["parameters"][-1]
        pls.fit(parameters-reference_point.cpu().numpy(),losses)

        d_hyperplane=pls.x_rotations_
        d_hyperplane=torch.tensor(d_hyperplane,dtype=torch.float32)
        mean=torch.tensor(pls._x_mean,dtype=torch.float32)
        std=torch.tensor(pls._x_std,dtype=torch.float32)

        

        projector=Projector(reference_point=reference_point,d_hyperplane=d_hyperplane,mean=mean,std=std)
        return projector

    def compute_subspace_classifier(self,
                                    projector: Projector,
                                    current_model : nn.Module):
        """
        This methods create a model compatible with the plugin
        """
        classifier= subspaceClassifierModel(projector= self.projector,
                                            current_model_checkpoint=current_model)
        
        return classifier

    def latin_hypercube_sampling(self,
                n : int =1000,
                coarse_limit :int =100,
                fine_limit : int=1):
        
        projector=self.projector
        Phi  = copy.deepcopy(self.inputs.current_network)
        dataloader = self.History["dataloaders"][-1]
        

        if projector.d>2:
            #Create a grid of points uniformely spaced in d-space
            sampler=qmc.LatinHypercube(d=D)
            coeffs=sampler.random(n)
            directions=torch.tensor(coeffs,dtype=torch.float32)
        elif projector.d==2:
            #Coarse
            n_steps=int(np.sqrt(n))
            xs = torch.linspace(-coarse_limit, coarse_limit, steps=n_steps).flatten()
            ys = torch.linspace(-coarse_limit, coarse_limit, steps=n_steps).flatten()
            xs,ys=torch.meshgrid(xs,ys)
            directions_c=torch.concatenate((xs.reshape(xs.shape+(1,)),ys.reshape(ys.shape+(1,))),dim=2)
            #Fine
            xs = torch.linspace(-fine_limit, fine_limit, steps=n_steps).flatten()
            ys = torch.linspace(-fine_limit, fine_limit, steps=n_steps).flatten()
            xs,ys=torch.meshgrid(xs,ys)
            directions_f=torch.concatenate((xs.reshape(xs.shape+(1,)),ys.reshape(ys.shape+(1,))),dim=2)
            directions=torch.concatenate([directions_c,directions_f],dim=0).reshape(-1,2)

        
            parameters=projector.unproject(directions.to(self.device))
            n_parameters=parameters.shape[0]

            losses=torch.zeros((n_parameters,1))

            with torch.no_grad():
                for i in tqdm(range(n_parameters)):
                    loss=sampling.get_parameters_loss(parameter=parameters[i,:],model=Phi,dataloader=dataloader)
                    losses[i]=loss
            
            if projector.d==2:
                fig,ax=plt.subplots()
        
                SC=ax.scatter(directions[:,0],directions[:,1],s=50,c=losses.cpu().numpy()[:,0],cmap="viridis")
                CB = fig.colorbar(SC, shrink=0.8)
                exp=self.inputs.current_exp
                plt.savefig(f"./sampling_N_{exp}.png")

            results={"parameters":parameters,
                    "losses":losses,
                    "projected_parameters":torch.tensor(directions,dtype=torch.float32).reshape(-1,2),
                    "projector":projector}
            return results

    def LossApprox_callback(self,reduction="mean"):
        """
        LossApproxOperation entropy function
        """
        global counter_writer
        
        if self.inputs.stage_name == "after_training_epoch":

            if self.inputs.current_exp!=0:
                return self.inputs

            weights=sampling.extract_parameters(self.inputs.current_network)
            loss=sampling.get_parameters_loss(parameter=weights,model=self.inputs.current_network,dataloader=self.inputs.dataloader)
    #         
            sh=weights.shape
            weights=torch.reshape(weights,(1,sh[0]))

            self.History["initial_trajectory"]["parameters"].append(weights.data)
            self.History["initial_trajectory"]["losses"].append(loss.data)
       
        if self.inputs.stage_name == "after_training_exp":
            self.History["dataloaders"].append(copy.deepcopy(self.inputs.dataloader))
            # Get model current classifier
            if self.inputs.current_exp>=1:
                traj=self.History["current_trajectory"]["parameters"]
                # losses_bce=list(map(lambda x:x[1].reshape(-1,1),self.inputs.temp_var["trajectory"]))
                traj=np.concatenate(traj,axis=0)
                # losses_bce=np.concatenate(losses_bce,axis=0).squeeze()

                approximator_models=self.History["approximators"]
                for approximator in approximator_models:
                    fig=approximator.plot
                    axes=fig.get_axes()
                    axes[0].scatter(traj[:,0],traj[:,1],c="r", marker="x", s=2,cmap="viridis")
                    axes[0].scatter(traj[0:10,0],traj[0:10,1],c="b", marker="x", s=20,cmap="viridis")
                    axes[0].scatter(traj[-10:,0],traj[-10:,1],c="g", marker="x", s=10,cmap="viridis")
                    plt.savefig(f"./traj.png")
                plt.show()
                traj=self.History["current_trajectory"]["parameters"]
            
        if self.inputs.stage_name == "before_training_exp":
            self.device=self.inputs.current_network.device
            # Here we compute our approximation of the loss landscape before training
            
            #If we just started , we don't need an approximation
            if self.inputs.current_exp==0:
                return self.inputs
            elif self.inputs.current_exp ==1 : 
                # We are about to begin the second task.
                # We need to estimate for once and all differents quantities
                # 1. The projector fonction alongside the last trajectory
                # 2. Update the trained network with classifier with the classifier which optimise only in the reduce dimension

                d=self.inputs.plugins_storage[self.name]["hyperparameters"]["d"]
                d=int(d)

                self.projector=self.compute_projector(d=d,
                                       training_trajectory=self.History["initial_trajectory"]).to(self.device)
                self.classifier=self.compute_subspace_classifier(projector=self.projector,
                                                                 current_model=self.inputs.current_network).to(self.device)
                
            
            
            # Our reccurent task here is to sample datas in the d-space and build an approximator to estimate the data
            # We apply the next procedure
            # 1. Sample on a d-latin hypercube n datas, n is a hyperparameters given by the User 
            # 2. Using the sampling, train the approximator 
            # 2. Back up this approximator in the History
            #We need estimated the loss of the last task

            n=self.inputs.plugins_storage[self.name]["hyperparameters"]["n"]
            n=int(n)
            results=self.latin_hypercube_sampling(n=n,coarse_limit=100.0,fine_limit=1.0)
            #Backup these sampling

            anchors_projected=results["projected_parameters"]
            anchors_raw_losses=results["losses"]
            
            #Train the approximator
            epochs=self.inputs.plugins_storage[self.name]["hyperparameters"]["epochs"]
            approximator=approximators.BasicMLPAprroximator(theta_refs=anchors_projected,
                                    theta_refs_raw_losses=anchors_raw_losses,
                                    callback_hyperparameters={"epochs":epochs,
                                            "lr":1e-3,
                                            "mlp_model":approximators.BasicMLPModule,
                                            "optimizer":torch.optim.Adam})

            approximator=approximator.to(self.device)
            self.History["approximators"].append(approximator.eval())
            self.inputs.current_network.classifier=self.classifier

        if self.inputs.stage_name == "before_backward":

            # kernel=approximators.TriangleKernel
            coefs=self.inputs.dataloader.dataset.splits.sum(axis=0)


            # Get the logits and the targets
            logits=self.inputs.logits
            targets=self.inputs.targets

            #Get the old loss approximators
            # A list of neural networks models approximating the loss of old tasks
            approximator_models=self.History["approximators"]
            
            if len(approximator_models)<1 :
                loss = F.cross_entropy(logits, targets,reduction=reduction)
                # loss=loss*coefs[self.inputs.current_exp]/coefs[:self.inputs.current_exp+1].sum()
                writer.add_scalar(f"loss_per_task/train_{-1}",loss,counter_writer)

            if len(approximator_models)>0 and True:
                
                # Get model current classifier
                # loss=0.0
                loss = F.cross_entropy(logits, targets,reduction=reduction)
                loss=loss*10.0*coefs[self.inputs.current_exp]/coefs[:self.inputs.current_exp+1].sum()
                writer.add_scalar(f"loss_per_task/train_{-1}",loss,counter_writer)
                weights=self.inputs.current_network.classifier.weight
                sh=weights.shape
                weights=torch.reshape(weights,(1,sh[0]))

                for approximator in approximator_models:

                    mu_x,sigma_x=approximator.data_mean["x"],approximator.data_std["x"]
                    mu_y,sigma_y=approximator.data_mean["y"],approximator.data_std["y"]
                    mu_y=mu_y.to("cuda:0")
                    mu_x=mu_x.to("cuda:0")
                    sigma_x=sigma_x.to("cuda:0")
                    sigma_y=sigma_y.to("cuda:0")

                    
                    approximator.eval()
                    # loss_apprx=approximator(weights,kernel=kernel)
                    loss_apprx=approximator((weights-mu_x)/sigma_x)

                    if np.random.rand()>0.5:
                        self.History["current_trajectory"]["parameters"].append(weights.clone().detach().cpu().numpy().reshape(1,2))

                    for i,li in enumerate(loss_apprx):
                        loss+=torch.squeeze(li)*10*coefs[i]/coefs[:self.inputs.current_exp+1].sum()
                        writer.add_scalar(f"loss_per_task/train_{i}",li,counter_writer)
                
            loss_coeff=1.0
            counter_writer+=1

            

            self.inputs.loss+=loss_coeff*loss
                  
        return self.inputs
    
