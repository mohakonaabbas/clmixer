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
from typing import Dict, List,Union
from tqdm import tqdm
import math
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

#============================== LOSSES ===============================#
# DONE
class CrossEntropyProjectionOperation(Operation):
    def __init__(self,
                 entry_point =["before_backward","before_training_exp","after_eval_forward","after_training_epoch","after_training_exp"],
                  inputs=None,
                callback=(lambda x:x), 
                paper_ref="",
                is_loss=True):
        super().__init__(entry_point, inputs, callback, paper_ref,is_loss)
    
        self.set_callback(self.ce_callback)
        self.set_config_template({
      "name": self.__class__.__name__,
      "hyperparameters": {},
      "function": "knowledge_incorporation"
    })
        # Projector
        self.projector=None
        # History dictionary
        self.History={"initial_trajectory":{"parameters":[],"losses":[]},
                      "dataloaders":[],
                      "approximators":[],
                      "projectors":[],
                      "current_trajectory":{"parameters":[],"losses":[]},
                      "initial_network_arch":None}


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
      b=len(losses)
      pls.fit(parameters[:b,:]-reference_point.cpu().numpy(),losses[:b])

      d_hyperplane=pls.x_rotations_
      d_hyperplane=torch.tensor(d_hyperplane,dtype=torch.float32)
      mean=torch.tensor(pls._x_mean,dtype=torch.float32)
      std=torch.tensor(pls._x_std,dtype=torch.float32)

      print(d_hyperplane)

      

      projector=Projector(reference_point=reference_point,d_hyperplane=d_hyperplane,mean=mean,std=std)
      return projector
    
  
    def find_intersections(self):
        """
        This function tries to compute an intersection of differents hyperplane
        
        """

        def build_constraints(x,bases):
            projections=torch.matmul(x,bases)
            constraint_vector=x-(projections*bases).sum(dim=1)
            return constraint_vector
        
        def build_x(theta,base):
            x=(theta*base).sum(dim=1)
            return x


        #Take all the necessary parameters
        projectors=self.History["projectors"]
        N_projectors=len(projectors)
        As=[]
        bs=[]

        for i,projector in enumerate(projectors):
            Ai=projector.std.reshape(-1,1)*projector.hyperplane
            bi=projector.mean+projector.reference_point
            As.append(Ai)
            bs.append(bi)

        A=torch.concatenate(As,dim=1)
        b=bs[1]-bs[0]

        
        
        dim_d=projectors[0].d
        hyperplanes=list(map(lambda x:x.hyperplane,projectors))
        
        # hyperplanes=torch.concatenate(hyperplanes,dim=1)
        device=hyperplanes[0].device

        
        #Create the parameters to optimize d +k lagrange multipliers = n_projectors-1
        n_param_optim=len(projectors)*dim_d
        # n_param_optim=dim_d #-1 +len(projectors)
        data=torch.rand(n_param_optim,dtype=torch.float32).to(device)

        optim_param=torch.nn.parameter.Parameter(data,requires_grad=True)
        # torch.Tensor(n_param_optim)
        stdv = 0.001
        optim_param.data.uniform_(-stdv, stdv)
        #Create the optimizer
        optimizer=torch.optim.Adam(params=[optim_param],lr=1e-0)
        # optim_param=optim_param.to(device)

        #create the loss to optimize
        # loss_func=torch.nn.MSELoss()

        #Optimize
        epochs=2000
        # targets=torch.zeros_like(hyperplanes[0][:,0])
        
        index_=len(projectors)-1

        
        
        pbar=tqdm(range(epochs))
        pause1=[]
        pause2=[]
        for epoch in pbar:
            loss=torch.norm(torch.matmul(A,optim_param)-b)
            #Form x
            # x=projectors[index_].unproject(optim_param)
            # # x=build_x(optim_param[:dim_d],hyperplanes[0])+projectors[0].reference_point+projectors[0].mean
            # for i in range(0,len(projectors)):
            #     if i !=index_:
            #       c_proj_i=projectors[i](x)
            #       x_unproj_i=projectors[i].unproject(c_proj_i)
            #       const=torch.abs(x-x_unproj_i).sum()
            #       const=const+0*F.cross_entropy(x,x_unproj_i.softmax(dim=1))+nn.KLDivLoss()(x.log_softmax(dim=1),x_unproj_i.softmax(dim=1))
            #       loss+=const

            optimizer.zero_grad()
            # logits=optim_param*hyperplanes[:,1:]+hyperplanes[:,0].reshape(-1,1)
            # logits=logits.sum(dim=1)
            # loss=consts
            # loss+=torch.abs(torch.norm(optim_param)-1.0)
            pbar.set_description("%s  " % loss.item())
            
            loss.backward()
            optimizer.step()
            losses={}
            x=torch.matmul(As[0],optim_param[:dim_d])+bs[0]
            for j in range(len(projectors)):
              with torch.no_grad():
                losses[j]=sampling.get_parameters_loss(parameter=torch.matmul(As[j],optim_param[j*dim_d:(j+1)*dim_d])+bs[j],model=copy.deepcopy(self.inputs.current_network),dataloader=self.History["dataloaders"][j]).item()
            pause1.append(losses[1])
            pause2.append(losses[0])
            # print(losses,c_proj_i.data,optim_param.data)
            # print("--------------------------------------------")
        plt.plot(np.arange(epochs),pause1,"r")
        plt.plot(np.arange(epochs),pause2,"b")
        plt.show()
            

        #Estimate the loss at this point on all previous 
        with torch.no_grad():
          # theta=build_x(optim_param[:dim_d],hyperplanes[0])+projectors[0].reference_point
        #   theta=projectors[index_].unproject(optim_param)
          # theta=theta.sum(dim=1)
          losses={}
          xs={}
          for i in range(len(projectors)):
            x=torch.matmul(As[i],optim_param[i*dim_d:(i+1)*dim_d])+bs[i]
            xs[i]=x
            losses[i]=sampling.get_parameters_loss(parameter=x,model=copy.deepcopy(self.inputs.current_network),dataloader=self.History["dataloaders"][i])

        print(losses)

        for alpha in np.linspace(0.0,1.0,100):
            x=alpha*xs[0]+(1-alpha)*xs[1]
            
            losses1=sampling.get_parameters_loss(parameter=x,model=copy.deepcopy(self.inputs.current_network),dataloader=self.History["dataloaders"][0])
            losses2=sampling.get_parameters_loss(parameter=x,model=copy.deepcopy(self.inputs.current_network),dataloader=self.History["dataloaders"][1])

            print(losses1.item(),losses2.item())
            print("=======================================--==================================")



        

        return x,optim_param
    

    def latin_hypercube_sampling(self,
                n : int =1000,
                coarse_limit :int =100,
                fine_limit : int=1,
                parameters: Union[torch.Tensor,None] = None,
                directions: Union[torch.Tensor,None] = None ):
        
        if parameters is not None:
            assert directions is not None
            generate_parameters=False
        else:
            generate_parameters=True

        projector=self.projector
        Phi  = copy.deepcopy(self.initial_network_arch)
        dataloader = self.History["dataloaders"][-1]

        if generate_parameters:
            
            

            if projector.d>2:
                #Create a grid of points uniformely spaced in d-space
                sampler=qmc.LatinHypercube(d=projector.d)
                coeffs=sampler.random(n)
                coeffs=qmc.scale(coeffs,[-coarse_limit]*projector.d,[coarse_limit]*projector.d)
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
                "projected_parameters":torch.tensor(directions,dtype=torch.float32).reshape(-1,projector.d),
                "projector":projector}
        return results


    def ce_callback(self,reduction="mean"):
        """
        Cross entropy function
        """

        if self.inputs.stage_name == "after_training_epoch":


            weights=sampling.extract_parameters(self.inputs.current_network)
            loss=sampling.get_parameters_loss(parameter=weights,model=self.inputs.current_network,dataloader=self.inputs.dataloader)
    #         
            sh=weights.shape
            weights=torch.reshape(weights,(1,sh[0]))

            self.History["current_trajectory"]["parameters"].append(weights.data)
            self.History["current_trajectory"]["losses"].append(loss.data)
      
        if self.inputs.stage_name == "after_training_exp":
            self.History["dataloaders"].append(copy.deepcopy(self.inputs.dataloader))
            self.initial_network_arch=copy.deepcopy(self.inputs.current_network)
            n=self.inputs.plugins_storage[self.name]["hyperparameters"]["n"]
            n=int(n)
            d=self.inputs.plugins_storage[self.name]["hyperparameters"]["d"]
            d=int(d)

            self.projector=self.compute_projector(d=d,
                                    training_trajectory=self.History["current_trajectory"]).to(self.device)
            
            self.History["projectors"].append(copy.deepcopy(self.projector))
            results=self.latin_hypercube_sampling(n=n,coarse_limit=10.0,fine_limit=1.0)
            
            
            
            self.History["current_trajectory"]["parameters"]=[]
            self.History["current_trajectory"]["losses"]=[]

            if self.inputs.current_exp>0:
                self.find_intersections()

        if (self.inputs.stage_name == "before_training_exp") :
            if self.inputs.current_exp<-1:
                self.inputs.current_network.classifier.reset_parameters()
          
        
        if (self.inputs.stage_name == "before_backward") or (self.inputs.stage_name == "after_eval_forward"):
            self.device=self.inputs.current_network.device
            logits=self.inputs.logits
            targets=self.inputs.targets
            
            loss = F.cross_entropy(logits.softmax(dim=1), targets,reduction=reduction)
            loss_coeff=1.0
            # if self.inputs.seen_classes_mask is None : loss_coeff=1
            # else:
            #     loss_coeff= (sum(self.inputs.task_mask)-sum(self.inputs.seen_classes_mask))/sum(self.inputs.task_mask)

            if reduction =="none":
                return loss
            self.inputs.loss+=loss_coeff*loss

        return self.inputs
#