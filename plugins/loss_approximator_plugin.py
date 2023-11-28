
from .base_plugin import Operation
from torch.nn import functional as F
import copy
import torch

from typing import List, Tuple

import numpy as np
from datasets import base
from utils import sampling,approximators


class LossApproxOperation(Operation):
# class LossLandscapeOperation():
    def __init__(self,
                entry_point =["before_backward","after_backward","after_training_exp"],
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
        # "current_task_loss_dataset":[],
        "n":10**4,
        # "epochs":100,
        # "bs":32,
        # "lr":1e-3

      },
      "function": "knowledge_retention"
    })


    def LossApprox_callback(self,reduction="mean"):
        """
        LossApproxOperation entropy function
        """
        
        # if self.inputs.stage_name == "after_backward":
        #     # Get model current classifier
        #     param_list=[]
        #     for name,param in self.inputs.current_network.named_parameters():
        #         if ("weight" in name) : # or ("bias" in name):
        #             param_list.append(torch.flatten(copy.copy(param.data)))
        #     weights=extract_parameters(self.inputs.current_network)
        #     self.input_shape=weights.shape
        #     loss=self.inputs.loss
        #     self.inputs.plugins_storage[self.name]["hyperparameters"]["current_task_loss_dataset"].append((weights,loss.data))

        if self.inputs.stage_name == "after_training_exp":

            n=self.inputs.plugins_storage[self.name]["hyperparameters"]["n"]
            n=int(n)

            # Sample some data
            sampler=sampling.EfficientSampler(dataloader=self.inputs.dataloader,Phi=copy.deepcopy(self.inputs.current_network),theta_mask=None)
            samples,losses=sampler.sample(n)

            # Create a new approximator model to approximate the loss landscape
            approximator=approximators.BasicKernelAprroximator(theta_refs=samples,theta_refs_losses=losses)


            # self.inputs.plugins_storage[self.name]["hyperparameters"]["current_task_loss_dataset"]=[]

            # Save the value network in "approximators"
            self.inputs.plugins_storage[self.name]["hyperparameters"]["approximators"].append(approximator.eval())

        if self.inputs.stage_name == "before_backward":

            # Get the logits and the targets
            logits=self.inputs.logits
            targets=self.inputs.targets

            #Get the old loss approximators
            # A list of neural networks models approximating the loss of old tasks
            approximator_models=self.inputs.plugins_storage[self.name]["hyperparameters"]["approximators"]
            
            loss = F.cross_entropy(logits.softmax(dim=1), targets,reduction=reduction)

            if len(approximator_models)>0:
                # Get model current classifier
                
                weights=sampling.extract_parameters(self.inputs.current_network)
                # print(weights)

                for approximator in approximator_models:
                    approximator.eval()
                    loss_apprx=approximator(weights)

                    loss+=loss_apprx
            
            loss_coeff=1.0

            self.inputs.loss+=loss_coeff*loss
        return self.inputs
    



        


