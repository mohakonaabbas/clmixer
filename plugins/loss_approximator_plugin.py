
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
                entry_point =["before_backward","after_training_epoch","after_training_exp"],
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
        
        if self.inputs.stage_name == "after_training_epoch":
            # Get model current classifier

            weights=sampling.extract_parameters(self.inputs.current_network)
            loss=sampling.get_parameters_loss(parameter=weights,model=self.inputs.current_network,dataloader=self.inputs.dataloader)
    #         
            sh=weights.shape
            weights=torch.reshape(weights,(1,sh[0]))

            self.inputs.plugins_storage[self.name]["hyperparameters"]["current_task_loss_dataset"]["weights"].append(weights.data)
            self.inputs.plugins_storage[self.name]["hyperparameters"]["current_task_loss_dataset"]["losses"].append(loss.data)

        if self.inputs.stage_name == "after_training_exp":

            n=self.inputs.plugins_storage[self.name]["hyperparameters"]["n"]
            n=int(n)

            # Get the recorded data
            rec_losses=torch.stack(self.inputs.plugins_storage[self.name]["hyperparameters"]["current_task_loss_dataset"]["losses"])
            rec_losses=torch.reshape(rec_losses,(rec_losses.shape[0],1))
            rec_weights=torch.concat(self.inputs.plugins_storage[self.name]["hyperparameters"]["current_task_loss_dataset"]["weights"])

            # Free the space 
            self.inputs.plugins_storage[self.name]["hyperparameters"]["current_task_loss_dataset"]["weights"]=[]
            self.inputs.plugins_storage[self.name]["hyperparameters"]["current_task_loss_dataset"]["losses"]=[]

            # Sample some data
            sampler=sampling.EfficientSampler(dataloader=self.inputs.dataloader,Phi=copy.deepcopy(self.inputs.current_network),theta_mask=None)

            rescaling_func=sampling.identity
            cb=sampling.retrain_sampler_callback
            training_epochs=self.inputs.plugins_storage[self.name]["hyperparameters"]["sampling_epochs"]
            results=sampler.sample(n,callback=cb,
                                   callback_hyperparameters={"epochs":training_epochs,"lr":1e-2},
                            rescaling_func= rescaling_func,
                            rescaling_func_hyperparameters={"mean":None,"std":None,"lambda":None,"min":None,"max":None})

            # sampled_anchors=results["parameters"]
            # sampled_anchors_losses=results["rescaled_losses"]
            # sampled_anchors_raw_losses=results["losses"]
            # rescaling_func_hyperparameters=results["rescaling_hyperparameters"]

            anchors=results["parameters"]
            anchors_losses=results["rescaled_losses"]
            anchors_raw_losses=results["losses"]
            rescaling_func_hyperparameters=results["rescaling_hyperparameters"]


            with torch.no_grad():
                anchors_losses=torch.concatenate([anchors_losses,rec_losses.to("cpu")])
                anchors = torch.concatenate([anchors,rec_weights.to("cpu")])
                anchors_raw_losses = torch.concatenate([anchors_raw_losses,rec_losses.to("cpu")])

            # anchors=torch.tensor(anchors)
            # anchors_losses=torch.tensor(anchors_losses)
            # anchors_raw_losses=torch.tensor(anchors_raw_losses)




            # samples,losses=sampler.sample(n)

            # Create a new approximator model to approximate the loss landscape
            # approximator=approximators.BasicKernelAprroximator(theta_refs=anchors,
            #                                                     theta_refs_raw_losses=anchors_raw_losses,
            #                                                     theta_refs_losses=anchors_losses)
            
            # approximator=approximators.AEMLPApproximator(theta_refs=anchors,
            #                                                     theta_refs_raw_losses=anchors_raw_losses)
            
            epochs=self.inputs.plugins_storage[self.name]["hyperparameters"]["epochs"]
            approximator=approximators.AEMLPApproximator(theta_refs=anchors,
                                    theta_refs_raw_losses=anchors_raw_losses,
                                    callback_hyperparameters={"epochs":epochs,
                                            "lr":1e-3,
                                            "mlp_model":approximators.AEMLPModule,
                                            "optimizer":torch.optim.Adam})
            
            
            #  #Calibrate this kernel h
            # n_h=20
            # kernel=approximators.TriangleKernel
            # results=sampler.sample(n_h,callback=cb,
            #                         rescaling_func= rescaling_func,
            #                         rescaling_func_hyperparameters=rescaling_func_hyperparameters)

            # calibration_samples=results["parameters"]
            # calibration_targets=results["rescaled_losses"]


            # h=approximator.calibrate_h(calibration_samples= calibration_samples , 
            #                             calibration_targets= calibration_targets, 
            #                             method= "knn", 
            #                             method_hyperparameters={"min_nbrs_neigh":10,"kernel":kernel})
            # approximator.set_rescaling_parameters(rescaling_func,parameters=rescaling_func_hyperparameters)

            approximator=approximator.to("cuda:0")



            # self.inputs.plugins_storage[self.name]["hyperparameters"]["current_task_loss_dataset"]=[]

            # Save the value network in "approximators"
            self.inputs.plugins_storage[self.name]["hyperparameters"]["approximators"].append(approximator.eval())

        if self.inputs.stage_name == "before_backward":

            # kernel=approximators.TriangleKernel

            # Get the logits and the targets
            logits=self.inputs.logits
            targets=self.inputs.targets

            #Get the old loss approximators
            # A list of neural networks models approximating the loss of old tasks
            approximator_models=self.inputs.plugins_storage[self.name]["hyperparameters"]["approximators"]
            
            
            

            

            if len(approximator_models)>0:
                loss = F.cross_entropy(logits.softmax(dim=1), targets,reduction=reduction)
                # Get model current classifier
                # loss=0.0
                
                weights=sampling.extract_parameters(self.inputs.current_network)
                # if not weights.requires_grad:
                #     print(weights)
                # else:
                #     print("ok")
                    
                sh=weights.shape
                weights=torch.reshape(weights,(1,sh[0]))
                # print(weights)
                
                for (i,approximator) in enumerate(approximator_models):
                    approximator.eval()
                    # loss_apprx=approximator(weights,kernel=kernel)
                    loss_apprx=approximator(weights)
                    # loss+=loss_apprx["pred"]
                    loss+=torch.squeeze(loss_apprx["pred"])

                # loss=loss/(i+1) 
                loss=loss/(i+2) 
                # print("Overal Loss",loss,"Value Network",torch.mean(torch.tensor(losses_approx)))
            else:
                loss = F.cross_entropy(logits.softmax(dim=1), targets,reduction=reduction)
            
            loss_coeff=1.0

            

            self.inputs.loss+=loss_coeff*loss
        return self.inputs
    



        


