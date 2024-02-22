
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
class LossApproxOperation(Operation):
# class LossLandscapeOperation():
    def __init__(self,
                entry_point =["before_backward","before_training_exp","after_training_epoch"],
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
        
        self.past_anchors=None
        self.past_anchors_raw_losses=None
        self.old_dataloaders=[]


    def LossApprox_callback(self,reduction="mean"):
        """
        LossApproxOperation entropy function
        """
        global counter_writer
        self.old_dataloaders.append(copy.deepcopy(self.inputs.dataloader))
        
        if self.inputs.stage_name == "after_training_epoch":
            # Get model current classifier

            weights=sampling.extract_parameters(self.inputs.current_network)
            loss=sampling.get_parameters_loss(parameter=weights,model=self.inputs.current_network,dataloader=self.inputs.dataloader)
    #         
            sh=weights.shape
            weights=torch.reshape(weights,(1,sh[0]))

            self.inputs.plugins_storage[self.name]["hyperparameters"]["current_task_loss_dataset"]["weights"].append(weights.data)
            self.inputs.plugins_storage[self.name]["hyperparameters"]["current_task_loss_dataset"]["losses"].append(loss.data)

        if self.inputs.stage_name == "before_training_exp":
            
            if self.inputs.current_exp<=0:
                return self.inputs


            

            n=self.inputs.plugins_storage[self.name]["hyperparameters"]["n"]
            n=int(n)

            # Get the recorded data
            rec_losses=torch.stack(self.inputs.plugins_storage[self.name]["hyperparameters"]["current_task_loss_dataset"]["losses"])
            rec_losses=torch.reshape(rec_losses,(rec_losses.shape[0],1))
            rec_weights=torch.concat(self.inputs.plugins_storage[self.name]["hyperparameters"]["current_task_loss_dataset"]["weights"])

            rec_weights,rec_losses=sampling.balance_dataset(rec_weights,rec_losses)

            # pca = PCA(n_components=2)
            

            # n_resample=rec_weights.shape[0]
            # modulo=max(1,n_resample//500)
            # rec_losses=rec_losses[::modulo]
            # rec_weights=rec_weights[::modulo]
           
            
        
            # Free the space 
            self.inputs.plugins_storage[self.name]["hyperparameters"]["current_task_loss_dataset"]["weights"]=[]
            self.inputs.plugins_storage[self.name]["hyperparameters"]["current_task_loss_dataset"]["losses"]=[]

            # Sample some data
            sampler=sampling.EfficientSampler(dataloader=self.inputs.dataloader,Phi=copy.deepcopy(self.inputs.current_network),theta_mask=None)

            rescaling_func=sampling.identity
            cb=sampling.retrain_sampler_callback
            training_epochs=self.inputs.epochs
            


            results=sampler.sample(3,callback=cb,
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

            optima=anchors[training_epochs-1::training_epochs]

            cb=sampling.random_dimension_reducer_sampler
            # training_epochs=self.inputs.plugins_storage[self.name]["hyperparameters"]["sampling_epochs"]
            


            results=sampler.sample(n,callback=cb,
                                   callback_hyperparameters={"optima":optima,"steps":100},
                            rescaling_func= rescaling_func,
                            rescaling_func_hyperparameters={"mean":None,"std":None,"lambda":None,"min":None,"max":None})


            # anchors,anchors_raw_losses=sampling.balance_dataset(anchors,anchors_raw_losses)
            

            

            if False:
                with torch.no_grad():
                    if self.past_anchors is not None:
                        #Format the losses
                        sh1,sh2,sh3=self.past_anchors_raw_losses.shape,anchors_raw_losses.shape,rec_losses.shape # sh1 is of form (n,m) and sh2 (n,1)
                        old_anchors_losses=sampling.get_batch_loss(self.past_anchors,
                                                model=copy.deepcopy(self.inputs.current_network),
                                                dataloader=self.inputs.dataloader)
                        losses_dim=(sh1[0]+sh2[0]+sh3[0],sh1[1]+1) # Add a new dimension to the losses. 
                        stacked_anchors_raw_losses=torch.ones(losses_dim)*torch.nan
                        
                        #Fill the losses
                        stacked_anchors_raw_losses[:sh1[0],:sh1[1]]=self.past_anchors_raw_losses
                        stacked_anchors_raw_losses[:sh1[0],-1]=torch.squeeze(old_anchors_losses)

                        stacked_anchors_raw_losses[sh1[0]:sh1[0]+sh2[0],-1]=torch.squeeze(anchors_raw_losses)
                        stacked_anchors_raw_losses[sh1[0]+sh2[0]:,-1]=torch.squeeze(rec_losses)
                        # anchors_losses=torch.concatenate([self.past_anchors_raw_losses,anchors_losses,rec_losses.to("cpu")])
                        anchors = torch.concatenate([self.past_anchors,anchors,rec_weights.to("cpu")])
                        # anchors_raw_losses = torch.concatenate([self.past_anchors_raw_losses,anchors_raw_losses,rec_losses.to("cpu")])
                        anchors_raw_losses=stacked_anchors_raw_losses
                    else:
                        # anchors_losses=torch.concatenate([anchors_losses,rec_losses.to("cpu")])
                        anchors = torch.concatenate([anchors,rec_weights.to("cpu")])
                        anchors_raw_losses = torch.concatenate([anchors_raw_losses,rec_losses.to("cpu")])

                pls = PLSRegression(n_components=2)
                pls.fit(anchors.cpu().numpy(),anchors_raw_losses.cpu().numpy())
                
                cb=sampling.pls_dimension_reducer_sampler
                training_epochs=self.inputs.plugins_storage[self.name]["hyperparameters"]["sampling_epochs"]
                


                results=sampler.sample(n,callback=cb,
                                    callback_hyperparameters={"projector":pls,"lr":1e-2,"reference":anchors[-1],"steps":5},
                                rescaling_func= rescaling_func,
                                rescaling_func_hyperparameters={"mean":None,"std":None,"lambda":None,"min":None,"max":None})

            #Backup these sampling
            anchors=results["parameters"]
            anchors_projected=results["projected_parameters"]
            anchors_raw_losses=results["losses"]
            rescaling_func_hyperparameters=results["rescaling_hyperparameters"]
            projection_func=results["projector"]
            self.past_anchors=copy.deepcopy(anchors)
            self.past_anchors_raw_losses=copy.deepcopy(anchors_raw_losses)

            # plt.hist(anchors_raw_losses.cpu().numpy().flatten())

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

            # approximator=approximators.AEMLPIncApproximator(theta_refs=anchors[::self.inputs.current_exp],
            #             theta_refs_raw_losses=anchors_raw_losses[::self.inputs.current_exp],
            #             callback_hyperparameters={"epochs":epochs,
            #                     "lr":1e-3,
            #                     "mlp_model":approximators.AEMLPIncModule,
            #                     "optimizer":torch.optim.Adam})
            
            approximator=approximators.BasicMLPAprroximator(theta_refs=anchors_projected[::self.inputs.current_exp],
                                    theta_refs_raw_losses=anchors_raw_losses[::self.inputs.current_exp],
                                    callback_hyperparameters={"epochs":epochs,
                                            "lr":1e-3,
                                            "mlp_model":approximators.BasicMLPModule,
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
            # self.inputs.plugins_storage[self.name]["hyperparameters"]["approximators"]=[approximator.eval()]
            self.inputs.plugins_storage[self.name]["hyperparameters"]["approximators"]=[(projection_func,approximator.eval())]
            # counter_writer=0

        if self.inputs.stage_name == "before_backward":

            # kernel=approximators.TriangleKernel
            coefs=self.inputs.dataloader.dataset.splits.sum(axis=0)


            # Get the logits and the targets
            logits=self.inputs.logits
            targets=self.inputs.targets

            #Get the old loss approximators
            # A list of neural networks models approximating the loss of old tasks
            approximator_models=self.inputs.plugins_storage[self.name]["hyperparameters"]["approximators"]
            
            
            

            # loss = F.cross_entropy(logits.softmax(dim=1), targets,reduction=reduction)
            loss = F.cross_entropy(logits, targets,reduction=reduction)
            # loss=loss*coefs[self.inputs.current_exp]/coefs[:self.inputs.current_exp+1].sum()
            writer.add_scalar(f"loss_per_task/train_{-1}",loss,counter_writer)

            if len(approximator_models)>0 and True:
                
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
                
                    
                for (projector,approximator) in approximator_models:
                    projector.to("cuda:0")
                    projector.eval()
                    weights_projected=projector(torch.squeeze(weights))
                    # fig=approximator.plot
                    # losses={"true":[],"pred":[]}
                    # drifts={"x":[],"y":[]}
                    
                    approximator.eval()
                    # loss_apprx=approximator(weights,kernel=kernel)
                    loss_apprx=approximator(weights_projected)
                    
                    # i_max=len(loss_apprx)
                    for i,li in enumerate(loss_apprx["pred"]):
                        with torch.no_grad():

                            true_loss=sampling.get_parameters_loss(parameter=weights,model=self.inputs.current_network,dataloader=self.old_dataloaders[i])
                            print(loss_apprx["encoding"],li.item(),true_loss.item())
                            # losses["true"].append(true_loss.item())
                            # losses["pred"].append(li.item())
                            # drifts["x"].append(loss_apprx["encoding"][0].item())
                            # drifts["y"].append(loss_apprx["encoding"][1].item())
                        
                        loss+=torch.squeeze(li)*coefs[i]/coefs[:self.inputs.current_exp+1].sum()
                        writer.add_scalar(f"loss_per_task/train_{i}",li,counter_writer)
                # plt.scatter(losses["true"],losses["pred"])

                # loss=loss/(i+1) 
                # loss=loss/(i+2) 
            #     # print("Overal Loss",loss,"Value Network",torch.mean(torch.tensor(losses_approx)))
            # else:
            #     loss = F.cross_entropy(logits, targets,reduction=reduction)
                # loss=loss*coefs[self.inputs.current_exp]/coefs[:self.inputs.current_exp+1].sum()
            
            loss_coeff=1.0
            counter_writer+=1

            

            self.inputs.loss+=loss_coeff*loss
        return self.inputs
    



        


