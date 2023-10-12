from .base_plugin import Operation
from torch.nn import functional as F
from torch import nn
import torch
from torch.utils import data
from tqdm import tqdm
import copy
from typing import List, Tuple
from ..storage import Storage
from .finetune_last_layer_plugin import FinetuneOperation
class LossLandscapeOperation(Operation):
# class LossLandscapeOperation():
    def __init__(self,
                entry_point =["before_backward","after_backward","after_training_exp"],
                inputs={},
            callback=(lambda x:x), 
            paper_ref="",
                is_loss=True):
        super().__init__(entry_point, inputs, callback, paper_ref,is_loss)

        self.set_callback(self.lanscape_callback)
        self.set_config_template({
      "name": self.__class__.__name__,
      "hyperparameters": {
        "approximators": [],
        "current_task_loss_dataset":[],
        "n":10**4,
        "epochs":100,
        "bs":32,
        "lr":1e-3

      },
      "function": "knowledge_retention"
    })


    def lanscape_callback(self,reduction="mean"):
        """
        Landscape entropy function
        """

        

        

        
        if self.inputs.stage_name == "after_backward":
            # Get model current classifier
            param_list=[]
            for name,param in self.inputs.current_network.named_parameters():
                if ("weight" in name) : # or ("bias" in name):
                    param_list.append(torch.flatten(copy.copy(param.data)))
            weights=torch.cat(param_list)
            self.input_shape=weights.shape
            loss=self.inputs.loss
            self.inputs.plugins_storage[self.name]["hyperparameters"]["current_task_loss_dataset"].append((weights,loss.data))

        if self.inputs.stage_name == "after_training_exp":
            #Finetune the model
            
            finetune_operation=FinetuneOperation(inputs=self.inputs)

            update={finetune_operation.name:{"hyperparameters":{}}}
            finetune_operation.inputs.plugins_storage.update(update)
            
            update={"finetune_epochs":finetune_operation.config_template["hyperparameters"]["finetune_epochs"]}
            finetune_operation.inputs.plugins_storage[finetune_operation.name]["hyperparameters"].update(update)
            update={"finetune_bs":finetune_operation.config_template["hyperparameters"]["finetune_bs"]}
            finetune_operation.inputs.plugins_storage[finetune_operation.name]["hyperparameters"].update(update)
            update={"finetune_lr":finetune_operation.config_template["hyperparameters"]["finetune_lr"]}
            finetune_operation.inputs.plugins_storage[finetune_operation.name]["hyperparameters"].update(update)
            update={"cls_budget":finetune_operation.config_template["hyperparameters"]["cls_budget"]}
            finetune_operation.inputs.plugins_storage[finetune_operation.name]["hyperparameters"].update(update)
            
            finetune_operation.finetune_callback()


            param_list=[]
            for name,param in self.inputs.current_network.named_parameters():
                if ("weight" in name) : # or ("bias" in name):
                    param_list.append(torch.flatten(copy.copy(param.data)))
            finetuned_weights=torch.cat(param_list)
            self.input_shape=finetuned_weights.shape
            loss=self.inputs.loss

            # Create a dataset from the current_task_loss_dataset
            trajectory=[[(finetuned_weights,loss.data)],self.inputs.plugins_storage[self.name]["hyperparameters"]["current_task_loss_dataset"]]
           
            

            # Create a new model to approximate the loss landscape
            approximator=LandScapeModel(input_dim=self.input_shape[0])
            n=self.inputs.plugins_storage[self.name]["hyperparameters"]["n"]
            n=int(n)
            
            # Expand the dataset by "efficient" high dimensionnal sampling
            sampler=EfficientSampler(trajectory)
            X,y=sampler._gen_sample(n)

            # TRain it
            epochs=self.inputs.plugins_storage[self.name]["hyperparameters"]["epochs"]
            bs=self.inputs.plugins_storage[self.name]["hyperparameters"]["bs"]
            lr=self.inputs.plugins_storage[self.name]["hyperparameters"]["lr"]
        

            trainer=approximatorTrainer(network=approximator,
                                        epochs=epochs,
                                        bs=bs,
                                        lr=lr,
                                        X=X,
                                        y=y)
            
            with torch.no_grad():
                approximator=trainer.network
                for p in approximator.parameters():
                    p.requires_grad = False
            # Reinitialize the storage

            self.inputs.plugins_storage[self.name]["hyperparameters"]["current_task_loss_dataset"]=[]

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
                param_list=[]
                for name,param in self.inputs.current_network.named_parameters():
                    if ("weight" in name) : # or ("bias" in name):
                        param_list.append(torch.flatten(param))
                weights=torch.cat(param_list)

            for approximator in approximator_models:
                approximator.eval()
                loss+=approximator(weights.reshape(1,-1)).squeeze()
            
            loss_coeff=1.0

            self.inputs.loss+=loss_coeff*loss
        return self.inputs
    

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
    

class EfficientSampler:
    def __init__(self,
                 trajectories : List[List[Tuple]] ):
        self.references_trajectories=trajectories
            

    def _gen_sample(self, n):
        """
        This functions takes in inputs a list of trajectories of training
        And augment it to generate N samples
        """
        print("AUGMENTATION NOT IMPLEMENTED YET , ")

        #Get a triangle sampling
        sampling=self.triangle_sampling(n)
        sampling=sampling.to("cuda:0")
        theta0,theta1,theta2=self.references_trajectories[0][0][0],self.references_trajectories[1][0][0],self.references_trajectories[1][-1][0]
        sampled_parameters= sampling[:,0].reshape(-1,1)*theta0+ sampling[:,1].reshape(-1,1)*theta1+ sampling[:,2].reshape(-1,1)*theta2
        #We need to evaluate these parameters to get their loss now
        
        trajectory=self.references_trajectories[0]
        X=torch.cat(list(map(lambda x:x[0].reshape(1,-1),trajectory)),axis=0)
        y=torch.cat(list(map(lambda x:x[1].reshape(1,-1),trajectory)),axis=0)
        return X,y

    def triangle_sampling(self,n):
        coeffs=torch.rand(n,3)
        coeffs=coeffs/torch.sum(coeffs,dim=1).reshape(-1,1)
        return coeffs

class approximatorTrainer:
    def __init__(self,
                 network,
                 epochs,
                 bs,
                 lr,
                 X,
                 y):
        """
        Args :
            epochs : Epochs
            bs: batch size
            lr:learning rate
            X: weights to regress. Size =  encoder output x Classifier outputs
            y : loss value to regress to
            criterion : regression criterion
        
        """
        
        # Create a balanced dataset

        loader = data.DataLoader(simpleDataset(X=X,y= y),shuffle=True,batch_size=bs)
        
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
        

class simpleDataset(torch.utils.data.Dataset):
    def __init__(self, X:list[str],
                 y:list[int]):
        self.X=X
        self.y=y

    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):

        x=torch.tensor(self.X[idx])
        y=torch.tensor(self.y[idx])
        return x, y
