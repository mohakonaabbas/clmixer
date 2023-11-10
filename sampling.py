import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Union,List
from tqdm import tqdm
from torch.utils import data
from sklearn.model_selection import KFold


class EfficientSampler:
    def __init__(self,
                X : Union[torch.tensor,List[str]],
                y : torch.tensor,
                Phi : torch.Module,
                theta_mask : torch.tensor ):
        """
        Args:
            - X : Represent the dataset of the current task.
            if X is a list of str, it must represent the paths of the embeddings, which will loaded and converted to torch.tensor
            - y : Represents the labels
            - Phi : The architecture of the model we wish to sample parameters with his current parameters
            - theta_mask : the mask of Phi parameters which should be sampled. This allows to handle fixed parameters
        """

        if not isinstance(X,torch.tensor):
            assert isinstance( X, list)

            #Load the X list to torch. tensor
            valid_X

        self.X=valid_X
        self.y=y

        self.mask=theta_mask
        self.model_reference= Phi

            

    def sample(self, n, callback):
        """
        This functions takes in inputs a list of trajectories of training
        And augment it to generate N samples
        """
        print("AUGMENTATION NOT IMPLEMENTED YET , ")

        #Get a triangle sampling
        thetas,losses=callback(n,self.X,self.y,self.model_reference,self.mask)
        return thetas,losses


    



# DATALOADERS
class simpleDataset(torch.utils.data.Dataset):
    def __init__(self, X:torch.tensor,
                 y:torch.tensor):
        self.X=X
        self.y=y

    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):

        x=torch.tensor(self.X[idx])
        y=torch.tensor(self.y[idx])
        return x, y
#SAMPLERS

def extract_parameters(model):
    parameter=[]
    start=0
    for name,param in model.named_parameters():
            if ("weight" not in name) :
                continue
            parameter+=param.data.tolist()
    return parameter

def insert_parameters(model,parameter,train=False):
    # Update model parameter
    start=0
    for name,param in model.named_parameters():
            if ("weight" not in name) :
                continue
            shape=param.shape
            flat_shape=np.prod(param.shape)
            end=start+flat_shape
            param.data=parameter[start:start+flat_shape].reshape(shape)
            param.requires_grad = train
            start=end
    return model


def optimize_return_parameter(network,
                              loader,
                            epochs,
                            lr,
                            criterion):
    
    network.train()
    network=network.to('cuda:0')
    loss=0.0
    optimizer = torch.optim.SGD(network.parameters(), lr=lr)
    pbar=tqdm(range(epochs))

    for epoch in pbar:
        
        for inputs,targets in loader:

            outputs=network(inputs)
            loss = criterion(outputs["logits"].softmax(dim=1),targets)
            pbar.set_description("%s  " % loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    parameter=torch.tensor(extract_parameters(network))
    return parameter

def get_parameters_loss(parameter,
                        model,
                        dataloader,
                        criterion=F.cross_entropy):

    # Update model parameter
    start=0
    for name,param in model.named_parameters():
            if ("weight" not in name) :
                continue
            shape=param.shape
            flat_shape=np.prod(param.shape)
            end=start+flat_shape
            param.data=parameter[start:start+flat_shape].reshape(shape)
            param.requires_grad = False
            start=end


    # Test on the data loader
    count=1
    loss=0.0
    for inputs, targets in dataloader:
        inputs=inputs.to("cuda:0")
        targets=targets.to("cuda:0")
        outputs=model(inputs)
        loss+=criterion(outputs["logits"].softmax(dim=1),targets)
        count+=1
    return loss/count

def random_sampler_callback(n : int ,
                            X : torch.tensor,
                            y : torch.tensor,
                            Phi : torch.Module,
                            theta_mask : torch.tensor,
                            dataloader_bs : int = 32):
    """
     Random sampler of parameters
    """


    D=torch.sum(theta_mask).item()
    coeffs=torch.rand(n,D)
    # coeffs=coeffs/torch.sum(coeffs,dim=1).reshape(-1,1)
    indices=torch.argwhere(theta_mask)
    parameters=np.zeros(n,len(theta_mask))
    parameters[:,indices]=coeffs

    X_train=X
    y_train=y
    loader = data.DataLoader(simpleDataset(X=X_train,
                                           y=y_train ,
                                           batch_size=dataloader_bs,
                                           shuffle=True))
    losses=np.zeros(n,1)

    for i,parameter in enumerate(parameters.tolist()):
        loss=get_parameters_loss(parameter=parameter,model=Phi,dataloader=loader)
        losses[i]=loss

    
    
    return parameters,losses


def kfold_sampler_callback(n : int ,
                            X : torch.tensor,
                            y : torch.tensor,
                            Phi : torch.Module,
                            theta_mask : torch.tensor,
                            dataloader_bs : int = 32):
    """
     Random sampler of parameters
    """

    parameters=np.zeros(n,len(theta_mask))
    losses=np.zeros(n,1)

    # Configuration options
    k_folds = n

  
  # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    dataset=simpleDataset(X=X,y=y)
    loader = data.DataLoader(dataset,batch_size=dataloader_bs,shuffle=True)
    


      # K-fold Cross Validation model evaluation
    for fold, train_ids in enumerate(kfold.split(dataset)):
    
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
    
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)

        
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=dataloader_bs, sampler=train_subsampler)

    
        folded_parameter=optimize_return_parameter(network=Phi, loader = trainloader,epochs=100,lr=1e-3,criterion=F.cross_entropy)
        loss=get_parameters_loss(parameter=folded_parameter,model=Phi,dataloader=loader)
        losses[fold]=loss
        parameters[fold,:]=extract_parameters(Phi)

    parameters=torch.concatenate(parameters)

    return parameters,losses


def value_imposed_sampler_callback(n : int ,
                            X : torch.tensor,
                            y : torch.tensor,
                            Phi : torch.Module,
                            theta_mask : torch.tensor,
                            dataloader_bs : int = 32):
    """
     Random sampler of parameters
    """

    D=torch.sum(theta_mask).item()
    coeffs=torch.rand(n,D)
    # coeffs=coeffs/torch.sum(coeffs,dim=1).reshape(-1,1)
    indices=torch.argwhere(theta_mask)
    parameters=np.zeros(n,len(theta_mask))
    parameters[:,indices]=coeffs
    targets_values=torch.rand(n,1)
    targets_values=10*targets_values+(1.0-targets_values)*(-10)
    losses=np.zeros(n,1)

    
    dataset=simpleDataset(X=X,y=y)
    loader = data.DataLoader(dataset,batch_size=dataloader_bs,shuffle=True)
    Phi.train()
    Phi=Phi.to('cuda:0')
    optimizer = torch.optim.SGD(Phi.parameters(), lr=1e-3)
    pbar=tqdm(range(100))



      # Optimise with a target Cross Validation model evaluation
    for i, loss_target ,parameter in enumerate(targets_values.tolist(), parameters.tolist()):
        Phi=insert_parameters(Phi,parameter,train=True)
        for epoch in pbar:
            count=1
            loss=0.0
            
            for inputs, targets in loader:
                inputs=inputs.to("cuda:0")
                targets=targets.to("cuda:0")
                outputs=Phi(inputs)
                loss+=F.cross_entropy(outputs["logits"].softmax(dim=1),targets)
                count+=1
            loss=(loss/count-loss_target)**2
            losses[i]=loss.item()/count
            count=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    

        parameters[i,:]=extract_parameters(Phi)


    return parameters,losses


