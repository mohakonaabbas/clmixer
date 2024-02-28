import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Union,List, Callable, Dict
from tqdm import tqdm
from torch.utils import data
from sklearn.model_selection import KFold
from scipy import stats
from copy import deepcopy
import warnings
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import qmc
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
#UTILS
def extract_parameters(model):
    parameter=[]
    start=0
    for name,param in model.named_parameters():
            if ("weight" not in name) :
                continue
            # parameter+=torch.flatten(param.data).tolist()
            parameter.append(torch.flatten(param))
    parameter=torch.cat(parameter)
    if not parameter.requires_grad:
        print("wierd")

    return parameter


def extract_parameters_grad(model):
    parameter=[]
    start=0
    for name,param in model.named_parameters():
            if ("weight" not in name) :
                continue
            if not param.requires_grad:
                print("wierd")
            # parameter+=torch.flatten(param.data).tolist()
            parameter.append(torch.flatten(param.grad))
    parameter=torch.cat(parameter)
    

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

def insert_parameters_grad(model,parameter,train=False):
    # Update model parameter
    start=0
    for name,param in model.named_parameters():
            if ("weight" not in name) :
                continue
            shape=param.shape
            flat_shape=np.prod(param.shape)
            end=start+flat_shape
            param.grad=parameter[start:start+flat_shape].reshape(shape)
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
            inputs=inputs.to("cuda:0")
            targets=targets.to("cuda:0")

            outputs=network(inputs)
            loss = criterion(outputs["logits"],targets)
            # print("%s  " % loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    parameter=torch.tensor(extract_parameters(network))
    return parameter

def get_parameters_loss(parameter,
                        model,
                        dataloader,
                        criterion=F.cross_entropy,
                        requires_grad=False):

    # Update model parameter
    start=0
    for name,param in model.named_parameters():
            if ("weight" not in name) :
                continue
            shape=param.shape
            flat_shape=np.prod(param.shape)
            end=start+flat_shape
            param.data=parameter[start:start+flat_shape].reshape(shape)
            param.requires_grad = requires_grad
            start=end


    # Test on the data loader
    with torch.no_grad():

        count=0
        loss=0.0
        model=model.to("cuda:0")
        model.eval()
        for inputs, targets in dataloader:
            inputs=inputs.to("cuda:0")
            targets=targets.to("cuda:0")
            outputs=model(inputs)
            loss+=criterion(outputs["logits"],targets)
            count+=1

    
    # Update model parameter
    start=0
    for name,param in model.named_parameters():
            # if ("weight" not in name) :
            #     continue
            # shape=param.shape
            # flat_shape=np.prod(param.shape)
            # end=start+flat_shape
            # param.data=parameter[start:start+flat_shape].reshape(shape)
            param.requires_grad = not requires_grad
            # start=end




    return loss/count

def get_batch_loss(parameters,
                        model,
                        dataloader,
                        criterion=F.cross_entropy,
                        requires_grad=False):
    losses=[]
    for parameter in parameters:
        loss=get_parameters_loss(parameter=parameter,model=model,dataloader=dataloader)
        losses.append(loss.view(1,-1))
    with torch.no_grad():
        losses=torch.concatenate(losses).cpu()
    
    return losses

def balance_dataset(parameters,parameters_loss):
    """
    Use a histogram strategy to sample data
    """
    local_parameters_losses=parameters_loss.clone().cpu().numpy()
    bins=np.histogram(local_parameters_losses,density=False,bins=10)[1]
    binned_losses=np.squeeze(np.digitize(local_parameters_losses,bins))
    arg_binned_losses=np.arange(len(binned_losses))
    bins_idx,counts=np.unique(binned_losses,return_counts=True)
    assert min(counts)>=1
    choices=[]
    for bin_idx in bins_idx:
        choices+=np.random.choice(arg_binned_losses[binned_losses==bin_idx],min(counts)).tolist()
    
    balanced_parameters=parameters[choices,:]
    balanced_parameters_losses=parameters_loss[choices,:]
    return balanced_parameters,balanced_parameters_losses

# SCALERS

def identity(inputs,
            hyperparameters={}):
    return inputs, hyperparameters
    
    
def minmax(inputs,
            hyperparameters={"min":None,"max":None}):
    """
    A Centered box cox data
    """
    if isinstance(inputs,torch.Tensor):
        w_inputs=inputs
    elif isinstance(inputs,list):
        w_inputs=torch.tensor(inputs)
    elif isinstance(inputs,np.ndarray):
        w_inputs=torch.tensor(inputs)
    else:
        raise ValueError


    assert isinstance(inputs,torch.Tensor)

    mini= hyperparameters["min"] if hyperparameters["min"] is not None else torch.min(w_inputs)
    maxi=hyperparameters["max"] if hyperparameters["max"] is not None else torch.max(w_inputs)
    outputs = (w_inputs-mini)/(maxi-mini+1e-32)



    return outputs, {"min":mini,"max":maxi}

def normal(inputs,
            hyperparameters={"mean":None,"std":None}):
    """
    A Centered box cox data
    """
    
    if isinstance(inputs,torch.Tensor):
        w_inputs=inputs
    elif isinstance(inputs,list):
        w_inputs=torch.tensor(inputs)
    elif isinstance(inputs,np.ndarray):
        w_inputs=torch.tensor(inputs)
    else:
        raise ValueError


    assert isinstance(inputs,torch.Tensor)

    mean= hyperparameters["mean"] if hyperparameters["mean"] is not None else torch.mean(w_inputs)
    std=hyperparameters["std"] if hyperparameters["std"] is not None else torch.std(w_inputs)
    
    outputs = (w_inputs-mean)/std
    return outputs, {"mean":mean,"std":std}


def box_cox(inputs,
            hyperparameters={"mean":None,"std":None,"lambda":None}):
    """
    A Centered box cox data
    """

    lambda_= hyperparameters["lambda"]

    isInference = lambda_ is not None
    if isInference:
        assert isinstance(inputs,torch.Tensor)
        mean= hyperparameters["mean"] 
        std=hyperparameters["std"]

        if lambda_== 0:
            outputs=torch.log(inputs)
        elif lambda_ != 0:
            outputs=(inputs**lambda_-1)/lambda_
        outputs = (outputs-mean)/std

        return outputs, hyperparameters


    if isinstance(inputs,torch.Tensor):
        w_inputs=inputs.detach().numpy()
    elif isinstance(inputs,list):
        w_inputs=np.array(inputs)
    elif isinstance(inputs,np.array):
        w_inputs=inputs
    else:
        raise ValueError
    
    assert isinstance(w_inputs,np.ndarray)
    w_inputs=np.squeeze(w_inputs)

    


    outputs, lambda_ = stats.boxcox(w_inputs,lmbda=lambda_)

    mean= hyperparameters["mean"] if hyperparameters["mean"] is not None else np.mean(outputs)
    std=hyperparameters["std"] if hyperparameters["std"] is not None else np.std(outputs)
    
    
    outputs = (outputs-mean)/std



    return outputs, {"mean":mean,"std":std,"lambda":lambda_}

class Projector(nn.Module):
    def __init__(self, reference_point, x,y) -> None:
        super().__init__()
        self.reference_point=nn.parameter.Parameter(reference_point,requires_grad=False)
        self.x=nn.parameter.Parameter(x,requires_grad=False)
        self.y=nn.parameter.Parameter(y,requires_grad=False)


    def forward(self,parameter):
        """
        Projector in he 2d planes defined by the hyperplane
        """
        vector=parameter-self.reference_point
        x=torch.dot(vector,self.x).reshape(-1,1)
        y=torch.dot(vector,self.y).reshape(-1,1)
        projected=torch.squeeze(torch.cat([x,y]))
        return projected



#CALLBACKS
def random_sampler_callback(n : int ,
                            dataloader : data.DataLoader,
                            Phi : torch.nn.Module,
                            theta_mask : torch.tensor,
                            callback_hyperparameters : Dict,
                            rescaling_func : Union[None,Callable]=None,
                            rescaling_func_hyperparameters : Union[None,Dict] = {}):
    """
    Random sampler of parameters
    n : int : The number of sample to generate
    dataloader : data.DataLoader : The dataloader with all the data
    Phi : torch.nn.Module : The model to condition the fit on
    theta_mask : torch.tensor : The mask of the parameters to sample
    rescaling_func : Callable : a function that transform the data
    rescaling_func_hyperparameters : the hyperparameters needed for rescaling_func
    """


    D=torch.sum(theta_mask).item()
    dimension=len(theta_mask)
    coeffs=torch.rand(n,D)
    coeffs=-1+2*coeffs
    # coeffs=coeffs/torch.sum(coeffs,dim=1).reshape(-1,1)
    indices=torch.argwhere(theta_mask)
    indices=torch.squeeze(indices)
    parameters=torch.ones((n,dimension)) #*torch.tensor(extract_parameters(Phi))
    # parameters=torch.zeros((n,len(theta_mask)))
    parameters[:,indices]=coeffs

    losses=torch.zeros((n,1))

    with torch.no_grad():
        for i in tqdm(range(n)):
            loss=get_parameters_loss(parameter=parameters[i,:],model=Phi,dataloader=dataloader)
            losses[i]=loss

    if rescaling_func is not None:
        rescaled_losses,rescaling_func_hyperparameters=rescaling_func(losses,rescaling_func_hyperparameters)
    else:
        rescaled_losses=None
        rescaling_func_hyperparameters=None

    # Map everything to torch tensor
    if not isinstance(rescaled_losses,torch.Tensor):
        rescaled_losses=torch.tensor(rescaled_losses)
    for key,value in rescaling_func_hyperparameters.items():
        if not isinstance(value,torch.Tensor):
            if value is not None:
                rescaling_func_hyperparameters[key]=torch.tensor(value)



    results={"initial_parameters":coeffs,"parameters":parameters,"losses":losses,"rescaled_losses":rescaled_losses,"rescaling_hyperparameters": rescaling_func_hyperparameters}


    
    
    return results

def stratified_random_sampler_callback(n : int ,
                            dataloader : data.DataLoader,
                            Phi : torch.nn.Module,
                            theta_mask : torch.tensor,
                            callback_hyperparameters : Dict,
                            rescaling_func : Union[None,Callable]=None,
                            rescaling_func_hyperparameters : Union[None,Dict] = {}):
    """
    Random sampler of parameters
    n : int : The number of sample to generate
    dataloader : data.DataLoader : The dataloader with all the data
    Phi : torch.nn.Module : The model to condition the fit on
    theta_mask : torch.tensor : The mask of the parameters to sample
    rescaling_func : Callable : a function that transform the data
    rescaling_func_hyperparameters : the hyperparameters needed for rescaling_func
    """


    D=torch.sum(theta_mask).item()
    dimension=len(theta_mask)
    sampler=qmc.LatinHypercube(d=D)
    coeffs=sampler.random(n)
    coeffs=torch.tensor(coeffs,dtype=torch.float32)
    coeffs=-1+2*coeffs
    # coeffs=torch.rand(n,D)
    # coeffs=coeffs/torch.sum(coeffs,dim=1).reshape(-1,1)
    indices=torch.argwhere(theta_mask)
    indices=torch.squeeze(indices)
    parameters=torch.ones((n,dimension)) #*torch.tensor(extract_parameters(Phi))
    # parameters=torch.zeros((n,len(theta_mask)))
    parameters[:,indices]=coeffs

    losses=torch.zeros((n,1))

    with torch.no_grad():
        for i in tqdm(range(n)):
            loss=get_parameters_loss(parameter=parameters[i,:],model=Phi,dataloader=dataloader)
            losses[i]=loss

    if rescaling_func is not None:
        rescaled_losses,rescaling_func_hyperparameters=rescaling_func(losses,rescaling_func_hyperparameters)
    else:
        rescaled_losses=None
        rescaling_func_hyperparameters=None

    # Map everything to torch tensor
    if not isinstance(rescaled_losses,torch.Tensor):
        rescaled_losses=torch.tensor(rescaled_losses)
    for key,value in rescaling_func_hyperparameters.items():
        if not isinstance(value,torch.Tensor):
            if value is not None:
                rescaling_func_hyperparameters[key]=torch.tensor(value)



    results={"initial_parameters":coeffs,"parameters":parameters,"losses":losses,"rescaled_losses":rescaled_losses,"rescaling_hyperparameters": rescaling_func_hyperparameters}


    
    
    return results


def kfold_sampler_callback(n : int ,
                            X : torch.tensor,
                            y : torch.tensor,
                            Phi : torch.nn.Module,
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
                            dataloader : data.DataLoader,
                            Phi : torch.nn.Module,
                            theta_mask : torch.Tensor,
                            callback_hyperparameters : Dict ,
                            rescaling_func : Union[None,Callable]=None,
                            rescaling_func_hyperparameters : Union[None,Dict] = {}):
    """
    Value imposed sampler of parameters
    n : int : The number of sample to generate
    dataloader : data.DataLoader : The dataloader with all the data
    Phi : torch.nn.Module : The model to condition the fit on
    theta_mask : torch.tensor : The mask of the parameters to sample
    rescaling_func : Callable : a function that transform the data
    rescaling_func_hyperparameters : the hyperparameters needed for rescaling_func
    """
    range_loss=callback_hyperparameters["range"]
    epochs=callback_hyperparameters["epochs"]
    lr=callback_hyperparameters["lr"]
    D=torch.sum(theta_mask).item()
    dimension=len(theta_mask)
    coeffs=torch.rand(n,D)
    coeffs=-1+2*coeffs
    # coeffs=coeffs/torch.sum(coeffs,dim=1).reshape(-1,1)
    indices=torch.argwhere(theta_mask)
    indices=torch.squeeze(indices)
    parameters=torch.ones((n,dimension)) #*torch.tensor(extract_parameters(Phi))
    parameters[:,indices]=coeffs

    losses=torch.zeros((n,1))
    targets_values=torch.rand(n,1)
    targets_values=targets_values*range_loss[0]+(1.0-targets_values)*range_loss[1]
    
    losses=torch.zeros((n,1))

    
    loader = dataloader
    
    optimizer = torch.optim.Adam(Phi.parameters(), lr=lr)
    pbar=tqdm(range(epochs))



      # Optimise with a target Cross Validation model evaluation
    for i in tqdm(range(n)):
        parameter=parameters[i,:]
        loss_target=targets_values[i,:]
        Phi=insert_parameters(Phi,parameter,train=True)
        Phi.train()
        Phi=Phi.to('cuda:0')
        for epoch in range(epochs):
            count=1
            loss=0.0
            loss1=0.0
            for inputs, targets in loader:
                inputs=inputs.to("cuda:0")
                targets=targets.to("cuda:0")
                
                outputs=Phi(inputs)
                loss+=F.cross_entropy(outputs["logits"],targets)
                count+=1
            
            loss_target=loss_target.to("cuda:0")
            loss1=torch.abs(loss_target-loss/count)

            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()
    

        parameters[i,:]=extract_parameters(Phi)

    with torch.no_grad():
        for i in tqdm(range(n)):
            loss=get_parameters_loss(parameter=parameters[i,:],model=Phi,dataloader=dataloader)
            losses[i]=loss
    print(list(zip(targets_values,losses)))

    if rescaling_func is not None:
        rescaled_losses,rescaling_func_hyperparameters=rescaling_func(losses,rescaling_func_hyperparameters)
    else:
        rescaled_losses=None
        rescaling_func_hyperparameters=None

    # Map everything to torch tensor
    if not isinstance(rescaled_losses,torch.Tensor):
        rescaled_losses=torch.tensor(rescaled_losses)
    for key,value in rescaling_func_hyperparameters.items():
        if not isinstance(value,torch.Tensor):
            if value is not None:
                rescaling_func_hyperparameters[key]=torch.tensor(value)


    results={"initial_parameters":coeffs, "parameters":parameters,"losses":losses,"rescaled_losses":rescaled_losses,"rescaling_hyperparameters": rescaling_func_hyperparameters}


    
    
    return results



def retrain_sampler_callback(n : int ,
                            dataloader : data.DataLoader,
                            Phi : torch.nn.Module,
                            theta_mask : torch.Tensor,
                            callback_hyperparameters : Dict ,
                            rescaling_func : Union[None,Callable]=None,
                            rescaling_func_hyperparameters : Union[None,Dict] = {}):
    """
    Value imposed sampler of parameters
    n : int : The number of sample to generate
    dataloader : data.DataLoader : The dataloader with all the data
    Phi : torch.nn.Module : The model to condition the fit on
    theta_mask : torch.tensor : The mask of the parameters to sample
    rescaling_func : Callable : a function that transform the data
    rescaling_func_hyperparameters : the hyperparameters needed for rescaling_func
    """
    epochs=callback_hyperparameters["epochs"]
    lr=callback_hyperparameters["lr"]
    last_only=callback_hyperparameters["save_only"]
    D=torch.sum(theta_mask).item()
    dimension=len(theta_mask)
    coeffs=torch.rand(n,D)
    coeffs=-1+2*coeffs
    # coeffs=coeffs/torch.sum(coeffs,dim=1).reshape(-1,1)
    indices=torch.argwhere(theta_mask)
    indices=torch.squeeze(indices)
    parameters=torch.ones((n,dimension)) #*torch.tensor(extract_parameters(Phi))
    parameters[:,indices]=coeffs

    losses=torch.zeros((n,1))


    
    loader = dataloader
    
    optimizer = torch.optim.Adam(Phi.parameters(), lr=lr)
    pbar=tqdm(range(epochs))



      # Optimise with a target Cross Validation model evaluation
    
    collected_theta=[]
    collected_theta_losses=[]

    losses_list=[]
    parameters_list=[]
    save=True
    

    for i in tqdm(range(n)):
        if last_only:
            save=False
        parameter=parameters[i,:]
        Phi=insert_parameters(Phi,parameter,train=True)
        Phi.train()
        Phi=Phi.to('cuda:0')
        collected_theta=[]
        # collected_theta_losses=[]
        for epoch in tqdm(range(epochs)):
            count=1
            avg_loss=0.0
            for inputs, targets in loader:
                loss=0.0
                optimizer.zero_grad()
                inputs=inputs.to("cuda:0")
                targets=targets.to("cuda:0")
                
                outputs=Phi(inputs)
                loss=F.cross_entropy(outputs["logits"],targets)
                count+=1
                
                loss.backward()
                optimizer.step()
                # avg_loss=avg_loss+(loss-avg_loss)/count
            # if np.random.choice([True,False],p=[0.5,0.5]):
            if epoch==epochs-1:
                save=True
            if save:
                collected_theta.append(extract_parameters(Phi).data.view(1,-1))
            # collected_theta_losses.append(avg_loss.data.view(1,-1))
        
        
        params=torch.concatenate(collected_theta).cpu()
        
        
        # params,loss_params=balance_dataset(torch.concatenate(collected_theta).cpu(),torch.concatenate(collected_theta_losses).cpu())
        parameters_list.append(params)
        # losses_list.append(loss_params)
        # print(i,avg_loss.data)

    

        # parameters[i,:]=extract_parameters(Phi)
    
    
    with torch.no_grad():
        parameters=torch.concatenate(parameters_list).cpu()
        for i in tqdm(range(n)):
            loss=get_parameters_loss(parameter=parameters[i,:],model=Phi,dataloader=dataloader)
            losses[i]=loss

    # print(losses)

    if rescaling_func is not None:
        rescaled_losses,rescaling_func_hyperparameters=rescaling_func(losses,rescaling_func_hyperparameters)
    else:
        rescaled_losses=None
        rescaling_func_hyperparameters=None

    # Map everything to torch tensor
    if not isinstance(rescaled_losses,torch.Tensor):
        rescaled_losses=torch.tensor(rescaled_losses)
    for key,value in rescaling_func_hyperparameters.items():
        if not isinstance(value,torch.Tensor):
            if value is not None:
                rescaling_func_hyperparameters[key]=torch.tensor(value)

    


    results={"parameters":parameters,"losses":losses,"rescaled_losses":rescaled_losses,"rescaling_hyperparameters": rescaling_func_hyperparameters,"initial_parameters":coeffs}


    
    
    return results

def pls_dimension_reducer_sampler(n : int ,
                            dataloader : data.DataLoader,
                            Phi : torch.nn.Module,
                            theta_mask : torch.Tensor,
                            callback_hyperparameters : Dict ,
                            rescaling_func : Union[None,Callable]=None,
                            rescaling_func_hyperparameters : Union[None,Dict] = {}):
    """
    This sampler is particular
    It takes as inputs a projector function , usally a PLS dim reducer to dim 2, a starting point for sampling
    n : int : The number of sample to generate
    dataloader : data.DataLoader : The dataloader with all the data
    Phi : torch.nn.Module : The model to condition the fit on
    theta_mask : torch.tensor : The mask of the parameters to sample
    rescaling_func : Callable : a function that transform the data
    rescaling_func_hyperparameters : the hyperparameters needed for rescaling_func
    """

    projector=callback_hyperparameters["projector"]
    reference_point=callback_hyperparameters["reference"]
    lr=callback_hyperparameters["lr"]
    n_steps=callback_hyperparameters["steps"]

    directions=1-2*np.random.rand(n,n_steps,2)
    directions=np.cumsum(directions,axis=1)
    origin=projector.transform(reference_point.reshape(1,-1))
    directions=origin.reshape(1,1,2)+directions
    # directions=lr*directions



    #Get the parameters
    parameters=projector.inverse_transform(directions.reshape(-1,2))

    out_shape=(directions.shape[0]*directions.shape[1],parameters.shape[-1])
    n_parameters=out_shape[0]
    parameters=torch.tensor(parameters.reshape(out_shape),dtype=torch.float32)
    losses=torch.zeros((n_parameters,1))

    with torch.no_grad():
        for i in tqdm(range(n_parameters)):
            loss=get_parameters_loss(parameter=parameters[i,:],model=Phi,dataloader=dataloader)
            losses[i]=loss
    # print(list(zip(targets_values,losses)))

    if rescaling_func is not None:
        rescaled_losses,rescaling_func_hyperparameters=rescaling_func(losses,rescaling_func_hyperparameters)
    else:
        rescaled_losses=None
        rescaling_func_hyperparameters=None

    # Map everything to torch tensor
    if not isinstance(rescaled_losses,torch.Tensor):
        rescaled_losses=torch.tensor(rescaled_losses)
    for key,value in rescaling_func_hyperparameters.items():
        if not isinstance(value,torch.Tensor):
            if value is not None:
                rescaling_func_hyperparameters[key]=torch.tensor(value)

    fig,ax=plt.subplots()
    # for i in range(n):
    # .reshape(n,n_steps,1)
    SC=ax.scatter(directions.reshape(-1,2)[:,0],directions.reshape(-1,2)[:,1],s=100,c=losses.cpu().numpy()[:,0],cmap="viridis")
    CB = fig.colorbar(SC, shrink=0.8)
    plt.savefig(f"./sampling.png")


    results={"parameters":parameters,
             "losses":losses,
             "rescaled_losses":rescaled_losses,
             "rescaling_hyperparameters": rescaling_func_hyperparameters,
             "projected_parameters":torch.tensor(directions,dtype=torch.float32).reshape(-1,2)}



    
    
    return results



def random_walk(n : int ,
                callback_hyperparameters : Dict):
    
    optima_points=callback_hyperparameters["optima"]
    
    reference_point=torch.mean(optima_points,dim=0)
    axis_0,axis_1=optima_points[0]-optima_points[1],optima_points[0]-optima_points[2]
    axis_0,axis_1=axis_0/torch.linalg.norm(axis_0),axis_1/torch.linalg.norm(axis_1)
    n_steps=callback_hyperparameters["steps"]

    directions=1-2*torch.rand(n,n_steps,2)
    directions=torch.cumsum(directions,axis=1)
    D=reference_point.shape[0]
    

    
    # directions=lr*directions

    #Get the parameters
    parameters=reference_point.reshape(1,1,D) + \
                directions[:,:,0].reshape(n,n_steps,1)*axis_0.reshape(1,1,D) + \
                directions[:,:,1].reshape(n,n_steps,1)*axis_1.reshape(1,1,D)
    



    
    optima_points=callback_hyperparameters["optima"]
    reference_point=torch.mean(optima_points,dim=0)
    axis_0,axis_1=optima_points[0]-optima_points[1],optima_points[0]-optima_points[2]
    axis_0,axis_1=axis_0/torch.linalg.norm(axis_0),axis_1/torch.linalg.norm(axis_1)
    n_steps=callback_hyperparameters["steps"]

    directions=1-2*torch.rand(n,n_steps,2)
    directions=torch.cumsum(directions,axis=1)
    D=reference_point.shape[0]
    

    
    # directions=lr*directions

    #Get the parameters
    parameters=reference_point.reshape(1,1,D) + \
                directions[:,:,0].reshape(n,n_steps,1)*axis_0.reshape(1,1,D) + \
                directions[:,:,1].reshape(n,n_steps,1)*axis_1.reshape(1,1,D)
    

    out_shape=(directions.shape[0]*directions.shape[1],parameters.shape[-1])
    n_parameters=out_shape[0]
    parameters=torch.tensor(parameters.reshape(out_shape),dtype=torch.float32)
    losses=torch.zeros((n_parameters,1))

    return parameters,losses,directions,(reference_point,axis_0,axis_1)


def grid_walk(n : int ,
                callback_hyperparameters : Dict):
    
    
    projector=callback_hyperparameters["projector"]
    # n_steps=callback_hyperparameters["steps"]
    
    #Coarse
    n_steps=int(np.sqrt(n))//4
    xs = torch.linspace(-100, 100, steps=n_steps).flatten()
    ys = torch.linspace(-100, 100, steps=n_steps).flatten()
    
    xs,ys=torch.meshgrid(xs,ys)
    directions_c=torch.concatenate((xs.reshape(xs.shape+(1,)),ys.reshape(ys.shape+(1,))),dim=2)


    #Fine
    # n_steps=int(np.sqrt(n))
    xs = torch.linspace(-1, 1, steps=n_steps).flatten()
    ys = torch.linspace(-1, 1, steps=n_steps).flatten()
    xs,ys=torch.meshgrid(xs,ys)

    # xs,ys=torch.meshgrid(xs,ys)
    directions_f=torch.concatenate((xs.reshape(xs.shape+(1,)),ys.reshape(ys.shape+(1,))),dim=2)

    directions=torch.concatenate([directions_c,directions_f],dim=0)

    


    
    xs=directions[:,:,0].flatten()
    ys=directions[:,:,1].flatten()

    if projector is not None:
        reference_point=projector.cpu().reference_point.data
        axis_0,axis_1=projector.cpu().x.data,projector.y.data
    else:
        
        optima_points=callback_hyperparameters["optima"]
        reference_point=optima_points[0]#torch.mean(optima_points,dim=0)
        axis_0,axis_1=optima_points[0]-optima_points[1],optima_points[0]-optima_points[2]
        axis_0,axis_1=axis_0/torch.linalg.norm(axis_0),axis_1/torch.linalg.norm(axis_1)

    
    
    
    out_shape=(xs.shape[0],reference_point.shape[0])
    D=reference_point.shape[0]
    

    
    # directions=lr*directions

    #Get the parameters
    parameters=reference_point.reshape(1,D) + \
                xs.reshape(-1,1)*axis_0.reshape(1,D) + \
                ys.reshape(-1,1)*axis_1.reshape(1,D)
    
    n_parameters=out_shape[0]
    parameters=torch.tensor(parameters.reshape(out_shape),dtype=torch.float32)
    losses=torch.zeros((n_parameters,1))

    return parameters,losses,directions,(reference_point,axis_0,axis_1),Projector(reference_point=reference_point, x=axis_0,y=axis_1)

def random_dimension_reducer_sampler(n : int ,
                            dataloader : data.DataLoader,
                            Phi : torch.nn.Module,
                            theta_mask : torch.Tensor,
                            callback_hyperparameters : Dict ,
                            rescaling_func : Union[None,Callable]=None,
                            rescaling_func_hyperparameters : Union[None,Dict] = {}):
    """
    This sampler is particular
    It takes as inputs a projector function , usally a PLS dim reducer to dim 2, a starting point for sampling
    n : int : The number of sample to generate
    dataloader : data.DataLoader : The dataloader with all the data
    Phi : torch.nn.Module : The model to condition the fit on
    theta_mask : torch.tensor : The mask of the parameters to sample
    rescaling_func : Callable : a function that transform the data
    rescaling_func_hyperparameters : the hyperparameters needed for rescaling_func
    """
    parameters,losses,directions,(reference_point,axis_0,axis_1),projector=grid_walk(n=n,callback_hyperparameters=callback_hyperparameters)
    n_parameters=parameters.shape[0]


    with torch.no_grad():
        for i in tqdm(range(n_parameters)):
            loss=get_parameters_loss(parameter=parameters[i,:],model=Phi,dataloader=dataloader)
            losses[i]=loss
    # print(list(zip(targets_values,losses)))

    if rescaling_func is not None:
        rescaled_losses,rescaling_func_hyperparameters=rescaling_func(losses,rescaling_func_hyperparameters)
    else:
        rescaled_losses=None
        rescaling_func_hyperparameters=None

    # Map everything to torch tensor
    if not isinstance(rescaled_losses,torch.Tensor):
        rescaled_losses=torch.tensor(rescaled_losses)
    for key,value in rescaling_func_hyperparameters.items():
        if not isinstance(value,torch.Tensor):
            if value is not None:
                rescaling_func_hyperparameters[key]=torch.tensor(value)

    fig,ax=plt.subplots()
    # for i in range(n):
    # .reshape(n,n_steps,1)
    # SC=ax.plot(directions.reshape(-1,2)[:,0],directions.reshape(-1,2)[:,1])
    
    SC=ax.scatter(directions.reshape(-1,2)[:,0],directions.reshape(-1,2)[:,1],s=50,c=losses.cpu().numpy()[:,0],cmap="viridis")
    CB = fig.colorbar(SC, shrink=0.8)
    exp=callback_hyperparameters["exp"]
    plt.savefig(f"./sampling_N_{exp}.png")



    # projector(parameter=parameters[0])

    results={"parameters":parameters,
             "losses":losses,
             "rescaled_losses":rescaled_losses,
             "rescaling_hyperparameters": rescaling_func_hyperparameters,
             "projected_parameters":torch.tensor(directions,dtype=torch.float32).reshape(-1,2),
             "projector":projector}



    
    
    return results






class EfficientSampler:
    def __init__(self,
                dataloader : data.DataLoader,
                Phi : torch.nn.Module,
                theta_mask : torch.tensor ):
        """
        Args:
            - dataloader : the dataloader with the data to condition sample on
            - Phi : The architecture of the model we wish to sample parameters with his current parameters
            - theta_mask : the mask of Phi parameters which should be sampled. This allows to handle fixed parameters
        """

        if not isinstance(dataloader,data.DataLoader):
            raise ValueError


        self.dataloader=dataloader
        if theta_mask is None:

            mask=extract_parameters(model=Phi)
            # mask=torch.tensor(mask)>0
            theta_mask=torch.ones(len(mask),dtype=torch.bool)

        self.mask=theta_mask
        self.model_reference= Phi

            

    def sample(self, n,
                callback=random_sampler_callback,
                callback_hyperparameters={},
                rescaling_func=None,
                rescaling_func_hyperparameters={"mean":None,"std":None,"lambda":None,"min":None,"max":None}):
        """
        This functions takes in inputs a list of trajectories of training
        And augment it to generate N samples
        """
        # print("AUGMENTATION NOT IMPLEMENTED YET , ")

        #Get a triangle sampling
        results=callback(n=n,
                        dataloader=self.dataloader,
                        Phi=self.model_reference,
                        callback_hyperparameters = callback_hyperparameters,
                        theta_mask=self.mask,
                        rescaling_func =rescaling_func,
                        rescaling_func_hyperparameters=rescaling_func_hyperparameters)
        return results



if __name__== "__main__":
    #Create dummy dataloader

    x_size=2
    # torch.manual_seed(194)
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import math
    class simpleDataset(torch.utils.data.Dataset):
        def __init__(self,n_clusters=10,n_data=100):

            alpha=np.linspace(0,1,n_clusters)

            tetha=-0*alpha+(1-alpha)*np.pi
            r=1*alpha+(1-alpha)*10
            X=[]
            y=[]
            c=list(mcolors.TABLEAU_COLORS)
            datum=0.5*np.random.randn(n_data,x_size)
            for i in range(n_clusters):
                x=np.array([r[i]*np.cos(tetha[i]),r[i]*np.sin(tetha[i])])
                print(x)
                x=datum+x
                X.append(x)
                y.append(np.ones(n_data)*i)
            #     plt.scatter(x[:,0],x[:,1],c=c[i])
            #     plt.grid(True)
            # plt.show()
            self.X=np.concatenate(X)
            self.y=np.concatenate(y)
                

        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):

            x=torch.tensor(self.X[idx,:],dtype=torch.float32)
            y=torch.tensor(self.y[idx],dtype=torch.long)
            return x, y
    

    

    dataloader=data.DataLoader(simpleDataset(),batch_size=50,shuffle=True)


    #Create dummy mode

    class CosineClassifier(nn.Module):
        def __init__(self, in_features, n_classes, sigma=True):
            super(CosineClassifier, self).__init__()
            self.in_features = in_features
            self.out_features = n_classes
            self.weight = nn.Parameter(torch.Tensor(n_classes, in_features))
            if sigma:
                self.sigma = nn.Parameter(torch.Tensor(1))
            else:
                self.register_parameter('sigma', None)
            self.reset_parameters()

        def reset_parameters(self):
            with torch.no_grad():
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.uniform_(-stdv, stdv)
                if self.sigma is not None:
                    self.sigma.data.fill_(1)  #for initializaiton of sigma

        def forward(self, input):
            out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
            if self.sigma is not None:
                out = self.sigma * out
            # return out
            return {"logits":out}

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
                        nn.ReLU(),
                        nn.Dropout(p=0.1))
        

        def forward(self, x):
            return {"logits":self.model(x)}

    # Sample some data

    # model=LandScapeModel(input_dim=x_size,out_dimension=5)
    model=CosineClassifier(in_features=x_size,n_classes=10)
    model=model.to("cuda:0")

    mask=extract_parameters(model=model)
    # mask=torch.tensor(mask)>0
    mask=torch.ones(len(mask),dtype=torch.bool)
    sampler=EfficientSampler(dataloader=dataloader,Phi=model,theta_mask=None)
    
    n=100

    # Define the sampler callback
    rescaling_func=identity
    results=sampler.sample(n,callback=stratified_random_sampler_callback,
                           callback_hyperparameters={},
                           rescaling_func= rescaling_func,
                           rescaling_func_hyperparameters={"mean":None,"std":None,"lambda":None,"min":None,"max":None})
    
    anchors=results["parameters"]
    anchors_losses=results["rescaled_losses"]
    anchors_raw_losses=results["losses"]
    rescaling_func_hyperparameters=results["rescaling_hyperparameters"]


    # range_tail_losses=[anchors_losses.min().item()/2.0,torch.quantile(anchors_losses,0.25).item()]
    range_tail_losses=[1.91,1.911]
    range_tail_losses=[0.01,0.011]
    results_imposed=None
    impose_results=True

    n_imposed=3
    training_epochs=100


    if impose_results:
        results_imposed=sampler.sample(n_imposed,callback=retrain_sampler_callback,
                            callback_hyperparameters={"range":range_tail_losses,"epochs":training_epochs,"lr":1e-2},
                            rescaling_func= rescaling_func,
                            rescaling_func_hyperparameters=rescaling_func_hyperparameters)

    if results_imposed is not None:
        anchors_losses=torch.concatenate([anchors_losses,results_imposed["rescaled_losses"]])
        anchors = torch.concatenate([anchors,results_imposed["parameters"]])
        anchors_raw_losses = torch.concatenate([anchors_raw_losses,results_imposed["losses"]])

        anchors=torch.tensor(anchors)
        anchors_losses=torch.tensor(anchors_losses)
        anchors_raw_losses=torch.tensor(anchors_raw_losses)


    

    
    
    # plt.hist(xt,bins=n//4)
    # plt.show()
    # distances_map=torch.cdist(samples,samples,p=2)
    # plt.imshow(distances_map)
    # plt.show()
   
    from approximators import BasicKernelAprroximator, AEMLPApproximator,\
    BasicMLPAprroximator, \
    BasicMLPModule, \
    AEMLPModule, \
    Epanechnikov_kernel, \
    IdentityKernel, \
    TriangleKernel, \
    active_anchors_choice

    # approximator=BasicKernelAprroximator(theta_refs=anchors,
    #                                      theta_refs_raw_losses=anchors_raw_losses,
    # 
    #                                      theta_refs_losses=anchors_losses)
    X_train, X_test, y_train, y_test=train_test_split(anchors,anchors_raw_losses,test_size=0.1)

    approx_simple=False
    if not approx_simple:
        epochs=400
        approximator=AEMLPApproximator(theta_refs=X_train,
                                            theta_refs_raw_losses=y_train,
                                            callback_hyperparameters={"epochs":epochs,
                                                    "lr":1e-3,
                                                    "mlp_model":AEMLPModule,
                                                    "optimizer":torch.optim.Adam})
    if approx_simple:
        approximator=BasicMLPAprroximator(theta_refs=X_train,
                                        theta_refs_raw_losses=y_train,
                                        callback_hyperparameters={"epochs":2000,
                                                "lr":1e-3,
                                                "mlp_model":BasicMLPModule,
                                                "optimizer":torch.optim.Adam})


    #Calibrate this kernel h
    if False:
        n_h=1000
        kernel=TriangleKernel
        results=sampler.sample(n_h,callback=stratified_random_sampler_callback,
                            rescaling_func= rescaling_func,
                            rescaling_func_hyperparameters=rescaling_func_hyperparameters)

        calibration_samples=results["parameters"]
        calibration_targets=results["rescaled_losses"]


        h=approximator.calibrate_h(calibration_samples= calibration_samples , 
                                calibration_targets= calibration_targets, 
                                method= "knn", 
                                method_hyperparameters={"min_nbrs_neigh":10,"kernel":kernel})
        approximator.set_rescaling_parameters(rescaling_func,parameters=rescaling_func_hyperparameters)

    # Improve 

    improve=False
    if improve:

        

        sampler_hyperparameters ={"random_sampler_callback":random_sampler_callback,
                                "rescaling_func":rescaling_func,
                                "rescaling_func_hyperparameters":rescaling_func_hyperparameters}
                            

        approximator=active_anchors_choice(approximator  = approximator,
                            approximator_hyperparameters = {"kernel":kernel},
                            sampler = sampler,
                            sampler_hyperparameters =sampler_hyperparameters,
                            n_anchors_max = 2*n,
                            n_rounds_improve =2,
                            n_targets_neighboors = 3)
    
    # Test the prediction
        
    fig = plt.figure(figsize=plt.figaspect(2.))
    ax= fig.add_subplot(2, 2, 1)
    # plt.subplots(2,1,subplot_kw={"projection": "3d"})
    # plt.grid(True)
    # plt.xlim(1.5,2.5)
    # plt.ylim(1.5,2.5)
    # Create linear regression object
    regr = linear_model.LinearRegression(fit_intercept=True)
    

   

    #Training predictions
    test_targets=y_train
    test_predictions= approximator(X_train.to("cuda:0"))
    pred_train=torch.squeeze(test_predictions["pred"]).detach().cpu().numpy()
    true_train=torch.squeeze(test_targets).detach().numpy()

    # Train the model using the training sets
    regr.fit(true_train.reshape(-1,1), pred_train)
    # Make predictions using the testing set
    line_train = regr.predict(true_train.reshape(-1,1))
    coef=regr.coef_
    r2_= r2_score(pred_train, line_train)
    ax.scatter(true_train,pred_train)
    ax.plot(true_train, line_train, color="red", linewidth=3)
    ax.set_xlabel(xlabel="true")
    ax.set_ylabel(ylabel="pred")
    ax.set_title(f"Train data --  slope : {coef} -  R2 : {r2_}")

    # True tests predictions
    # n_test=anchors.shape[0]
    # results=sampler.sample(n_test,callback=stratified_random_sampler_callback,
    #                        rescaling_func= rescaling_func,
    #                        rescaling_func_hyperparameters=rescaling_func_hyperparameters)
    
    # test_samples=results["parameters"]
    # test_targets=results["rescaled_losses"]
    test_predictions= approximator(X_test.to("cuda:0"))
    pred_test=torch.squeeze(test_predictions["pred"]).detach().cpu().numpy()
    true_test=torch.squeeze(y_test).detach().numpy()

    # Train the model using the training sets
    regr.fit(true_test.reshape(-1,1), pred_test)
    # Make predictions using the testing set
    line_test = regr.predict(true_test.reshape(-1,1))
    coef=regr.coef_
    ax= fig.add_subplot(2, 2, 2)
    r2_= r2_score(pred_test, line_test)
    ax.scatter(true_test,pred_test)
    ax.plot(true_test, line_test, color="red", linewidth=3)
    ax.set_xlabel(xlabel="true")
    ax.set_ylabel(ylabel="pred")
    ax.set_title(f"Test data --  slope : {coef} -  R2 : {r2_}")
    fig.suptitle('Pred vs True ')



    # Show the loss function ( apply only if embedding is 2d)
    # Second subplot
    ax = fig.add_subplot(2, 1, 2, projection='3d')

    X = np.arange(-2.0, 2.0, 0.1)
    Y = np.arange(-2.0, 2.0, 0.1)
    X, Y = np.meshgrid(X, Y)
    with torch.no_grad():
        X_=torch.tensor(X,dtype=torch.float32)
        X_=torch.reshape(X_,X.shape+(1,))
        Y_=torch.tensor(Y,dtype=torch.float32)
        Y_=torch.reshape(Y_,Y_.shape+(1,))
        data=torch.concatenate((X_,Y_),dim=-1)

        Z=approximator.network.predHead(data.reshape(-1,2).to("cuda:0"))
        Z=torch.squeeze(Z.reshape(X_.shape)).detach().cpu().numpy()


    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        linewidth=0, antialiased=False)
    # ax.set_zlim(-1, 1)
    xs,ys,zs=test_predictions["encoding"][:,0],test_predictions["encoding"][:,1],true_test
    xs,ys=torch.squeeze(xs.detach()).cpu().numpy(),torch.squeeze(ys.detach()).cpu().numpy()

    ax.scatter(xs, ys, zs,c="tab:orange",s=22, cmap="viridis",alpha=0.7)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')



    plt.show()
    print()

    


