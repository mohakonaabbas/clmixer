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
warnings.filterwarnings("ignore")
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
            inputs=inputs.to("cuda:0")
            targets=targets.to("cuda:0")

            outputs=network(inputs)
            loss = criterion(outputs["logits"].softmax(dim=1),targets)
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
    count=1
    loss=0.0
    model=model.to("cuda:0")
    for inputs, targets in dataloader:
        inputs=inputs.to("cuda:0")
        targets=targets.to("cuda:0")
        outputs=model(inputs)
        loss+=criterion(outputs["logits"].softmax(dim=1),targets)
        count+=1
    return loss/count


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




#CALLBACKS
def random_sampler_callback(n : int ,
                            dataloader : data.DataLoader,
                            Phi : torch.nn.Module,
                            theta_mask : torch.tensor,
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
    # coeffs=coeffs/torch.sum(coeffs,dim=1).reshape(-1,1)
    indices=torch.argwhere(theta_mask)
    indices=torch.squeeze(indices)
    parameters=torch.ones((n,dimension)) #*torch.tensor(extract_parameters(Phi))
    # parameters=torch.zeros((n,len(theta_mask)))
    parameters[:,indices]=coeffs

    losses=torch.zeros((n,1))

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



    results={"parameters":parameters,"losses":losses,"rescaled_losses":rescaled_losses,"rescaling_hyperparameters": rescaling_func_hyperparameters}


    
    
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
                            X : torch.tensor,
                            y : torch.tensor,
                            Phi : torch.nn.Module,
                            theta_mask : torch.tensor,
                            dataloader_bs : int = 32):
    """
     SAmpling by optimizing a value sampler of parameters
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
                        theta_mask=self.mask,
                        rescaling_func =rescaling_func,
                        rescaling_func_hyperparameters=rescaling_func_hyperparameters)
        return results



if __name__== "__main__":
    #Create dummy dataloader

    x_size=2
    torch.manual_seed(194)
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    class simpleDataset(torch.utils.data.Dataset):
        def __init__(self,n_clusters=5,n_data=100):

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


    #Create dummy model

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

    model=LandScapeModel(input_dim=x_size,out_dimension=10)
    model=model.to("cuda:0")

    mask=extract_parameters(model=model)
    # mask=torch.tensor(mask)>0
    mask=torch.ones(len(mask),dtype=torch.bool)
    sampler=EfficientSampler(dataloader=dataloader,Phi=model,theta_mask=None)
    
    n=100

    # Define the sampler callback
    rescaling_func=minmax
    results=sampler.sample(n,callback=random_sampler_callback,
                           rescaling_func= rescaling_func,
                           rescaling_func_hyperparameters={"mean":None,"std":None,"lambda":None,"min":None,"max":None})
    
    anchors=results["parameters"]
    anchors_losses=results["rescaled_losses"]
    anchors_raw_losses=results["losses"]
    rescaling_func_hyperparameters=results["rescaling_hyperparameters"]
    

    
    
    # plt.hist(xt,bins=n//4)
    # plt.show()
    # distances_map=torch.cdist(samples,samples,p=2)
    # plt.imshow(distances_map)
    # plt.show()
   
    from approximators import BasicKernelAprroximator,Epanechnikov_kernel,IdentityKernel,TriangleKernel, active_anchors_choice

    approximator=BasicKernelAprroximator(theta_refs=anchors,
                                         theta_refs_raw_losses=anchors_raw_losses,
                                         theta_refs_losses=anchors_losses)

    #Calibrate this kernel h
    n_h=100
    results=sampler.sample(n_h,callback=random_sampler_callback,
                           rescaling_func= rescaling_func,
                           rescaling_func_hyperparameters=rescaling_func_hyperparameters)

    calibration_samples=results["parameters"]
    calibration_targets=results["rescaled_losses"]


    h=approximator.calibrate_h(calibration_samples= calibration_samples , 
                             calibration_targets= calibration_targets, 
                             method= "knn", 
                             method_hyperparameters={"min_nbrs_neigh":5,"kernel":Epanechnikov_kernel})

    # Improve 

    approximator.set_rescaling_parameters(rescaling_func,parameters=rescaling_func_hyperparameters)

    sampler_hyperparameters ={"random_sampler_callback":random_sampler_callback,
                              "rescaling_func":rescaling_func,
                              "rescaling_func_hyperparameters":rescaling_func_hyperparameters}
                          

    approximator=active_anchors_choice(approximator  = approximator,
                          approximator_hyperparameters = {"kernel":Epanechnikov_kernel},
                          sampler = sampler,
                          sampler_hyperparameters =sampler_hyperparameters,
                          n_anchors_max = 1000,
                          n_rounds_improve =10,
                          n_targets_neighboors = 10)
    
    # Test the prediction

    n_test=100
    results=sampler.sample(n_h,callback=random_sampler_callback,
                           rescaling_func= rescaling_func,
                           rescaling_func_hyperparameters=rescaling_func_hyperparameters)
    
    test_samples=results["parameters"]
    test_targets=results["rescaled_losses"]

    test_predictions= approximator(test_samples)

    pred=torch.squeeze(test_predictions).detach().numpy()
    true=torch.squeeze(test_targets).detach().numpy()


    from sklearn import datasets, linear_model
    from sklearn.metrics import mean_squared_error, r2_score



    # Create linear regression object
    regr = linear_model.LinearRegression(fit_intercept=False)

    # Train the model using the training sets
    regr.fit(true.reshape(-1,1), pred)

    # Make predictions using the testing set
    line = regr.predict(true.reshape(-1,1))

    coef=regr.coef_
    r2_= r2_score(pred, line)

    plt.scatter(true,pred)
    plt.plot(true, line, color="red", linewidth=3)
    plt.grid(True)
    # plt.xlim(0,5)
    # plt.ylim(0,5)
    plt.xlabel(xlabel="true")
    plt.ylabel(ylabel="pred")

    plt.title(f" slope : {coef} \n R2 : {r2_}")
    plt.show()
    print()

    


