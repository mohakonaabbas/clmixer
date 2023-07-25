from typing import Callable,Any, Union
from base_plugin import Operation
from torch.nn import functional as F
from imbalance import BiC,WA
from storage import Storage
import numpy as np
import torch
from datasets.base import simpleDataset
from torch.utils import data
import copy
from losses.distillation import pod, dirichlet_uncertain
from tqdm import tqdm
#============================== LOSSES ===============================#
# DONE
class CrossEntropyOperation(Operation):
    def __init__(self, name = "cross_entropy", 
                 entry_point ="before_backward",
                  inputs=None,
                callback=(lambda x:x), 
                paper_ref="",
                is_loss=True):
        super().__init__(name, entry_point, inputs, callback, paper_ref,is_loss)
    
        self.set_callback(self.ce_callback)


    def ce_callback(self,reduction="mean"):
        """
        Cross entropy function
        """

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
# DONE
class KnowledgeDistillationOperation(Operation):
    def __init__(self, name = "knowledge_distillation", 
                entry_point ="before_backward",
                inputs={},
            callback=(lambda x:x), 
            paper_ref="",
                is_loss=True):
        super().__init__(name, entry_point, inputs, callback, paper_ref,is_loss)

        self.set_callback(self.kd_callback)


    def kd_callback(self,reduction="batchmean"):
        """
        Cross entropy function
        """
        
        try:
            assert self.inputs.old_logits is not None
        except AssertionError :
            return self.inputs
        logits=self.inputs.logits
        
        temperature=self.inputs.plugins_storage[self.name]["hyperparameters"]["temperature"]
        old_logits=self.inputs.old_logits
        task_mask=self.inputs.seen_classes_mask

        log_probs_new = (logits[:, task_mask] / temperature).log_softmax(dim=1)

        probs_old = (old_logits[:, task_mask] / temperature).softmax(dim=1)
        loss = F.kl_div(log_probs_new, probs_old, reduction=reduction)

        loss_coeff= sum(self.inputs.seen_classes_mask)/sum(self.inputs.task_mask)
        # loss_coeff= 1.0
        if reduction=="none":
            return loss.sum(dim=1)/loss.shape[1]
        self.inputs.loss+=loss_coeff*loss
        return self.inputs

#DONE
class PodLossOperation(Operation):
    def __init__(self, name = "pod_loss", 
                entry_point ="before_backward",
                inputs={},
            callback=(lambda x:x), 
            paper_ref="https://arxiv.org/pdf/2004.13513.pdf",
                is_loss=True):
        super().__init__(name, entry_point, inputs, callback, paper_ref,is_loss)

        self.set_callback(self.pod_callback)


    def pod_callback(self,reduction="batchmean"):
        """
        pod loss function
        """
        
        try:
            assert self.inputs.old_logits is not None
        except AssertionError :
            return self.inputs
        

        
        # New logits
        logits=self.inputs.logits
        attentions=self.inputs.attentions


        # Olds logits and attentions
        old_logits=self.inputs.old_logits
        old_attentions=self.inputs.old_attentions
        task_mask=self.inputs.task_mask

        loss=pod([attentions,logits],[old_attentions,old_logits])
        loss_coeff= sum(self.inputs.seen_classes_mask)/sum(self.inputs.task_mask)
        
        if reduction=="none":
            return loss
        

        self.inputs.loss+=loss_coeff*loss.mean()


        return self.inputs

class ProspectiveLossOperation(Operation):

    pass
# DONE
class DirichletUncertaintyLossOperation(Operation):
    def __init__(self, name = "dirichlet_uncertainty_loss", 
                entry_point ="before_backward",
                inputs={},
                callback=(lambda x:x), 
                paper_ref="https://arxiv.org/pdf/1806.01768.pdf",
                is_loss=True):
        super().__init__(name, entry_point, inputs, callback, paper_ref,is_loss)

        self.set_callback(self.dir_uncert_callback)
    
    def dir_uncert_callback(self,reduction="mean"):
        """
        dirichlet_uncertainty loss function
        """
        
        # New logits
        logits=self.inputs.logits
        targets=self.inputs.targets
        kl_annealing_step=self.inputs.plugins_storage[self.name]["hyperparameters"]["kl_annealing_step"]
        epoch=self.inputs.current_epoch
        regressor_type=self.inputs.plugins_storage[self.name]["hyperparameters"]["regressor"]
        
        

        loss=dirichlet_uncertain(y_pred= logits,
                           y= targets,
                           epoch= epoch,
                           kl_annealing_step= kl_annealing_step,
                           type=regressor_type
                           )
        



        # loss_coeff= sum(self.inputs.seen_classes_mask)/sum(self.inputs.task_mask)
        loss_coeff=1.0
        
        if reduction=="none":
            return loss
        

        self.inputs.loss+=loss_coeff*loss.mean()
        return self.inputs

class RetrospectiveLossOperation(Operation):

    pass

class MDLLossOperation(Operation):

    pass

class SparsityLossOperation(Operation):

    pass
#============================== NETWORK ===============================#
#DONE
class FinetuneOperation(Operation):
    """
    Finetune the last layer of a neural network
    """
    def __init__(self, name = "finetune_network", 
                 entry_point ="after_training_exp",
                  inputs={},
                callback=(lambda x:x), 
                paper_ref=""):
        super().__init__(name, entry_point, inputs, callback, paper_ref)

        self.set_callback(self.finetune_callback)

    def finetune_callback(self):
        """
        Finetune the last layer function
        """

        #Get the current model
        network=self.inputs.current_network

        # Freeze the backbone layers
        network,nets_trainables=network.freeze_backbone(state=True,nets_trainables=[])
        # Create a balanced dataset
        epochs=self.inputs.plugins_storage[self.name]["hyperparameters"]["finetune_epochs"]
        bs=self.inputs.plugins_storage[self.name]["hyperparameters"]["finetune_bs"]
        lr=self.inputs.plugins_storage[self.name]["hyperparameters"]["finetune_lr"]
        
        RandomMemoryUpdaterOperation.random_callback(self)

        # Retrain the last layer

            # Assign the memory to the balanced dataset

        loader = data.DataLoader(simpleDataset(X=self.inputs.dataloader.dataset.activated_files_subset_memory,
                                                        y= self.inputs.dataloader.dataset.activated_files_labels_subset_memory,
                                                        predictor=self.inputs.dataloader.dataset.backbone),
                                                        batch_size=bs,
                                                        shuffle=True)
        
        loss_criterion = F.cross_entropy
        network.train()
        loss=0.0
        optimizer = torch.optim.SGD(network.classifier.parameters(), lr=lr, momentum=0.9)
        for epoch in tqdm(range(epochs)):
            for inputs,targets in loader:
                inputs=inputs.to(self.inputs.device)
                targets=targets.to(self.inputs.device)
                outputs=network(inputs)
                loss = loss_criterion(outputs["logits"], targets)
                # print(f"Finetuning loss :  {loss.item()}")
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        


        # Unfreeze 
        self.inputs.current_network=network
       
        
        self.inputs.current_network.freeze_backbone(state=False,nets_trainables=nets_trainables)

        self.inputs.current_network.train()
        return self.inputs

        
     
#DONE
class DuplicateNetworkBackboneOperation(Operation):
    """
    Duplicate the neural network
    """
    def __init__(self, name = "duplicate_network", 
                 entry_point ="before_training_exp",
                  inputs={},
                callback=(lambda x:x), 
                paper_ref=""):
        super().__init__(name, entry_point, inputs, callback, paper_ref)

        self.set_callback(self.duplicate_callback)

    def duplicate_callback(self):
        """
        duplicate function
        """

        
        # Get the current model
        network=self.inputs.current_network
        # Only upgrade the model if we have more than one class
        if network.ntask == 0:
            return self.inputs
        # Freeze the backbone layers
        network,_=network.freeze_backbone(state=True,nets_trainables=[])

        # Duplicate the network
        network._add_classes_multi_backbone()
        self.inputs.current_network=network


        return self.inputs
    
class GrowNewOutputsOperation(Operation):
    """
    Grow a new output when needed
    The network is from a partiular class
    """
    def __init__(self, name = "grow_network", 
                 entry_point ="before_training_exp",
                  inputs={},
                callback=(lambda x:x), 
                paper_ref=""):
        super().__init__(name, entry_point, inputs, callback, paper_ref)
        self.set_callback(self.grow_callback)

    def grow_callback(self):
        
        
        network=self.inputs["network"]
        n_classes=self.inputs["n_classes"]
        network=self.inputs["network"]
        n_classes=self.inputs["n_classes"]
        network._add_classes_single_fc(n_classes)

        return {"network":network}

class PruneNetworkOperation(Operation):
    """
    Prune network weights with a mask
    """
    def __init__(self, name = "prune_network", 
                 entry_point ="after_training_exp",
                  inputs={},
                callback=(lambda x:x), 
                paper_ref=""):
        super().__init__(name, entry_point, inputs, callback, paper_ref)

        pass


#======================= INITIALISATION ==========================

class CoTransportInitialisationOperation(Operation):
    def __init__(self, name = "aligned_init", 
                 entry_point ="before_training_exp",
                  inputs={},
                callback=(lambda x:x), 
                paper_ref=""):
        super().__init__(name, entry_point, inputs, callback, paper_ref)

        pass

class LwFWarmInitialisation(Operation):
    def __init__(self, name = "lfw_init", 
                 entry_point ="before_training_exp",
                  inputs={},
                callback=(lambda x:x), 
                paper_ref=""):
        super().__init__(name, entry_point, inputs, callback, paper_ref)

        pass


#====================== REGULARISATION TRICKS ==========================
#DONE
class BICOperation(Operation):
    def __init__(self, name = "BIC", 
                entry_point =["after_training_exp","after_eval_forward"],
                inputs={},
            callback=(lambda x:x), 
            paper_ref=""):
        super().__init__(name, entry_point, inputs, callback, paper_ref)

        self.set_callback(self.bic_callback)
        self.bic=None


    def bic_callback(self):
        """
        bic function
        """

        if self.inputs.stage_name=="after_eval_forward":
            #Apply WA transform on data
            self.inputs.logits=self.bic.post_process(self.inputs.logits)
        
        elif self.inputs.stage_name=="after_training_exp":
            epochs=self.inputs.plugins_storage[self.name]["hyperparameters"]["bic_epochs"]
            lr=self.inputs.plugins_storage[self.name]["hyperparameters"]["bic_lr"]
            wd=self.inputs.plugins_storage[self.name]["hyperparameters"]["bic_wd"]
            bs=self.inputs.plugins_storage[self.name]["hyperparameters"]["bic_bs"]
            lr_decay=self.inputs.plugins_storage[self.name]["hyperparameters"]["bic_decay"]
            # scheduling=self.inputs.plugins_storage[self.name]["hyperparameters"]["bic_scheduling"]

            

            # Create the memory in a balanced way
            RandomMemoryUpdaterOperation.random_callback(self)

            # Assign the memory to the balanced dataset

            bic_loader = data.DataLoader(simpleDataset(X=self.inputs.dataloader.dataset.activated_files_subset_memory,
                                                            y= self.inputs.dataloader.dataset.activated_files_labels_subset_memory,
                                                            predictor=self.inputs.dataloader.dataset.backbone),
                                                            batch_size=bs,
                                                            shuffle=True)
            
            current_model=self.inputs.current_network
            current_model.eval()


            if self.bic is None:
                self.bic=BiC(lr=lr,lr_decay_factor=lr_decay,weight_decay=wd,batch_size=bs,epochs=epochs)

            self.bic.reset()
            self.bic.update(current_model,bic_loader)
            result={"beta": self.bic.beta, "gamma": self.bic.gamma}

            self.inputs.plugins_storage[self.name].update(result)

            current_model.train()

        return self.inputs
#DONE
class WeightAlignOperation(Operation):
    def __init__(self, name = "WA", 
                entry_point =["after_training_exp","after_eval_forward"],
                inputs={},
            callback=(lambda x:x), 
            paper_ref=""):
        super().__init__(name, entry_point, inputs, callback, paper_ref)


        self.set_callback(self.wa_callback)
        self.wa=WA()

    def wa_callback(self):


        try:
            assert self.inputs.old_logits is not None
        except AssertionError :
            return self.inputs
        
        # wa=WA()

        if self.inputs.stage_name=="after_eval_forward":
            #Apply WA transform on data
            self.inputs.logits=self.wa.post_process(self.inputs.logits,self.inputs.task_mask)
            
        elif self.inputs.stage_name=="after_training_exp":
            network=self.inputs.current_network
            task_mask=self.inputs.seen_classes_mask
            result=self.wa.update(network.classifier,task_mask)

            self.inputs.plugins_storage[self.name].update({"gamma":wa.gamma}
    )

        return self.inputs


#======================= DATASETS TRICKS ==========================
#DONE
class RandomMemoryUpdaterOperation(Operation):
    def __init__(self, name= "random_memory", 
                entry_point= "after_training_exp", 
                inputs= None, 
                callback= (lambda x:x), 
                paper_ref="", is_loss=False):
        super().__init__(name, entry_point, inputs, callback, paper_ref, is_loss)
    
        self.set_callback(self.random_callback)
       
    def random_callback(self):
        """
        This callback create a random memory
        # Args : budget ( Nbrs of imgs per cls)
        It uses the task_mask to get all the currently seen classes

        """
        #Get the list
        dataset_lbls=self.inputs.dataloader.dataset.activated_files_labels_subset
        dataset=self.inputs.dataloader.dataset.activated_files_subset
        # Get lbls name
        present_lbs_name=np.argwhere(np.array(self.inputs.task_mask)==True).squeeze().tolist()
        if type(present_lbs_name) != list : present_lbs_name=[present_lbs_name]

        n_budget=self.inputs.plugins_storage[self.name]["hyperparameters"]["cls_budget"]
        memory=[]
        memory_lbl=[]

        # For each label
        for lbl in present_lbs_name:
        #Get the mask representing only the considered label

            mask=np.array(dataset_lbls)==lbl
            #Get the labels
            subset=np.array(dataset_lbls)[mask]
            # Shuffle them
            n=len(subset)
            random_order=np.random.choice(np.arange(n),n)

            # Cut to the imposed budget

            n_cut=min(n,n_budget)
            memory+=(np.array(dataset)[mask][random_order][:n_cut]).tolist()
            memory_lbl+=(np.array(dataset_lbls)[mask][random_order][:n_cut]).tolist()

            #Update the storage

        self.inputs.dataloader.dataset.activated_files_subset_memory = memory
        self.inputs.dataloader.dataset.activated_files_labels_subset_memory= memory_lbl



        return self.inputs

#DONE
class NCMMemoryUpdaterOperation(Operation):
    def __init__(self, name= "ncm_memory", 
                entry_point= "after_training_exp", 
                inputs= None, 
                callback= (lambda x:x), 
                paper_ref="https://arxiv.org/abs/1611.07725", 
                is_loss=False):
        super().__init__(name, entry_point, inputs, callback, paper_ref, is_loss)
    
        self.set_callback(self.ncm_callback)
       
    def ncm_callback(self):
        """
        This callback create a ncm memory according to icarl
        # Args : budget ( Nbrs of imgs per cls)
        It uses the task_mask to get all the currently seen classes

        For all the data in a class , compute the prototypes of the classes
        Do an ordered list of exemplar according to 

        """

        

        # Get lbls name
        present_lbs_name=np.argwhere(np.array(self.inputs.task_mask)==True).squeeze().tolist()
        if type(present_lbs_name) != list : present_lbs_name=[present_lbs_name]

        # List of data
        x_subsets={}
        x_path_subsets={}
        for lbl in present_lbs_name:
            x_subsets[lbl]=[]
            x_path_subsets[lbl]=[]

        #Get the dataloader


        with torch.no_grad():
            for inputs,targets in self.inputs.dataloader:
                # Get the inputs and outputs
                outputs=self.inputs.current_network(inputs)

                # Map to the right placeholder the outputs
                for lbl in present_lbs_name:
                    lbl_mask=targets==lbl
                    x_subsets[lbl].append(outputs["attentions"][lbl_mask])
                    x_path_subsets[lbl].append(np.array(self.inputs.dataloader.dataset.current_batch_paths)[lbl_mask])
                self.inputs.dataloader.dataset.current_batch_paths=[]
        # For each placeholder , put it in a vector size
        X={}
        X_path={}
        for key, val in x_subsets.items():
            X[key]=torch.concatenate(val)

        for key, val in x_path_subsets.items():
            X_path[key]=np.concatenate(val)



        # NCM Part, compute the ncm

        n_budget=self.inputs.plugins_storage[self.name]["hyperparameters"]["cls_budget"]
        memory=[]
        memory_lbl=[]
        

        for key, data in X.items():
            # For each key

            mu=data.mean(dim=0) # Get the mean
            chosen_samples=[False]*data.shape[0]
            order=np.arange(len(chosen_samples))
            n_cut=min(data.shape[0],n_budget)

            if data.shape[0]<=n_budget:
               memory+=(X_path[key]).tolist()
               memory_lbl+=[key]*len(memory)
            else:
                for k in range(1,n_cut):
                    unchosen_samples=list(map(lambda x: not x,chosen_samples))
                    # Read the paper to understand the following lines
                    data_k=data[unchosen_samples]/k
                    phi_pk=torch.sum(data[chosen_samples],dim=0)/k

                    obj=mu-data_k-phi_pk


                    loss=torch.norm(input=obj, dim=1)
                    index=torch.argmin(loss)
                    order_=order[unchosen_samples]
                    chosen_samples[order_[index]]=True
                
                memory+=(X_path[key][chosen_samples]).tolist()
                memory_lbl+=[key]*len(memory)


        self.inputs.dataloader.dataset.activated_files_subset_memory = memory
        self.inputs.dataloader.dataset.activated_files_labels_subset_memory= memory_lbl



        return self.inputs

#DONE
class MIRMemoryUpdaterOperation(Operation):
    def __init__(self, name= "mir_memory", 
                entry_point= ["before_backward","after_training_exp"], 
                inputs= None, 
                callback= (lambda x:x), 
                paper_ref="https://arxiv.org/abs/1908.04742", 
                is_loss=False):
        super().__init__(name, entry_point, inputs, callback, paper_ref, is_loss)
    
        self.set_callback(self.mir_callback)
       
    def mir_callback(self):
        """
        This callback create a ncm memory according to icarl
        # Args : budget ( Nbrs of imgs per cls)
        It uses the task_mask to get all the currently seen classes

        For all the data in a class , compute the prototypes of the classes
        Do an ordered list of exemplar according to 

        """
        if self.inputs.stage_name=="after_training_exp":
            RandomMemoryUpdaterOperation.random_callback(self)
            
        elif self.inputs.stage_name=="before_backward":
            try:
                assert self.inputs.old_network is not None
            except AssertionError :
                return self.inputs

            

            # Get lbls name
            present_lbs_name=np.argwhere(np.array(self.inputs.task_mask)==True).squeeze().tolist()
            if type(present_lbs_name) != list : present_lbs_name=[present_lbs_name]

            # List of data
            x_subsets={}
            x_path_subsets={}
            for lbl in present_lbs_name:
                x_subsets[lbl]=[]
                x_path_subsets[lbl]=[]

            #Get the dataloader
            batch_size_mem=len(self.inputs.dataloader.dataset.activated_files_labels_subset_memory)

            memory_dataloader = data.DataLoader(simpleDataset(X=self.inputs.dataloader.dataset.activated_files_subset_memory,
                                                            y= self.inputs.dataloader.dataset.activated_files_labels_subset_memory,
                                                            predictor=self.inputs.dataloader.dataset.backbone),
                                                            batch_size=batch_size_mem,
                                                            shuffle=True)
            
            
            batch_size_mem=np.random.randint(0,batch_size_mem)
            
            def update_temp(model, grad, lr):
                model_copy = copy.deepcopy(model)
                for g, p in zip(grad, model_copy.parameters()):
                    if g is not None:
                        p.data = p.data - lr * g
                return model_copy

            def cycle(loader):
                while True:
                    for batch in loader:
                        yield batch
            #Compute the gradients
                
            gradients = torch.autograd.grad(
                self.inputs.loss,
                self.inputs.current_network.parameters(),
                retain_graph=True,
                allow_unused=True,
            )

            # Compute the model virtual update

            virtually_updated_model=update_temp(self.inputs.current_network,gradients,self.inputs.lr)
            self.inputs.current_network.eval()
            virtually_updated_model.eval()

            with torch.no_grad():
                inputs, targets = next(cycle(memory_dataloader))

                # for inputs,targets in memory_dataloader:
                # Get the inputs and outputs
                inputs=inputs.to(self.inputs.device)
                targets=targets.to(self.inputs.device)

                #  Perform the temporary update with current data

                # Gather all losses
                old_loss,new_loss=0.0,0.0
                old_output = self.inputs.old_network(inputs)
                next_output = virtually_updated_model(inputs)
                current_output=self.inputs.current_network(inputs)

                for plugin in self.inputs.activated_plugins:
                    # Create a shallow copy of the plugin
                    temp_plugin=type(plugin)() # Create a empty plugin of same type to poulate and avoid references issues
                    temp_storage=Storage()
                    temp_storage.plugins_storage=self.inputs.plugins_storage
                    temp_storage.seen_classes_mask=self.inputs.seen_classes_mask
                    temp_storage.task_mask=self.inputs.task_mask
                    


                    

                    if plugin.is_loss:
                        reduction = "none"
                        temp_storage.old_logits=old_output["logits"]
                    
                        temp_storage.logits=current_output["logits"]
                        temp_storage.targets=targets
                        temp_plugin.set_inputs(temp_storage)
                    
                        plugin_loss= temp_plugin.callback(reduction)
                        old_loss+=plugin_loss
                        
                        temp_storage.logits=next_output["logits"]
                        temp_plugin.set_inputs(temp_storage)

                        new_loss += temp_plugin.callback(reduction)

                        loss_diff = new_loss - old_loss
            
            chosen_samples_indexes = torch.argsort(loss_diff)[len(inputs) - batch_size_mem :]
                

            # Choose the samples and add their loss to the current loss
            chosen_samples_x, chosen_samples_y = (inputs[chosen_samples_indexes],targets[chosen_samples_indexes])
            replay_output = self.inputs.current_network(chosen_samples_x)
            replay_loss=0.0
            
            for plugin in self.inputs.activated_plugins:
                if plugin.is_loss:
                    plugin.callback()

            self.inputs.loss += replay_loss

                    
            self.inputs.current_network.train()


            # Populate the memory
            RandomMemoryUpdaterOperation.random_callback(self)
        return self.inputs




def return_plugin(name):

    """
    FUnction used to register and return a plugin
    """

    if name=="WA":
        return WeightAlignOperation
    elif name=="BIC":
        return BICOperation
    elif name=="lfw_init":
        return LwFWarmInitialisation
    elif name=="aligned_init":
        return CoTransportInitialisationOperation
    elif name=="freeze_network":
        return FreezePartialNetworkOperation
    elif name=="prune_network":
        return PruneNetworkOperation
    elif name=="grow_network":
        return GrowNewOutputsOperation
    elif name=="duplicate_network":
        return DuplicateNetworkBackboneOperation
    elif name=="finetune_network":
        return FinetuneOperation
    elif name=="knowledge_distillation":
        return KnowledgeDistillationOperation
    elif name=="cross_entropy":
        return CrossEntropyOperation
    elif name=="random_memory":
        return RandomMemoryUpdaterOperation
    elif name=="ncm_memory":
        return NCMMemoryUpdaterOperation
    elif name=="mir_memory":
        return MIRMemoryUpdaterOperation
    elif name=="pod_loss":
        return PodLossOperation
    elif name=="dirichlet_uncertainty_loss":
        return DirichletUncertaintyLossOperation
    
    


if __name__=="__main__":
    operation=Operation(name="CrossEntropyLoss",entry_point="before_training",inputs={},callback=(lambda x:x),paper_ref="")
    operation



    