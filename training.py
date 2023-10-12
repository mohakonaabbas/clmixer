from plugins import return_plugin,Operation,EntryPoints
import json
import factory
import numpy as np
# from dataloader_vit import *
from datasets.base import BaseDataset
from torch.utils import data
from models import ExpandableNet
from storage import Storage
import copy
import metrics
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import os.path as osp
import sys
repo_name = 'clmixer'
base_dir = osp.realpath(".")[:osp.realpath(".").index(repo_name) + len(repo_name)]
sys.path.insert(0, base_dir)

from sacred import Experiment
ex=Experiment(base_dir=base_dir)

# MongoDB Observer
observe=False
if observe:
    from sacred.observers import MongoObserver
    ex.observers.append(MongoObserver.create(url='127.0.0.1:27017', db_name='representation_free'))
from typing import Union

# Register the json pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()
import jsonpickle


class Trainer:
    """
    This class aim at implementings the training of CL algorith in a modular way.
    It loads basics componnents such as optimizer and the plugins needed

    """

    def __init__(self,config_path : Union[str,dict]) -> None:

        if type(config_path)==str:
            self.config_path=config_path
            with open(self.config_path,'r') as f:
                self.config=json.load(f)
        elif type(config_path)==dict:
            self.config=config_path
        self.dataloader=self.parse_config_and_initialize_dataloader()
        self.storage=self.parse_config_and_initialize_storage()
        self.plugins=self.parse_config_and_initialize_plugins()

        self.epochMetric=metrics.ConfusionMatrix(scope='epoch')
        self.taskMetric=metrics.ConfusionMatrix(scope="task")
        self.writer = SummaryWriter()
        self.early_stopper= metrics.EarlyStopper(patience=500,min_delta_percentage=0.005)
        
        print("Trainer initialized ! ")
    

    def train(self,_run):
        # Create the optimizer
        

        # Get the data loader
        self.before_training(self.plugins)

        #Step the dataloader
        last_ending_epoch=0
        for exp in range(self.storage.nbrs_experiments):
            self.storage.current_network.train()
            self.storage=self.before_training_exp(self.plugins)
            self.storage=self.before_train_dataset_adaptation(self.plugins)
            self.storage=self.after_train_dataset_adaptation(self.plugins)
            # cm=[]
            
            for epoch in tqdm(range(self.storage.epochs)):
                counter = 1  # Counter to compute avg loss
                loss_accu=0.0
                self.before_training_epoch(self.plugins)
                
                for inputs,targets in self.storage.dataloader:
                    targets=targets.to(self.storage.device)
                    inputs=inputs.to(self.storage.device)
                    self.storage.targets=targets # torch tensor of targets NOT in one hot vector form
                    self.storage.inputs=inputs
                    self.before_training_iteration(self.plugins)
                    self.during_training_iteration(self.plugins)
                    self.after_training_iteration(self.plugins)
                    counter += 1
                    loss_accu+=self.storage.loss
                
                loss_value=loss_accu/counter if isinstance(loss_accu,float) else loss_accu.item()/counter
                self.writer.add_scalar("loss/train",loss_value,last_ending_epoch+epoch)
                self.after_training_epoch(self.plugins)

                # # print("Start Evaluation >>>")
                # self.eval('val', self.val_dataloader, exp, _run, compute_metric=False)  # On val data
                # self.writer.add_scalar("loss/val", self.storage.eval_loss, last_ending_epoch + epoch)
                
                # print("End Evaluation <<<")
                # if self.early_stopper.early_stop(self.storage.eval_loss):
                #     print("Stopping the learning due to early stopping based on val loss")
                #     break
                if self.early_stopper.early_stop(loss_value):
                    print("Stopping the learning due to early stopping based on train loss")
                    break
            last_ending_epoch+=epoch

                # self.epochMetric.update({"y":targets,"y_pred":self.storage.logits}) # Compute with the last batch a proxy of epoch Confusion Matrix
                # Log the metrics and flush it
                # Use it and flush it later
                # cm.append(self.epochMetric.result["cm"][0])
            
                #Flush it
                # self.epochMetric.reset()
                

            self.after_training_exp(self.plugins)
            
            #Eval
            # Flush the buffer
            self.storage.dataloader.dataset.buffer_x={}
            #self.eval('train',self.storage.dataloader,exp) # On train data
            self.eval('val', self.val_dataloader, exp, _run, compute_metric=True)
            self.early_stopper.reset()
            
            
        
        self.after_training(self.plugins)
        print("End Training <<< \n\n ")
        _run.info["confusion_matrix"]=self.storage.confusion_matrix
   
        return self.storage

    def eval(self,type, dataloader,exp,_run,compute_metric=False):
         # Eval the network
        self.storage=self.before_eval(self.plugins)
        self.storage=self.before_eval_exp(self.plugins)
        self.storage=self.before_eval_dataset_adaptation(self.plugins)
        self.storage = self.after_eval_dataset_adaptation(self.plugins)
        
        counter = 1 # Counter to compute eval loss
        eval_loss=0.0

        for inputs, targets in dataloader:
            with torch.no_grad():
                targets=targets.to(self.storage.device)
                inputs = inputs.to(self.storage.device)
                self.storage.targets=targets # torch tensor of targets NOT in one hot vector form
                self.storage.inputs = inputs

                # Old logits
                if self.storage.dataloader.dataset.current_experiment>0:
                   

                    old_output=self.storage.old_network(self.storage.inputs)
                    self.storage.old_logits=old_output["logits"]  # torch tensor of logits from last experience model
                    self.storage.old_attentions = old_output["attentions"]
                    
                
                self.before_eval_iteration(self.plugins)
                self.before_eval_forward(self.plugins)
                self.during_eval_forward(self.plugins)
                self.after_eval_forward(self.plugins)
                eval_loss += self.storage.loss #Loss is being accumulated

                self.after_eval_iteration(self.plugins)
                self.taskMetric.update({"y": targets, "y_pred": self.storage.logits})
                counter += 1

        eval_loss = eval_loss / counter  # get the Eval loss
        self.storage.eval_loss = eval_loss if isinstance(eval_loss, float) else eval_loss.item()
        
        if not compute_metric:
            return self.storage

        self.after_eval_exp(self.plugins)

        self.storage.confusion_matrix[type].update({exp:self.taskMetric.result['cm'][0]})
        accuracies=metrics.Accuracy().compute(self.storage.confusion_matrix[type])
        # print("Confusion Matrix",self.storage.confusion_matrix[type][exp])
        accuracy=accuracies[list(accuracies.keys())[-1]]
        accuracy=accuracy[self.storage.seen_classes_mask]
        accuracy=accuracy.tolist()
        cls_name=np.arange(len(self.storage.seen_classes_mask))
        cls_name=cls_name[self.storage.seen_classes_mask]

        cls_name=list(map(lambda x : str(x)+"_cls_"+self.storage.dataloader.dataset.label_dict_inverted[x],cls_name))
        plot=dict(zip(cls_name,accuracy))

        self.writer.add_scalars("eval/acc_i/",plot,exp)

        # Log in sacred
        for key,value in plot.items():
            _run.log_scalar(key, value, exp)

        self.after_eval(self.plugins)
        self.taskMetric.reset()

        # Compute the other metrics 


        return self.storage

    def parse_config_and_initialize_plugins(self):
        """
        Parse the configuration to get the plugins description
        """
        self.plugins=[]
        
        # REPRESENTATION PLUGINS 
        # Plugins placeholder for customs parameters

        for plugin_cfg in self.config["plugins"]:
            name=plugin_cfg["name"]
            try:
                plugin = return_plugin(name)()

                if plugin is not None : self.plugins.append(plugin)
                self.storage.plugins_storage[name]=plugin_cfg
            except:
                print(f"plugin {name} does not exit in database")
            
        return self.plugins

    def parse_config_and_initialize_storage(self):
        
        self.storage=Storage()


        # model 
        model_type=self.config["model"]["model_type"]
        self.storage.current_network=ExpandableNet(**{"netType":model_type,"input_dim":self.dataloader.dataset.output_shape,
                                               "hidden_dims":[1],
                                                "out_dimension":self.config["model"]["hidden_size"],
                                                "device":torch.device("cuda" if torch.cuda.is_available() else "cpu")})

        #Optimisation related
        self.storage.batch_size=self.config["optimisation"]["batch_size"]
        self.storage.device=self.config["optimisation"]["device"]
        self.storage.epochs=self.config["optimisation"]["epochs"]
        self.storage.lr=self.config["optimisation"]["lr"]
        self.storage.batch_size=self.config["optimisation"]["batch_size"]

        optimizer_name=self.config["optimisation"]["optimizer"]["type"]
        weight_decay=self.config["optimisation"]["optimizer"]["weight_decay"]
        self.storage.optimizer=factory.get_optimizer(params=self.storage.current_network.parameters(),
                                                     optimizer=optimizer_name,
                                                     lr=self.storage.lr,
                                                     weight_decay=weight_decay)
        
        self.storage.dataloader=self.dataloader

        
        self.set_current_task_mask()
        self.storage.nbrs_experiments=self.dataloader.dataset.n_splits
        
        return self.storage
    def set_current_task_mask(self):
        all_cls=np.arange(self.dataloader.dataset.max_classes)
        task_existing_cls=set(self.dataloader.dataset.activated_files_labels_subset)
        task_mask=[]
        for cls in all_cls.tolist():
            if cls in task_existing_cls:
                task_mask.append(True)
            else :
                task_mask.append(False)
        
        self.storage.task_mask= task_mask # Get all the existing classes in the current experiments
        return task_mask

    
    def parse_config_and_initialize_dataloader(self):

        dataset_name=self.config["data"]["dataset_name"]
        data_path=self.config["data"]["data_path"]
        n_experiments=self.config["data"]["n_experiments"]
        backbone_name=self.config["data"]["backbone"]
        split_mode=self.config["data"]["scenario"]
        
        train_dataset=BaseDataset(url=data_path,
                                name=dataset_name,
                                backbone_name=backbone_name,
                                n_splits=n_experiments,
                                mode="train",
                                save_embedding=True,
                                split_mode=split_mode)
        
        val_dataset=BaseDataset(url=data_path,
                                name=dataset_name,
                                backbone_name=backbone_name,
                                n_splits=n_experiments,
                                mode="test",
                                save_embedding=True,
                                split_mode=split_mode,
                                split_distribution=train_dataset.split_distributions,
                                label_dict=train_dataset.label_dict)

        initial_bs = self.config["optimisation"]["batch_size"]

        train_N_min_split = min(list(map(len, train_dataset.datasets)))
        val_N_min_split=min(list(map(len,val_dataset.datasets)))
        minimal_optim_loop = 3
        
        train_optimised_bs = min(initial_bs, train_N_min_split // (minimal_optim_loop))
        val_optimised_bs = min(initial_bs,  val_N_min_split // (minimal_optim_loop))
        

        print("Optimised Batch size for training : ", train_optimised_bs)
        print("Optimised Batch size for testing : ", val_optimised_bs)

        
       
        self.dataloader = data.DataLoader(train_dataset, 
                                    batch_size=initial_bs,
                                    shuffle=True,
                                    drop_last=True)

        # N = sum(val_dataset.counts)
        # minimal_optim_loop=2
        # val_optimised_bs = min(initial_bs, N // (n_experiments * minimal_optim_loop))
        # print("Optimised Batch size for testing : ", val_optimised_bs)
        
        self.val_dataloader = data.DataLoader(val_dataset, 
                                    batch_size=initial_bs,
                                    shuffle=True,
                                    drop_last=False)
        

    


        return self.dataloader

    #Training loop
    def before_training(self,plugins : list[Operation]):
        # Update the storage 

        # Initialise the optimiser

        # Initialise the scheduler

        # Initialise the metrics
        return self.storage
      
    def before_training_exp(self,plugins : list[Operation]):
        self.storage.dataloader.dataset.step()
        self.val_dataloader.dataset.step()
        self.set_current_task_mask()
        

        self.storage=self.build_and_run_relevant_pipeline(stage_name="before_training_exp")
        # Update the model outputs
        self.storage.current_network.add_classes(current_task_classes=self.storage.task_mask,
                                                 old_task_classes=self.storage.seen_classes_mask)
        

        return self.storage
    
    def before_train_dataset_adaptation(self,plugins : list[Operation]):
        self.storage=self.build_and_run_relevant_pipeline(stage_name="before_train_dataset_adaptation")
        return self.storage
    
    def after_train_dataset_adaptation(self,plugins : list[Operation]):
        self.storage=self.build_and_run_relevant_pipeline(stage_name="after_train_dataset_adaptation")
        return self.storage
    
    def before_training_epoch(self,plugins : list[Operation]):
        self.storage=self.build_and_run_relevant_pipeline(stage_name="before_training_epoch")
        return self.storage
    
    def before_training_iteration(self,plugins : list[Operation]):
        self.storage=self.build_and_run_relevant_pipeline(stage_name="before_training_iteration")
        self.storage.loss=0.0
        return self.storage
    
    def during_training_iteration(self,plugins : list[Operation]):

        
        plugins_in_launch_order=self.sort_plugin(stage_name="during_training_iteration",plugins=self.plugins,precedence=self.storage.precedence)
        if len(plugins_in_launch_order)>0 :
            # Modified behavior VS Modified Behavior
            # Use a different iteration scheme>
            # Need to be implemented as a plugin 
            # Typicall example are Experience Replay or MIR methods
            self.storage=self.build_and_run_relevant_pipeline(stage_name="during_training_iteration")
            return self.storage

        # Default behavior
        
        self.before_forward(self.plugins)

        output=self.storage.current_network(self.storage.inputs)
        self.storage.attentions=output["attentions"] # Torch tensor of tensors representing the intermediates outputs of each layers. If MLP or CNN, layers outputs 
        self.storage.logits=output["logits"] # Torch tensor of last logits

        if self.storage.dataloader.dataset.current_experiment>0:
            old_output=self.storage.old_network(self.storage.inputs)
            self.storage.old_logits=old_output["logits"]  # torch tensor of logits from last experience model
            self.storage.old_attentions=old_output["attentions"]

        #Update here the storage with outputs
        
        self.after_forward(self.plugins)
        self.storage=self.before_backward(self.plugins)
        self.storage.loss.backward()
        self.after_backward(self.plugins)
        self.before_update(self.plugins)
        self.storage.optimizer.step()
        self.after_update(self.plugins)
        
        return self.storage
    
    def before_forward(self,plugins : list[Operation]):
        self.storage=self.build_and_run_relevant_pipeline(stage_name="before_forward")
        return self.storage
    
    def after_forward(self,plugins : list[Operation]):
        self.storage=self.build_and_run_relevant_pipeline(stage_name="after_forward")
        return self.storage
    
    def before_backward(self,plugins : list[Operation]):
        self.storage=self.build_and_run_relevant_pipeline(stage_name="before_backward")
        # print(f"Loss : {self.storage.loss.item()}")
        return self.storage
    
    def after_backward(self,plugins : list[Operation]):
        self.storage=self.build_and_run_relevant_pipeline(stage_name="after_backward")
        
        return self.storage
    
    def before_update(self,plugins : list[Operation]):
        
        self.storage=self.build_and_run_relevant_pipeline(stage_name="before_update")
        return self.storage
    
    def after_update(self,plugins : list[Operation]):
        
        self.storage=self.build_and_run_relevant_pipeline(stage_name="after_update")
        
        return self.storage
    
    def after_training_iteration(self,plugins : list[Operation]):
        self.storage=self.build_and_run_relevant_pipeline(stage_name="after_training_iteration")
        # Reset the batch counter
        self.storage.dataloader.dataset.current_batch_paths=[]
        return self.storage
    
    def after_training_epoch(self,plugins : list[Operation]):
        self.storage=self.build_and_run_relevant_pipeline(stage_name="after_training_epoch")

        return self.storage
    
    def after_training_exp(self,plugins : list[Operation]):
        
        
        # Update seen classes list
        if self.storage.seen_classes_mask is None:
            self.storage.seen_classes_mask=copy.copy(self.storage.task_mask)
        else:
            for cnt,(old,new )in enumerate(zip(self.storage.seen_classes_mask,self.storage.task_mask)):
                self.storage.seen_classes_mask[cnt] = old | new
                
        self.storage=self.build_and_run_relevant_pipeline(stage_name="after_training_exp")

        
        self.storage.old_network=self.storage.current_network.copy().freeze() # A non trainable ExpandableNet storing the old weights and biases of last experience training
         
        return self.storage
    
    def after_training(self,plugins : list[Operation]):
        self.storage=self.build_and_run_relevant_pipeline(stage_name="after_training")
        return self.storage
    

    # Eval loop

     
    def before_eval(self,plugins : list[Operation]):
        self.storage = self.build_and_run_relevant_pipeline(stage_name="before_eval")
        
        return self.storage
    
 
    def before_eval_exp(self,plugins : list[Operation]):
        self.storage=self.build_and_run_relevant_pipeline(stage_name="before_eval_exp")
        return self.storage
    
 
    def before_eval_dataset_adaptation(self,plugins : list[Operation]):
        self.storage=self.build_and_run_relevant_pipeline(stage_name="before_eval_dataset_adaptation")
        return self.storage
    
 
    def after_eval_dataset_adaptation(self,plugins : list[Operation]):
        self.storage=self.build_and_run_relevant_pipeline(stage_name="after_eval_dataset_adaptation")
        return self.storage
    
 
    def before_eval_iteration(self,plugins : list[Operation]):
        self.storage = self.build_and_run_relevant_pipeline(stage_name="before_eval_iteration")
        self.storage.loss=0.0
        return self.storage
    
 
    def before_eval_forward(self,plugins : list[Operation]):
        self.storage=self.build_and_run_relevant_pipeline(stage_name="before_eval_forward")
        return self.storage
    
    def during_eval_forward(self,plugins : list[Operation]):

        
        plugins_in_launch_order=self.sort_plugin(stage_name="during_eval_forward",plugins=self.plugins,precedence=self.storage.precedence)
        if len(plugins_in_launch_order)>0 :
            # Modified behavior VS Modified Behavior
            # Use a different iteration scheme>
            # Need to be implemented as a plugin 
            # Typicall example are Experience Replay or MIR methods
            self.storage=self.build_and_run_relevant_pipeline(stage_name="during_eval_forward")
            return self.storage

        # Default behavior
        # Update the eval logits depending on the plugins behaviour
        self.storage.current_network.eval()
        with torch.no_grad():
            
            output=self.storage.current_network(self.storage.inputs)
            self.storage.attentions=output["attentions"] # Torch tensor of tensors representing the intermediates outputs of each layers. If MLP or CNN, layers outputs 
            self.storage.logits=output["logits"] # Torch tensor of last logits
        
        return self.storage
 
    def after_eval_forward(self,plugins : list[Operation]):
        self.storage=self.build_and_run_relevant_pipeline(stage_name="after_eval_forward")
        return self.storage
    
 
    def after_eval_iteration(self,plugins : list[Operation]):
        self.storage=self.build_and_run_relevant_pipeline(stage_name="after_eval_iteration")
        return self.storage
    
 
    def after_eval_exp(self,plugins : list[Operation]):
        self.storage = self.build_and_run_relevant_pipeline(stage_name="after_eval_exp")
        self.storage.loss=0.0
        return self.storage
    
    def after_eval(self,plugins : list[Operation]):
        self.storage = self.build_and_run_relevant_pipeline(stage_name="after_eval")
        
        return self.storage
    



    def serialize(self,plugins_in_launch_order: list[Operation]):
        """
        Serialise plugins
        """

        for plugin in plugins_in_launch_order: # A plugin of class Operation
            plugin.set_inputs(self.storage)
            self.storage=plugin.callback() # Do the plugin computation
            # self.storage.plugins_storage[plugin.name].update(result) # Return and update the storage
            
            # if plugin.is_loss : 
            #     try:
            #         assert result["loss"] is not None
            #     except AssertionError :
            #         return self.storage

            #     # Compute the ponderation

            #     plugin_alpha=self.storage.plugins_storage[plugin.name]["loss_coeff"]
            #     self.storage.loss+=plugin_alpha*result["loss"] # Update the whole loss if the plugin is a loss function
        
        return self.storage

    
    def sort_plugin(self,stage_name: str,plugins: list[Operation], precedence: list[int]):
        """
        Sort the plugins so that only the ones applicable at one stage are used
        precedence is a list of integer , the first to be call has a higher value for the same stage 
        eg if we should duplicate and prune a network, we may want to prune first then duplicate, then pruning precedence value must be higher the duplicate plugin
        """
        # Stage definition
        stage=EntryPoints[stage_name]

        #Relevant plugins list
        relevant_plugins=[]
        relevant_precedence=[]

        for pos,plugin in enumerate(plugins):
            if stage in plugin.entry_point:
                relevant_plugins.append(plugin)
                # relevant_precedence.append(precedence[pos])
        
        # relevant_plugins=sorted(relevant_plugins,key=lambda i:relevant_precedence[relevant_plugins.index(i)],reverse=True)

        return relevant_plugins
    
    def build_and_run_relevant_pipeline(self,stage_name: str):
        #Ensure we have the good stage at the start and the end of a stage
        self.storage.stage_name=stage_name

        plugins_in_launch_order=self.sort_plugin(stage_name=stage_name,plugins=self.plugins,precedence=self.storage.precedence)
        self.storage.activated_plugins=plugins_in_launch_order
        if len(plugins_in_launch_order)>0:
            self.storage=self.serialize(plugins_in_launch_order=plugins_in_launch_order)

        self.storage.stage_name=stage_name

        return self.storage

@ex.main
def main(_run):
    # print(_run.config)
    trainer=Trainer(_run.config)
    trainer.train(_run)


    

if __name__=="__main__":
    ex.run_commandline()