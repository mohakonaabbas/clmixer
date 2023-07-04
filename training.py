from plugins import return_plugin
from base_plugin import Operation,EntryPoints
import json
import factory
import numpy as np
# from dataloader_vit import *
from datasets.base import BaseDataset
from torch.utils import data
from network import ExpandableNet
from storage import Storage
import copy
import metrics
from tqdm import tqdm
import torch

from torch.utils.tensorboard import SummaryWriter
class Trainer:
    """
    This class aim at implementings the training of CL algorith in a modular way.
    It loads basics componnents such as optimizer and the plugins needed

    """

    def __init__(self,config_path) -> None:
        self.config_path=config_path
        with open(self.config_path,'r') as f:
            self.config=json.load(f)
        self.dataloader=self.parse_config_and_initialize_dataloader()
        self.storage=self.parse_config_and_initialize_storage()
        self.plugins=self.parse_config_and_initialize_plugins()

        self.epochMetric=metrics.ConfusionMatrix(scope='epoch')
        self.taskMetric=metrics.ConfusionMatrix(scope="task")
        self.writer=SummaryWriter()
        
        print("Trainer initialized ! ")
    

    def train(self):
        # Create the optimizer
        

        # Get the data loader
        self.before_training(self.plugins)

        #Step the dataloader

        for exp in range(self.storage.nbrs_experiments):
            self.storage.current_network.train()
            self.storage=self.before_training_exp(self.plugins)
            self.storage=self.before_train_dataset_adaptation(self.plugins)
            self.storage=self.after_train_dataset_adaptation(self.plugins)
            cm=[]
            for epoch in tqdm(range(self.storage.epochs)):
                self.before_training_epoch(self.plugins)
                
                for inputs,targets in self.storage.dataloader:
                    targets=targets.to(self.storage.device)
                    inputs=inputs.to(self.storage.device)
                    self.storage.targets=targets # torch tensor of targets NOT in one hot vector form
                    self.storage.inputs=inputs
                    self.before_training_iteration(self.plugins)
                    self.during_training_iteration(self.plugins)
                    self.after_training_iteration(self.plugins)
                self.writer.add_scalar("loss/train",self.storage.loss.item(),exp*self.storage.epochs+epoch)
                self.after_training_epoch(self.plugins)

                self.epochMetric.update({"y":targets,"y_pred":self.storage.logits}) # Compute with the last batch a proxy of epoch Confusion Matrix
                # Log the metrics and flush it
                # Use it and flush it later
                cm.append(self.epochMetric.result["cm"][0])
            
                #Flush it
                self.epochMetric.reset()
                

            self.after_training_exp(self.plugins)
            
            #Eval
            # Flush the buffer
            self.storage.dataloader.dataset.buffer_x={}
            #self.eval('train',self.storage.dataloader,exp) # On train data
            print("Start Evaluation >>>")
            self.eval('val',self.val_dataloader,exp) # On val data
            print("End Evaluation <<<")
            
        
        self.after_training(self.plugins)

                
        return self.storage

    def eval(self,type, dataloader,exp):
         # Eval the network
        self.storage=self.before_eval(self.plugins)
        self.storage=self.before_eval_exp(self.plugins)
        self.storage=self.before_eval_dataset_adaptation(self.plugins)
        self.storage=self.after_eval_dataset_adaptation(self.plugins)
       
        for inputs,targets in dataloader:
            targets=targets.to(self.storage.device)
            inputs=inputs.to(self.storage.device)
            self.storage.targets=targets # torch tensor of targets NOT in one hot vector form
            self.storage.inputs=inputs
            self.before_eval_iteration(self.plugins)
            self.before_eval_forward(self.plugins)
            self.during_eval_forward(self.plugins)
            self.after_eval_forward(self.plugins)
            self.after_eval_iteration(self.plugins)
            self.taskMetric.update({"y":targets,"y_pred":self.storage.logits})


        self.after_eval_exp(self.plugins)
        self.storage.confusion_matrix[type].update({exp:self.taskMetric.result['cm'][0]})
        accuracies=metrics.Accuracy().compute(self.storage.confusion_matrix[type])
        accuracy=accuracies[list(accuracies.keys())[-1]]
        accuracy=accuracy.tolist()
        cls_name=np.arange(len(accuracy))
        cls_name=list(map(lambda x : str(x)+"_cls_"+self.storage.dataloader.dataset.label_dict_inverted[x],cls_name))
        plot=dict(zip(cls_name,accuracy))

        self.writer.add_scalars("eval/acc_i/",plot,exp)
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
        
        train_dataset=BaseDataset(url=data_path,
                                name=dataset_name,
                                backbone_name=backbone_name,
                                n_splits=n_experiments,
                                mode="train",
                                save_embedding=True)
        
        val_dataset=BaseDataset(url=data_path,
                                name=dataset_name,
                                backbone_name=backbone_name,
                                n_splits=n_experiments,
                                mode="test",
                                save_embedding=True)
       
        self.dataloader = data.DataLoader(train_dataset, 
                                    batch_size=self.config["optimisation"]["batch_size"],
                                    shuffle=True,
                                    drop_last=False)
        
        self.val_dataloader = data.DataLoader(val_dataset, 
                                    batch_size=self.config["optimisation"]["batch_size"],
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
        self.storage=self.build_and_run_relevant_pipeline(stage_name="after_training_exp")
        
        # Update seen classes list
        if self.storage.seen_classes_mask is None:
            self.storage.seen_classes_mask=copy.copy(self.storage.task_mask)
        else:
            for cnt,(old,new )in enumerate(zip(self.storage.seen_classes_mask,self.storage.task_mask)):
                self.storage.seen_classes_mask[cnt]= old | new 

        
        self.storage.old_network=self.storage.current_network.copy().freeze() # A non trainable ExpandableNet storing the old weights and biases of last experience training
         
        return self.storage
    
    def after_training(self,plugins : list[Operation]):
        self.storage=self.build_and_run_relevant_pipeline(stage_name="after_training")
        return self.storage
    

    # Eval loop

     
    def before_eval(self,plugins : list[Operation]):
        self.storage=self.build_and_run_relevant_pipeline(stage_name="before_eval")
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
        self.storage=self.build_and_run_relevant_pipeline(stage_name="before_eval_iteration")
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
        self.storage=self.build_and_run_relevant_pipeline(stage_name="after_eval_exp")
        return self.storage
    
    def after_eval(self,plugins : list[Operation]):
        self.storage=self.build_and_run_relevant_pipeline(stage_name="after_eval")
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
    

if __name__=="__main__":
    config_path="config copy.json"
    trainer=Trainer(config_path)
    trainer.train()