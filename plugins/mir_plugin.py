from .base_plugin import Operation
from .random_memory_plugin import RandomMemoryUpdaterOperation
from storage import Storage
import numpy as np
import torch
from datasets.base import simpleDataset
from torch.utils import data
import copy


class MIRMemoryUpdaterOperation(Operation):
    def __init__(self,
                 entry_point= ["before_backward","after_training_exp"], 
                inputs= None, 
                callback= (lambda x:x), 
                paper_ref="https://arxiv.org/abs/1908.04742", 
                is_loss=False):
        super().__init__(entry_point, inputs, callback, paper_ref, is_loss)
    
        self.set_callback(self.mir_callback)
        self.set_config_template({
      "name": self.__class__.__name__,
      "hyperparameters": {},
      "function": "knowledge_retention"
    })

        
       
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

