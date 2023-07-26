from .base_plugin import Operation
from torch.nn import functional as F
from ..imbalance import BiC
from datasets.base import simpleDataset
from torch.utils import data

class BICOperation(Operation):
    def __init__(self,
                 entry_point =["after_training_exp","after_eval_forward"],
                inputs={},
            callback=(lambda x:x), 
            paper_ref=""):
        super().__init__(entry_point, inputs, callback, paper_ref)

        self.set_callback(self.bic_callback)
        self.set_config_template({"name": self.__class__.__name__,
                                "hyperparameters": {},
                                "function": "bias_mitigation"
                                })
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



if __name__=="__main__":
    operation=BICOperation(name="CrossEntropyLoss",entry_point="before_training",inputs={},callback=(lambda x:x),paper_ref="")
    operation



    