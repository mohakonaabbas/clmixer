from .base_plugin import Operation
from .random_memory_plugin import RandomMemoryUpdaterOperation
from torch.nn import functional as F
import torch
from datasets.base import simpleDataset
from torch.utils import data
from tqdm import tqdm

class FinetuneOperation(Operation):
    """
    Finetune the last layer of a neural network
    """
    def __init__(self,
                 entry_point ="after_training_exp",
                  inputs={},
                callback=(lambda x:x), 
                paper_ref=""):
        super().__init__(entry_point, inputs, callback, paper_ref)

        self.set_callback(self.finetune_callback)
        self.set_config_template({
      "name": self.__class__.__name__,
      "hyperparameters": {
        "finetune_epochs": 10,
        "finetune_bs":32,
        "finetune_lr":1e-3,
        "cls_budget":15
      },
      "function": "bias_mitigation"
    })

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
