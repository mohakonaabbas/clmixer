from .base_plugin import Operation
from torch.nn import functional as F
import torch
#============================== LOSSES ===============================#
# DONE
class WeightedCrossEntropyOperation(Operation):
    def __init__(self,
                 entry_point =["before_backward","after_eval_forward","after_training_exp"],
                  inputs=None,
                callback=(lambda x:x), 
                paper_ref="",
                is_loss=True):
        super().__init__(entry_point, inputs, callback, paper_ref,is_loss)
    
        self.set_callback(self.wce_callback)
        self.set_config_template({
      "name": self.__class__.__name__,
      "hyperparameters": {},
      "function": "knowledge_incorporation"
    })


    def wce_callback(self,reduction="mean"):
        """
        Cross entropy function
        """
        if self.inputs.stage_name == "after_training_exp":
            self.inputs.current_network.nets[0].reset()
            return self.inputs

        logits=self.inputs.logits
        targets=self.inputs.targets
        weights = self.inputs.dataloader.dataset.splits[:,self.inputs.current_exp]
        weights = 2.0-weights/weights.max()
        weights = torch.Tensor(weights).to(logits.device)
        
        loss = F.cross_entropy(logits.softmax(dim=1), targets, weight = weights,reduction=reduction)
        
        
        # loss = F.cross_entropy(logits.softmax(dim=1), targets,reduction=reduction)
        loss_coeff=1.0
        # if self.inputs.seen_classes_mask is None : loss_coeff=1
        # else:
        #     loss_coeff= (sum(self.inputs.task_mask)-sum(self.inputs.seen_classes_mask))/sum(self.inputs.task_mask)

        if reduction =="none":
            return loss
        self.inputs.loss+=loss_coeff*loss
        return self.inputs
#