from .base_plugin import Operation
from torch.nn import functional as F
import torch
#============================== LOSSES ===============================#
# DONE
class MSEOperation(Operation):
    def __init__(self,
                 entry_point ="before_backward",
                  inputs=None,
                callback=(lambda x:x), 
                paper_ref="",
                is_loss=True):
        super().__init__(entry_point, inputs, callback, paper_ref,is_loss)
    
        self.set_callback(self.mse_callback)
        self.set_config_template({
      "name": self.__class__.__name__,
      "hyperparameters": {},
      "function": "knowledge_incorporation"
    })


    def mse_callback(self,reduction="mean"):
        """
        Cross entropy function
        """

        logits=self.inputs.logits
        targets=self.inputs.targets
        n_classes=len(self.inputs.task_mask)
        
        loss = F.mse_loss(logits, F.one_hot(targets,num_classes=n_classes).to(torch.float32),reduction=reduction)
        loss_coeff=1.0
        # if self.inputs.seen_classes_mask is None : loss_coeff=1
        # else:
        #     loss_coeff= (sum(self.inputs.task_mask)-sum(self.inputs.seen_classes_mask))/sum(self.inputs.task_mask)

        if reduction =="none":
            return loss
        self.inputs.loss+=loss_coeff*loss
        return self.inputs
#