from .base_plugin import Operation
from torch.nn import functional as F
#============================== LOSSES ===============================#
# DONE
class CrossEntropyOperation(Operation):
    def __init__(self,
                 entry_point =["before_backward","after_eval_forward"],
                  inputs=None,
                callback=(lambda x:x), 
                paper_ref="",
                is_loss=True):
        super().__init__(entry_point, inputs, callback, paper_ref,is_loss)
    
        self.set_callback(self.ce_callback)
        self.set_config_template({
      "name": self.__class__.__name__,
      "hyperparameters": {},
      "function": "knowledge_incorporation"
    })


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
#