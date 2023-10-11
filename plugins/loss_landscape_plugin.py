from .base_plugin import Operation
from torch.nn import functional as F


class KnowledgeDistillationOperation(Operation):
    def __init__(self,
                entry_point =["before_backward","after_eval_forward"],
                inputs={},
            callback=(lambda x:x), 
            paper_ref="",
                is_loss=True):
        super().__init__(entry_point, inputs, callback, paper_ref,is_loss)

        self.set_callback(self.kd_callback)
        self.set_config_template({
      "name": self.__class__.__name__,
      "hyperparameters": {
        "temperature": 2.0
      },
      "function": "knowledge_retention"
    })


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
