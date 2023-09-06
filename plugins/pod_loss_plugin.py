from .base_plugin import Operation
from losses.distillation import pod

class PodLossOperation(Operation):
    def __init__(self,
                 entry_point =["before_backward","after_eval_forward"],
                inputs={},
            callback=(lambda x:x), 
            paper_ref="https://arxiv.org/pdf/2004.13513.pdf",
                is_loss=True):
        super().__init__(entry_point, inputs, callback, paper_ref,is_loss)

        self.set_callback(self.pod_callback)
        self.set_config_template({
      "name": self.__class__.__name__,
      "hyperparameters": {},
      "function": "knowledge_retention"
    })


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
