
from .base_plugin import Operation
from torch.nn import functional as F
from losses.distillation import dirichlet_uncertain,dirichlet_KLDivergence_loss


class DirichletKLLossOperation(Operation):
    def __init__(self, 
                entry_point =["before_backward","after_eval_forward"],
                inputs={},
                callback=(lambda x:x), 
                paper_ref="https://arxiv.org/pdf/1806.01768.pdf",
                is_loss=True):
        super().__init__(entry_point, inputs, callback, paper_ref,is_loss)

        self.set_config_template({
      "name": self.__class__.__name__,
      "hyperparameters": {
        "kl_annealing_step": 150
      },
      "function": "bias_mitigation"
    })
        self.set_callback(self.dir_kl_callback)
    
    def dir_kl_callback(self,reduction="mean"):
        """
        dirichlet_uncertainty loss function
        """
        
        # New logits
        logits=self.inputs.logits
        targets=self.inputs.targets
        kl_annealing_step=self.inputs.plugins_storage[self.name]["hyperparameters"]["kl_annealing_step"]
        epoch=self.inputs.current_epoch
        
        
        

        loss=dirichlet_KLDivergence_loss(y_pred= logits,
                           y= targets,
                           epoch= epoch,
                           kl_annealing_step= kl_annealing_step
                           )
        



        # loss_coeff= sum(self.inputs.seen_classes_mask)/sum(self.inputs.task_mask)
        loss_coeff=1.0
        
        if reduction=="none":
            return loss
        

        self.inputs.loss+=loss_coeff*loss.mean()
        return self.inputs

