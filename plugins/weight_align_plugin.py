from .base_plugin import Operation
from torch.nn import functional as F
from imbalance import BiC,WA


class WeightAlignOperation(Operation):
    def __init__(self,
                 entry_point =["after_training_exp","after_eval_forward"],
                inputs={},
            callback=(lambda x:x), 
            paper_ref=""):
        super().__init__(entry_point, inputs, callback, paper_ref)


        self.set_callback(self.wa_callback)
        self.set_config_template({
      "name": self.__class__.__name__,
      "hyperparameters": {},
      "function": "bias_mitigation"
    })
        self.wa=WA()

    def wa_callback(self):


        try:
            assert self.inputs.old_logits is not None
        except AssertionError :
            return self.inputs
        
        # wa=WA()

        if self.inputs.stage_name=="after_eval_forward":
            #Apply WA transform on data
            self.inputs.logits=self.wa.post_process(self.inputs.logits,self.inputs.seen_classes_mask)
            
        elif self.inputs.stage_name=="after_training_exp":
            network=self.inputs.current_network
            task_mask=self.inputs.seen_classes_mask
            result=self.wa.update(network.classifier,task_mask)

            self.inputs.plugins_storage[self.name].update({"gamma":self.wa.gamma}
    )

        return self.inputs
