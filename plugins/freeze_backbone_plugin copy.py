from .base_plugin import Operation


class FreezeNetworkBackboneOperation(Operation):
    """
    Duplicate the neural network
    """
    def __init__(self,
                 entry_point =["before_training_exp"],
                  inputs={},
                callback=(lambda x:x), 
                paper_ref=""):
        super().__init__(entry_point, inputs, callback, paper_ref)

        self.set_callback(self.freeze_callback)
        self.set_config_template({
      "name": self.__class__.__name__,
      "hyperparameters": {},
      "function": "representation_learning"
    })

    def freeze_callback(self):
        """
        Freeze function
        """

        
        # Get the current model
        network=self.inputs.current_network
        # Only upgrade the model if we have more than one class
        if self.inputs.current_exp == 0:
            return self.inputs
        # Freeze the backbone layers
        network,_=network.freeze_backbone(state=True,nets_trainables=[])

        self.inputs.current_network=network

        return self.inputs
