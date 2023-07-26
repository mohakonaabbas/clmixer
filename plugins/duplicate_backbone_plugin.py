from .base_plugin import Operation


class DuplicateNetworkBackboneOperation(Operation):
    """
    Duplicate the neural network
    """
    def __init__(self,
                 entry_point ="before_training_exp",
                  inputs={},
                callback=(lambda x:x), 
                paper_ref=""):
        super().__init__(entry_point, inputs, callback, paper_ref)

        self.set_callback(self.duplicate_callback)
        self.set_config_template({
      "name": self.__class__.__name__,
      "hyperparameters": {},
      "function": "representation_learning"
    })

    def duplicate_callback(self):
        """
        duplicate function
        """

        
        # Get the current model
        network=self.inputs.current_network
        # Only upgrade the model if we have more than one class
        if network.ntask == 0:
            return self.inputs
        # Freeze the backbone layers
        network,_=network.freeze_backbone(state=True,nets_trainables=[])

        # Duplicate the network
        network._add_classes_multi_backbone()
        self.inputs.current_network=network


        return self.inputs
