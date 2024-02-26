class Storage:
    """
    This class loads the whole configuration and keeps tracks of al usefull quantities needed for Trainer
    """

    def __init__(self):
        # Training related data
        self.logits=None # Torch tensor of last logits
        self.old_logits=None # torch tensor of logits from last experience model
        self.targets=None # torch tensor of targets NOT in one hot vector form
        self.attentions=None # Torch tensor of tensors representing the intermediates outputs of each layers. If MLP or CNN, layers outputs 
        self.old_attentions=None # Torch tensor of tensors representing the intermediates outputs of each layers. If MLP or CNN, layers outputs 
        
        self.old_network=None # A non trainable ExpandableNet storing the old weights and biases of last experience training
        self.current_network=None # The current ExpandableNet being trained
        self.generator=None # AN eventual old samples generator

        self.epochs=None # The total epochs per training experience
        self.current_epoch=0 # The current epoch
        self.batch_size=None # The trainer batch size
        self.scheduling_steps=None # If lr scheduling by step, represent the stepws
        self.device=None # The device the model is being trained on , likely to be 'cuda'
        self.lr=None

        # Task related data
        self.task_mask=None # This mask represent the already encountered labels during all past experiences. This mask take in account all classes an oracle may know existing even not encountered
        #THis is different from the task_size concept in CIL
        self.seen_classes_mask=None
        self.nbrs_experiments=None
        self.dataloader=None
        self.current_exp=None

        # Plugins placeholder for customs parameters
        self.plugins_storage={} # A list of dict of plugins parameters. Each plugins such as knowledge distillation temperature, beta for Weight align saves here its own parameters
        self.precedence=[] # A list of integers that represent what plugins should be run before others. 
        self.activated_plugins=None
        self.temp_var={"trajectory":[]}
        # Memory
        self.training_memory=None # A dict with 2 torch array for samples and labels representing the saved memory 
        self.balanced_training_memory=None # A dict with 2 torhc array representing a subset of a balanced version of the saved memory
        self.generated_memory=None, # A dict of 2 torch array representing  an eventual generated data  buffer

        # Pruning
        self.pruning_mask=None

        # Metrics
        self.mica=None
        self.acc=None
        self.confusion_matrix={'train':{},'val':{}}

        # LOsses
        self.loss = 0.0
        self.val_loss=10000000.0

        # OPtimizer
        self.optimizer=None

        # Pointer which says where we are in the execution
        # Used now to modify the behavior of a plugin depening on the stage of advancement
        self.stage_name=""
    
    def update(self):
        pass





if __name__=="__main__":
    config_path="config copy.json"
    trainer=Storage()
