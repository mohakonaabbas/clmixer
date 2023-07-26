from .base_plugin import Operation
import numpy as np

class RandomMemoryUpdaterOperation(Operation):
    def __init__(self,
                 entry_point= "after_training_exp", 
                inputs= None, 
                callback= (lambda x:x), 
                paper_ref="", is_loss=False):
        super().__init__(entry_point, inputs, callback, paper_ref, is_loss)
    
        self.set_callback(self.random_callback)
        self.set_config_template({
      "name": self.__class__.__name__,
      "hyperparameters": {
        "cls_budget": 10
      },
      "function": "knowledge_retention"
    })
       
    def random_callback(self):
        """
        This callback create a random memory
        # Args : budget ( Nbrs of imgs per cls)
        It uses the task_mask to get all the currently seen classes

        """
        #Get the list
        dataset_lbls=self.inputs.dataloader.dataset.activated_files_labels_subset
        dataset=self.inputs.dataloader.dataset.activated_files_subset
        # Get lbls name
        present_lbs_name=np.argwhere(np.array(self.inputs.task_mask)==True).squeeze().tolist()
        if type(present_lbs_name) != list : present_lbs_name=[present_lbs_name]

        n_budget=self.inputs.plugins_storage[self.name]["hyperparameters"]["cls_budget"]
        memory=[]
        memory_lbl=[]

        # For each label
        for lbl in present_lbs_name:
        #Get the mask representing only the considered label

            mask=np.array(dataset_lbls)==lbl
            #Get the labels
            subset=np.array(dataset_lbls)[mask]
            # Shuffle them
            n=len(subset)
            random_order=np.random.choice(np.arange(n),n)

            # Cut to the imposed budget

            n_cut=min(n,n_budget)
            memory+=(np.array(dataset)[mask][random_order][:n_cut]).tolist()
            memory_lbl+=(np.array(dataset_lbls)[mask][random_order][:n_cut]).tolist()

            #Update the storage

        self.inputs.dataloader.dataset.activated_files_subset_memory = memory
        self.inputs.dataloader.dataset.activated_files_labels_subset_memory= memory_lbl



        return self.inputs
