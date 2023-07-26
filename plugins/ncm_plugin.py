from .base_plugin import Operation
import numpy as np
import torch


class NCMMemoryUpdaterOperation(Operation):
    def __init__(self,
                 entry_point= "after_training_exp", 
                inputs= None, 
                callback= (lambda x:x), 
                paper_ref="https://arxiv.org/abs/1611.07725", 
                is_loss=False):
        super().__init__(entry_point, inputs, callback, paper_ref, is_loss)
    
        self.set_callback(self.ncm_callback)
        self.set_config_template({
      "name": self.__class__.__name__,
      "hyperparameters": {},
      "function": "knowledge_retention"
    })
       
    def ncm_callback(self):
        """
        This callback create a ncm memory according to icarl
        # Args : budget ( Nbrs of imgs per cls)
        It uses the task_mask to get all the currently seen classes

        For all the data in a class , compute the prototypes of the classes
        Do an ordered list of exemplar according to 

        """

        

        # Get lbls name
        present_lbs_name=np.argwhere(np.array(self.inputs.task_mask)==True).squeeze().tolist()
        if type(present_lbs_name) != list : present_lbs_name=[present_lbs_name]

        # List of data
        x_subsets={}
        x_path_subsets={}
        for lbl in present_lbs_name:
            x_subsets[lbl]=[]
            x_path_subsets[lbl]=[]

        #Get the dataloader


        with torch.no_grad():
            for inputs,targets in self.inputs.dataloader:
                # Get the inputs and outputs
                outputs=self.inputs.current_network(inputs)

                # Map to the right placeholder the outputs
                for lbl in present_lbs_name:
                    lbl_mask=targets==lbl
                    x_subsets[lbl].append(outputs["attentions"][lbl_mask])
                    x_path_subsets[lbl].append(np.array(self.inputs.dataloader.dataset.current_batch_paths)[lbl_mask])
                self.inputs.dataloader.dataset.current_batch_paths=[]
        # For each placeholder , put it in a vector size
        X={}
        X_path={}
        for key, val in x_subsets.items():
            X[key]=torch.concatenate(val)

        for key, val in x_path_subsets.items():
            X_path[key]=np.concatenate(val)



        # NCM Part, compute the ncm

        n_budget=self.inputs.plugins_storage[self.name]["hyperparameters"]["cls_budget"]
        memory=[]
        memory_lbl=[]
        

        for key, data in X.items():
            # For each key

            mu=data.mean(dim=0) # Get the mean
            chosen_samples=[False]*data.shape[0]
            order=np.arange(len(chosen_samples))
            n_cut=min(data.shape[0],n_budget)

            if data.shape[0]<=n_budget:
               memory+=(X_path[key]).tolist()
               memory_lbl+=[key]*len(memory)
            else:
                for k in range(1,n_cut):
                    unchosen_samples=list(map(lambda x: not x,chosen_samples))
                    # Read the paper to understand the following lines
                    data_k=data[unchosen_samples]/k
                    phi_pk=torch.sum(data[chosen_samples],dim=0)/k

                    obj=mu-data_k-phi_pk


                    loss=torch.norm(input=obj, dim=1)
                    index=torch.argmin(loss)
                    order_=order[unchosen_samples]
                    chosen_samples[order_[index]]=True
                
                memory+=(X_path[key][chosen_samples]).tolist()
                memory_lbl+=[key]*len(memory)


        self.inputs.dataloader.dataset.activated_files_subset_memory = memory
        self.inputs.dataloader.dataset.activated_files_labels_subset_memory= memory_lbl



        return self.inputs
