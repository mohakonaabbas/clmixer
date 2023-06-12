import json
from enum import Enum
from typing import Callable , Union ,Dict
from  storage import Storage
class EntryPoints(Enum):
    """
    #Training loop
    before_training=1
        before_training_exp=2
            before_train_dataset_adaptation=3
            after_train_dataset_adaptation=4
            before_training_epoch=4
                before_training_iteration=5
                during_training_iteration=15
                    before_forward=6
                    after_forward=7
                    before_backward=8
                    after_backward=9
                    before_update=9
                    after_update=10
                after_training_iteration=11
            after_training_epoch=12
        after_training_exp=13
    after_training=14

    #Eval loop
    before_eval=100
        before_eval_exp=101
            before_eval_dataset_adaptation=102
            after_eval_dataset_adaptation=103
            before_eval_iteration=104
                before_eval_forward=105
                after_eval_forward=106
            after_eval_iteration=107
        after_eval_exp=108
    after_eval=109

    """
    #Training loop
    before_training=1
    before_training_exp=2
    before_train_dataset_adaptation=3
    after_train_dataset_adaptation=4
    before_training_epoch=4
    before_training_iteration=5
    before_forward=6
    after_forward=7
    before_backward=8
    after_backward=9
    before_update=9
    after_update=10
    after_training_iteration=11
    after_training_epoch=12
    after_training_exp=13
    after_training=14
    during_training_iteration=15

    #Eval loop
    before_eval=100
    before_eval_exp=101
    before_eval_dataset_adaptation=102
    after_eval_dataset_adaptation=103
    before_eval_iteration=104
    before_eval_forward=105
    after_eval_forward=106
    after_eval_iteration=107
    after_eval_exp=108
    after_eval=109
    during_eval_forward=110

class Operation:
    """
    This class represent a Base class to represent any method or oeration to be conducted in the CL setup
    An operaion is defined by: 
    - A name
    - An entry point ( defined according to the one defined by the Avalanche methodology)
    - A set of inputs 
    - A callback function to perform the operation
    - An output 
    - A paper reference 
    """
    def __init__(self,
                 name : str,
                 entry_point : str,
                 inputs : Union[dict, Storage,None],
                 callback : Callable,
                 paper_ref: str,
                 is_loss : bool = False):
        
        self.set_name(name)
        self.set_entry_point(entry_point)
        self.set_inputs(inputs)
        self.set_callback(callback)
        self.paper_reference(paper_ref)
        self.set_loss(is_loss)
    

    def set_loss(self,is_loss):
        self.is_loss=is_loss
        
    
    
    def set_name(self, name : str):
        self.name=name

    
    def set_entry_point(self, entry_point : Union[str, list[str]]):
        """
        Set entry points lists
        """
        self.entry_point=[]
        if type(entry_point)==str:
            entry_point=[entry_point]
        
        for entry in entry_point: self.entry_point.append(EntryPoints[entry])
    
    def set_inputs(self, inputs_dict : Union[Storage,dict]):
        """
        A list of input to be used by the operations to work with
        """
        self.inputs=inputs_dict

    def set_callback(self,callback : Union[Dict[str,Callable],Callable]):
        """
        A callback that implement the method internals and return a result as a dict
        """
        self.callback=callback
        # print(self.callback)

    
    def set_outputs(self,outputs):
        """
        Save the last output in the data
        """
        self.outputs=outputs

    def paper_reference(self,url : str):
        """
        Give the paper references if any
        """
        self.reference=url



if __name__=="__main__":
    operation=Operation(name="CrossEntropyLoss",entry_point="before_training",inputs={},callback=(lambda x:x),paper_ref="")
    operation



    