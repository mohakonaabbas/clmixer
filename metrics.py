import numpy as np
from copy import deepcopy
import torch

class OperationMetric:

    def __init__(self,name : str,
                 scope : str):
        """
        Create a class that compute a metrics and log it
        scope means how often do we compute the metric
        
        """
        self.metric_name=name
        self.scope=scope
        self.internal_count=0
        self.result={}

    

    def update(self,inputs : dict):
        """
        Using the final prediction ( after all postprocessing needed ), compute the metric from the predicted 
        """
        pass

    def reset(self):
        """
        Reset the final prediction
        """
        self.result['cm']={}
        self.internal_count=0



class ConfusionMatrix(OperationMetric):
    """
    Finetune the last layer of a neural network
    """
    def __init__(self, name="confusion_matrix" ,
                 scope="epochs" ):
        super().__init__(name, scope)

        self.result['cm']={}

    def update(self,inputs : dict):
        """
        Using the final prediction ( after all postprocessing needed ), compute the metric from the predicted 
        """

        # Get the maximum size of the confusion matrix
        m=inputs["y_pred"].shape[-1]


        y_pred=inputs["y_pred"].argmax(dim=1)

        targets=inputs["y"]
        
        
        if self.result['cm'] =={}:
            self.result['cm'][self.internal_count]=np.zeros((m,m))
        else:
            prev_m=self.result['cm'][self.internal_count].shape[0]
            if prev_m<m:
                cm=np.zeros((m,m))
                cm[:prev_m,:prev_m]=self.result['cm'][self.internal_count]
                self.result['cm'][self.internal_count]=cm
        
        for i in range(len(y_pred)): # prediction loop
            # for j in range(len(targets)): # Target loop
            #     #Extract the correspondance
            #     # mask1= y_pred==i
            #     # mask2=targets==j
            self.result['cm'][self.internal_count][y_pred[i],targets[i]]+=1
        # print(self.result['cm'][self.internal_count])
                
        # print(self.result['cm'][self.internal_count])
        # self.internal_count+=1
        




class Accuracy(OperationMetric):
    """
    Finetune the last layer of a neural network
    """
    EPSILON=10**-12
    def __init__(self, name='accuracy' ,scope='any' ):
        super().__init__(name, scope)

    
    def compute(self,cm_inputs : dict):
        """
        Using the confusion matrix , compute the accuracy
        compute the metric from the predicted 
        """

        result={}

        for key, value in cm_inputs.items():
            result[key]= np.diagonal(value)/(np.sum(value,axis=1)+self.EPSILON)
                                             

        return result



class EarlyStopper:
    def __init__(self, patience=5, min_delta_percentage=0.001):
        self.patience = patience
        self.min_delta = min_delta_percentage
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if np.abs((validation_loss - self.min_validation_loss) / self.min_validation_loss) < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.min_validation_loss = EPSILON=10**-12+ min(self.min_validation_loss, validation_loss)
            self.counter=0
    
        return False
    
    def reset(self):
        self.counter = 0
        self.min_validation_loss = np.inf


# class MICA(OperationMetric):
#     """
#     Finetune the last layer of a neural network
#     """
#     def __init__(self, name ,scope ):
#         super().__init__(name, scope)

#         pass