from torch import nn
import torch

class Identity(nn.Module):

    def __init__(self, 
                 name : str = "identity",
                 input_dim : int =0.1,
                 out_dimension : int =-1,
                 input_dropout : int =0.1
                 ):
        super().__init__()
        self.out_dim=int(input_dim)
        if "out_dim" is None : 
            raise Exception("output dimension should be  set")

    def forward(self, x):
        return {"attention": x}


if __name__=="__main__":
    x=torch.rand(17,384,requires_grad=True)
    mlp=Identity(input_dim=384)
    result=mlp(x)
