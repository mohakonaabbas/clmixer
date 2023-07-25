from torch import nn
import torch

class MLP(nn.Module):

    def __init__(self, 
                 name : str = "mlp",
                 input_dim : int =4096,
                 out_dimension : int =256,
                 input_dropout : int =0.1):
        super().__init__()

        self.out_dim=out_dimension
        if "out_dim" is None : 
            raise Exception("output dimension should be  set")

       

        layers = []
        layers.append(nn.Linear(input_dim, out_dimension, bias=True))
        layers.append(nn.BatchNorm1d(out_dimension))
        layers.append(nn.LeakyReLU(negative_slope=0.2))
        layers.append(nn.Dropout(p=input_dropout))
        self.mini_mlps=nn.Sequential(*layers)




    def forward(self, x):

        x_i=self.mini_mlps(x) # Linear layer
        return {"attention": x_i}


if __name__=="__main__":
    x=torch.rand(17,384,requires_grad=True)
    mlp=MLP(input_dim=384,out_dimension=16)
    result=mlp(x)
