from torch import nn
import torch

class MLP_SAM(nn.Module):

    def __init__(self, input_dim=4096,
                 hidden_dims=[1],
                 out_dimension=256,
                 input_dropout=0.1):
        super().__init__()
        self.n_layers=out_dimension
        self.out_dim=out_dimension

       
        self.mini_mlps=nn.ModuleList()
        for _ in range(out_dimension):
            layers = []
            dim=hidden_dims[0]
           
            layers.append(nn.Linear(input_dim, dim, bias=True))
            # nn.init.normal_(layers[-1].weight, std=0.01)
            # nn.init.constant_(layers[-1].bias, 0.)

            
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            layers.append(nn.Dropout(p=input_dropout))
            self.mini_mlps.append(nn.Sequential(*layers))


        
        # layers = []

        # layers.append(nn.Linear(out_dimension, out_dimension))
        # layers.append(nn.BatchNorm1d(out_dimension))
        # # layers.append(nn.Softmax())

        # self.head=nn.Sequential(*layers)
        # nn.init.normal_(layers[-1].weight, std=0.01)
        # nn.init.constant_(layers[-1].bias, 0.)
        
        # self.layers=layers


    def forward(self, x):
        attentions=[]

        for i in range(self.n_layers):
            x_i=self.mini_mlps[i](x[:,i,:]) # Linear layer
            attentions.append(x_i)

        attentions=torch.cat(attentions,dim=1)
        # raw_features=self.head(attentions) # Before softmax
        # features=nn.functional.softmax(raw_features)


        return {"attention": attentions}


if __name__=="__main__":
    x=torch.rand(17,256,4096,requires_grad=True)
    mlp=MLP_SAM()
    result=mlp(x)
