from torch import nn
import torch
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, 
                 name : str = "mlp",
                 input_dim : int =4096,
                 out_dimension : int =256,
                 input_dropout : int =0.1):
        super().__init__()
        out_dimension=int(out_dimension)
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


class MLPLora(nn.Module):

    def __init__(self, 
                 name : str = "mlp_lora",
                 input_dim : int =4096,
                 out_dimension : int =256,
                 input_dropout : int =0.1):
        super().__init__()
        out_dimension=int(out_dimension)
        self.out_dim=out_dimension
        if "out_dim" is None : 
            raise Exception("output dimension should be  set")

       

        layers = []
        self.A = nn.Parameter(torch.Tensor(out_dimension,1))
        self.B= nn.Parameter(torch.Tensor(1,input_dim))
        self.linear_bias = nn.Parameter(torch.Tensor(out_dimension))

        nn.init.kaiming_normal_(self.A, nonlinearity="relu")
        nn.init.kaiming_normal_(self.B, nonlinearity="relu")
        # nn.init.kaiming_normal_(self.linear_bias, nonlinearity="relu")
        
        # layers.append(nn.Linear(input_dim, out_dimension, bias=True))
        layers.append(nn.BatchNorm1d(out_dimension))
        layers.append(nn.LeakyReLU(negative_slope=0.2))
        layers.append(nn.Dropout(p=input_dropout))
        self.mini_mlps=nn.Sequential(*layers)




    def forward(self, x):

        self.linear_weight = torch.matmul(self.A,self.B)
        out = F.linear(x, self.linear_weight,self.linear_bias)
        x_i=self.mini_mlps(out) # Linear layer
        return {"attention": x_i}
    


class MLPLoraSubspace(nn.Module):

    def __init__(self, 
                 name : str = "mlp_subspace_lora",
                 input_dim : int = 4096,
                 out_dimension : int = 256,
                 input_dropout : int = 0.1):
        super().__init__()
        out_dimension=int(out_dimension)
        self.out_dim=out_dimension
        if "out_dim" is None : 
            raise Exception("output dimension should be  set")

       

        layers = []

        projection_space : int = 20
        decoupled_space : bool = False

        # Parameters
        self.d = projection_space
        self.decoupled_space = decoupled_space
        self.alphas_A = nn.Parameter((torch.Tensor(self.d,1)))

        if decoupled_space : 
            self.alphas_B = nn.Parameter((torch.Tensor(1,self.d)))

        self.controls_A = torch.Tensor(self.d,out_dimension)
        self.controls_B = torch.Tensor(input_dim,self.d)

        self.linear_bias = nn.Parameter(torch.Tensor(out_dimension))
        
        # layers.append(nn.Linear(input_dim, out_dimension, bias=True))
        layers.append(nn.BatchNorm1d(out_dimension))
        layers.append(nn.LeakyReLU(negative_slope=0.2))
        layers.append(nn.Dropout(p=input_dropout))
        self.mini_mlps=nn.Sequential(*layers)



        # nn.init.kaiming_normal_(self.controls_A, nonlinearity="relu")
        # nn.init.kaiming_normal_(self.controls_B, nonlinearity="relu")
        # nn.init.kaiming_normal_(self.alphas_A, nonlinearity="relu")


        nn.init.uniform_(self.controls_A)#, nonlinearity="relu")
        nn.init.uniform_(self.controls_B)#, nonlinearity="relu")
        nn.init.uniform_(self.alphas_A)#, nonlinearity="relu")


        if decoupled_space :
            nn.init.kaiming_normal_(self.alphas_B, nonlinearity="relu")

        
        for n,p in self.named_parameters():
            if p.requires_grad:
                print(n,p.numel()) 
            





    def forward(self, x):

        A = (self.alphas_A*self.controls_A.to(self.alphas_A.device)).sum(dim = 0)
        A=A.view(-1,1)
        if self.decoupled_space : 
            B= (self.alphas_B*self.controls_B.to(self.alphas_A.device)).sum(dim = 1)
        else:
            B= (self.alphas_A.T*self.controls_B.to(self.alphas_A.device)).sum(dim = 1)
        
        A=A.view(-1,1)
        B=B.view(1,-1)

        self.linear_weight = torch.matmul(A,B)
        out = F.linear(x, self.linear_weight,self.linear_bias)
        x_i=self.mini_mlps(out) # Linear layer
        return {"attention": x_i}



class MLPSubspace(nn.Module):

    def __init__(self, 
                 name : str = "mlp_subspace_lora",
                 input_dim : int = 4096,
                 out_dimension : int = 256,
                 input_dropout : int = 0.1):
        super().__init__()
        out_dimension=int(out_dimension)
        self.out_dim=out_dimension
        if "out_dim" is None : 
            raise Exception("output dimension should be  set")

       

        layers = []

        projection_space : int = 80
        # decoupled_space : bool = False

        # Parameters
        self.d = projection_space
        # self.decoupled_space = decoupled_space
        self.alphas = nn.Parameter((torch.Tensor(self.d,1,1)))

        # if decoupled_space : 
        #     self.alphas_B = nn.Parameter((torch.Tensor(1,self.d)))

        self.linear_weight = torch.Tensor(self.d, out_dimension,input_dim)
        # self.controls_B = torch.Tensor(input_dim,self.d)

        self.linear_bias = nn.Parameter(torch.Tensor(out_dimension))
        
        # layers.append(nn.Linear(input_dim, out_dimension, bias=True))
        layers.append(nn.BatchNorm1d(out_dimension))
        layers.append(nn.LeakyReLU(negative_slope=0.2))
        layers.append(nn.Dropout(p=input_dropout))
        self.mini_mlps=nn.Sequential(*layers)



        nn.init.uniform_(self.linear_weight)
        # nn.init.kaiming_normal_(self.controls_B, nonlinearity="relu")
        nn.init.kaiming_normal_(self.alphas, nonlinearity="relu")


        # nn.init.uniform_(self.linear_weight)#, nonlinearity="relu")
        # nn.init.uniform_(self.controls_B)#, nonlinearity="relu")
        # nn.init.uniform_(self.alphas_A)#, nonlinearity="relu")


        # if decoupled_space :
        #     nn.init.kaiming_normal_(self.alphas_B, nonlinearity="relu")

        
        for n,p in self.named_parameters():
            if p.requires_grad:
                print(n,p.numel()) 

    def reset(self):
        nn.init.kaiming_normal_(self.alphas, nonlinearity="relu")

            





    def forward(self, x):

        linear_weight = (self.alphas*self.linear_weight.to(self.alphas.device)).sum(dim = 0)
        # A=A.view(-1,1)
        # if self.decoupled_space : 
        #     B= (self.alphas_B*self.controls_B.to(self.alphas_A.device)).sum(dim = 1)
        # else:
        #     B= (self.alphas_A.T*self.controls_B.to(self.alphas_A.device)).sum(dim = 1)
        
        # A=A.view(-1,1)
        # B=B.view(1,-1)

        # self.linear_weight = torch.matmul(A,B)
        out = F.linear(x, linear_weight,self.linear_bias)
        x_i=self.mini_mlps(out) # Linear layer
        return {"attention": x_i}




if __name__=="__main__":
    x=torch.rand(17,384,requires_grad=True)
    mlp=MLP(input_dim=384,out_dimension=16)
    result=mlp(x)
