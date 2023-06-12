from torch import nn


class MLP(nn.Module):

    def __init__(self, input_dim=1280, hidden_dims=[256], out_dimension=10,use_bn=True, input_dropout=0.1, hidden_dropout=0.1):
        super().__init__()
        self.n_layers=len(hidden_dims)

        layers = []
        for index, dim in enumerate(hidden_dims+[out_dimension]):
            layers.append(nn.Linear(input_dim, dim, bias=True))
            nn.init.normal_(layers[-1].weight, std=0.01)
            nn.init.constant_(layers[-1].bias, 0.)

            if index < len(hidden_dims) - 1:
                if use_bn:
                    layers.append(nn.BatchNorm1d(dim))
                layers.append(nn.LeakyReLU(negative_slope=0.2))
            if input_dropout and index == 0:
                layers.append(nn.Dropout(p=input_dropout))
            elif hidden_dropout and index < len(hidden_dims) - 1:
                layers.append(nn.Dropout(p=hidden_dropout))

            input_dim = dim

        layers.append(nn.Linear(input_dim, hidden_dims[-1]))
        nn.init.normal_(layers[-1].weight, std=0.01)
        nn.init.constant_(layers[-1].bias, 0.)
        
        self.layers=layers


    def forward(self, x):
        attentions=[]
        for i in range(self.n_layers):
            x=self.layers[i](x) # Linear layer
            x=self.layers[i+1](x) # Activation layer
            x=self.layers[i+2](x) # Dropout layer
            attentions.append(x)

        x=self.layers[-1](x)
        features=x
        raw_features=x

        return {"raw_features": raw_features, "features": features, "attention": attentions}
