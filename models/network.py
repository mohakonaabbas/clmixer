import copy

import torch
from torch import nn
import torch.nn.functional as F

import factory

import math

from torch.nn.parameter import Parameter

from torch.nn import Module

# from inclearn.convnet.classifier import CosineClassifier
class CosineClassifier(Module):
    def __init__(self, in_features, n_classes, sigma=False):
        super(CosineClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = n_classes
        self.weight = Parameter(torch.Tensor(n_classes, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            # self.sigma=None
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.sigma is not None:
                self.sigma.data.fill_(1)  #for initializaiton of sigma

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out
    

class CosineLoRAClassifier(Module):
    def __init__(self, in_features, n_classes, sigma=False):
        super(CosineLoRAClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = n_classes
        self.A = Parameter(torch.Tensor(n_classes,1))
        self.B=Parameter(torch.Tensor(1,in_features))
        
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            # self.sigma=None
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_normal_(self.A, nonlinearity="linear")
            nn.init.kaiming_normal_(self.B, nonlinearity="linear")
            # stdvA = 1. / math.sqrt(self.A.size(0))
            # stdvB = 1. / math.sqrt(self.B.size(1))

            # self.A.data.uniform_(-stdvA, stdvA)
            # self.B.data.uniform_(-stdvB, stdvB)
            if self.sigma is not None:
                self.sigma.data.fill_(1)  #for initializaiton of sigma

    def forward(self, input):
        self.weight = torch.matmul(self.A,self.B)
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out


class LinearLoRAClassifier(Module):
    def __init__(self, in_features, n_classes, sigma=False):
        super(LinearLoRAClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = n_classes
        self.A = Parameter(torch.Tensor(n_classes,1))
        self.B=Parameter(torch.Tensor(1,in_features))
        self.linear_bias = nn.Parameter(torch.Tensor(n_classes))
        
       
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():

            nn.init.kaiming_normal_(self.A, nonlinearity="relu")
            nn.init.kaiming_normal_(self.B, nonlinearity="relu")
            # stdvA = 1. / math.sqrt(self.A.size(0))
            # stdvB = 1. / math.sqrt(self.B.size(1))

            # self.A.data.(-stdvA, stdvA)
            # self.B.data.uniform_(-stdvB, stdvB)


    def forward(self, x):
        self.linear_weight = torch.matmul(self.A,self.B)
        out = F.linear(x, self.linear_weight,self.linear_bias)
        out = F.relu(out)

        return out
    


class LinearClassifier(Module):
    def __init__(self, in_features, n_classes, sigma=False):
        super(LinearClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = n_classes
        self.linear_weight = Parameter(torch.Tensor(n_classes,in_features))
        # self.B=Parameter(torch.Tensor(1,in_features))
        self.linear_bias = nn.Parameter(torch.Tensor(n_classes))
        
       
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            stdvA = 1. / math.sqrt(self.linear_weight.size(1))
            # stdvB = 1. / math.sqrt(self.B.size(1))

            self.linear_weight.data.uniform_(-stdvA, stdvA)
            # self.B.data.uniform_(-stdvB, stdvB)


    def forward(self, x):
        # self.linear_weight = torch.matmul(self.A,self.B)
        out = F.linear(x, self.linear_weight,self.linear_bias)
        out = F.relu(out)

        return out

class ExpandableNet(nn.Module):
    def __init__(
        self,
        netType,
        input_dim,
        out_dimension,
        hidden_dims,
        use_bias=True,
        init="kaiming",
        device=None,
        weight_normalization=True,
        lora=False
    ):
        super(ExpandableNet, self).__init__()
        
        self.init = init
        self.netType = netType
        self.input_dim=input_dim
        self.out_dimension=out_dimension
        self.hidden_dims=hidden_dims


        self.remove_last_relu = True
        self.use_bias = use_bias
        self.reuse_oldfc=True

        self.weight_normalization=weight_normalization

        print("Enable dynamical representation expansion!")
        self.nets = nn.ModuleList()
        
        self.nets.append(self.maybeToMultipleGpu(
            factory.get_net(self.netType,**{"input_dim":self.input_dim,
                 "out_dimension": self.out_dimension})))
                 
        
        try:
            self.out_dim = self.nets[0].out_dim
        except:
            self.out_dim = self.nets[0].module.out_dim
 
        
        self.classifier = None
        self.aux_classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device
        self.lora = lora

        
        self.to(self.device)

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

       
        outputs = [convnet(x) for convnet in self.nets]

        attentions = [output["attention"] for output in outputs]
        attentions = torch.cat(attentions, 1)

        
        logits = self.classifier(attentions)

        # aux_logits = self.aux_classifier(features[:, -self.out_dim:]) if features.shape[1] > self.out_dim else None
        return {'attentions': attentions, 'logits': logits}

    @property
    def features_dim(self):
        return self.out_dim * len(self.nets)


    def freeze(self,state=True):
        for param in self.parameters():
            param.requires_grad = not state
        self.eval()
        return self

    
    def freeze_backbone(self,state=True,nets_trainables=[]):
        """
        Freeze the backbone
        Revert the backbone state during defreeze
        """
        count=0
        for net in self.nets:
            if state == True:
                nets_trainables.append(next(net.parameters()).requires_grad) # Save the previous states of backbones
                for param in net.parameters():
                    param.requires_grad=False
            else:
                # Unfreeze the network depending on previous state
                for param in net.parameters():
                    param.requires_grad=nets_trainables[count]
            count+=1
        
            
        return self,nets_trainables


    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, current_task_classes, old_task_classes):
        """
        THis function takes in account a task mask which show all and yet to encounter data we already know about
        It put True where the class has already been encountered and False elsewhere
        """
        
        n_classes=len(current_task_classes)
        # if (self.classifier is None) or ((old_task_classes is not None) and (n_classes>len(old_task_classes)))  :
        self.ntask += 1
        self._add_classes_multi_fc(current_task_classes, old_task_classes)
        self.n_classes = n_classes

    def _add_classes_multi_fc(self, current_task_classes, old_task_classes):
        
        cond2=self.classifier is not None

        if not cond2 :
            fc = self._gen_classifier(self.out_dim * len(self.nets), len(current_task_classes))
            self.classifier = self.maybeToMultipleGpu(fc)
            return
        
        cond1=len(current_task_classes)==len(old_task_classes) # Single head setup
        if not cond1:
            fc = self._gen_classifier(self.out_dim * len(self.nets), len(current_task_classes))
            if cond2:

                if self.lora : 

                    if isinstance(self.classifier, torch.nn.DataParallel):
                        A = self.classifier.module.A.data
                        B = self.classifier.module.B.data

                    else:
                        A = self.classifier.A.data
                        B = self.classifier.B.data
                    
                        weight = copy.deepcopy(self.classifier.weight.data)
                    fc.A.data = A
                    fc.B.data = B  
                else:
                    if isinstance(self.classifier, torch.nn.DataParallel):
                        weight = copy.deepcopy(self.classifier.module.weight.data)
                    else:
                        weight = copy.deepcopy(self.classifier.weight.data)

                if self.reuse_oldfc:
                    fc.weight.data[old_task_classes, :weight.shape[1]] = weight[old_task_classes,:]
            del self.classifier
            self.classifier = self.maybeToMultipleGpu(fc)




        
        # if self.classifier is not None:
        #     if isinstance(self.classifier, torch.nn.DataParallel):
        #         weight = copy.deepcopy(self.classifier.module.weight.data)
        #     else:
        #         weight = copy.deepcopy(self.classifier.weight.data)
        # else:
        #     fc = self._gen_classifier(self.out_dim * len(self.nets), len(current_task_classes))

        # if len(current_task_classes)!=len(old_task_classes):
        #     fc = self._gen_classifier(self.out_dim * len(self.nets), len(current_task_classes))


        #     if self.classifier is not None and self.reuse_oldfc:
        #         old_weight_shape=weight.shape
        #         if fc.weight.data.shape[1]==old_weight_shape[1]:
        #             fc.weight.data=weight
        #         else:
        #             fc.weight.data[old_task_classes, :old_weight_shape[1]] = weight[old_task_classes,:]

        #     del self.classifier
        #     self.classifier = self.maybeToMultipleGpu(fc)

    def _add_classes_multi_backbone(self):
        if self.ntask >= 1:
            new_clf=factory.get_net(self.netType,
                                    **{"input_dim":self.input_dim,
                                       "out_dimension":self.out_dimension}).to(self.device)

            new_clf.load_state_dict(self.nets[-1].state_dict())
            self.nets.append(self.maybeToMultipleGpu(new_clf))


    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            if self.lora:
                # classifier = LinearLoRAClassifier(in_features, n_classes).to(self.device)
                classifier = CosineLoRAClassifier(in_features, n_classes).to(self.device)
            else:
                classifier = CosineClassifier(in_features, n_classes).to(self.device)
                # classifier = LinearClassifier(in_features, n_classes).to(self.device)

            
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)

        return classifier

    def maybeToMultipleGpu(self, module):
        # return module
        if torch.cuda.device_count()>1:
            module = torch.nn.DataParallel(module, device_ids=range(2)) # counts the gpu & performs data parallel if  > 1 gpu
        return module

