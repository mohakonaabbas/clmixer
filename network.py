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
    def __init__(self, in_features, n_classes, sigma=True):
        super(CosineClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = n_classes
        self.weight = Parameter(torch.Tensor(n_classes, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  #for initializaiton of sigma

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
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
        weight_normalization=True
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

        print("Enable dynamical reprensetation expansion!")
        self.nets = nn.ModuleList()
        self.nets.append(
            factory.get_net(self.netType,**{"input_dim":self.input_dim,
                 "hidden_dims":self.hidden_dims,
                 "out_dimension":self.out_dimension}))
        self.out_dim = self.nets[0].out_dim
        
        self.classifier = None
        self.aux_classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        
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

    
    def freeze_backbone(self,state=True):
        """
        Freeze the backbone
        """

        for net in self.nets:
            for param in net.parameters():
                param.requires_grad=not state
        
        return self


    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, current_task_classes, old_task_classes):
        """
        THis function takes in account a task mask which show all and yet to encounter data we already know about
        It put True where the class has already been encountered and False elsewhere
        """
        
        n_classes=len(current_task_classes)
        self.ntask += 1
        self._add_classes_multi_fc(current_task_classes, old_task_classes)


        self.n_classes = n_classes

    def _add_classes_multi_fc(self, current_task_classes, old_task_classes):

        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)


        fc = self._gen_classifier(self.out_dim * len(self.nets), len(current_task_classes))


        if self.classifier is not None and self.reuse_oldfc:
            old_weight_shape=weight.shape
            fc.weight.data[old_task_classes, :old_weight_shape[1]] = weight[old_task_classes,:]
        del self.classifier
        self.classifier = fc

    def _add_classes_multi_backbone(self):
        if self.ntask > 1:
            new_clf=factory.get_net(self.netType,
                                    **{"input_dim":self.input_dim,
                                       "hidden_dims":self.hidden_dims,
                                       "out_dimension":self.out_dimension}).to(self.device)

            new_clf.load_state_dict(self.nets[-1].state_dict())
            self.nets.append(new_clf)


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
            classifier = CosineClassifier(in_features, n_classes).to(self.device)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)

        return classifier
