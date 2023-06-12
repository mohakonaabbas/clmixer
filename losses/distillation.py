# import functools


import os
import sys
# from inclearn.strategies.finetuning import Finetuning
# current_dir=os.getcwd()
# base_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.getcwd())
# sys.path.append(base_dir)



import math
# from xmlrpc.client import TRANSPORT_ERROR
# from sklearn.preprocessing import OneHotEncoder
# from sympy import false

import torch
import torch. nn as nn
from torch.nn import functional as F
# from dit.divergences import earth_movers_distance
# from scipy.optimize import linprog
import numpy as np
import time
# import sinkhorn
# # https://pythonot.github.io/index.html
# import ot 
#https://stackoverflow.com/questions/72284627/is-this-custom-pytorch-loss-function-differentiable

def pod(
    list_attentions_a,
    list_attentions_b,
    collapse_channels="spatial",
    normalize=True,
    memory_flags=None,
    only_old=False,
    **kwargs
):
    """Pooled Output Distillation.

    Reference:
        * Douillard et al.
          Small Task Incremental Learning.
          arXiv 2020.

    :param list_attentions_a: A list of attention maps, each of shape (b, n, w, h).
    :param list_attentions_b: A list of attention maps, each of shape (b, n, w, h).
    :param collapse_channels: How to pool the channels.
    :param memory_flags: Integer flags denoting exemplars.
    :param only_old: Only apply loss to exemplars.
    :return: A float scalar loss.
    """
    assert len(list_attentions_a) == len(list_attentions_b)

    loss = torch.zeros(list_attentions_a[0].shape[0]).to(list_attentions_a[0].device)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        # shape of (b, n, w, h)
        assert a.shape == b.shape, (a.shape, b.shape)

        # Put the tensor in the shape ( b,n,w,h)
        if len(a.shape)==2:
            a_shape=(a.shape[0],a.shape[1],1,1)
            a=a.reshape(a_shape)
            b=b.reshape(a_shape)

        if only_old:
            a = a[memory_flags]
            b = b[memory_flags]
            if len(a) == 0:
                continue

        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        if collapse_channels == "channels":
            a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
            b = b.sum(dim=1).view(b.shape[0], -1)
        elif collapse_channels == "width":
            a = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * h)
            b = b.sum(dim=2).view(b.shape[0], -1)
        elif collapse_channels == "height":
            a = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * w)
            b = b.sum(dim=3).view(b.shape[0], -1)
        elif collapse_channels == "gap":
            a = F.adaptive_avg_pool2d(a, (1, 1))[..., 0, 0]
            b = F.adaptive_avg_pool2d(b, (1, 1))[..., 0, 0]
        elif collapse_channels == "spatial":
            a_h = a.sum(dim=3).view(a.shape[0], -1)
            b_h = b.sum(dim=3).view(b.shape[0], -1)
            a_w = a.sum(dim=2).view(a.shape[0], -1)
            b_w = b.sum(dim=2).view(b.shape[0], -1)
            a = torch.cat([a_h, a_w], dim=-1)
            b = torch.cat([b_h, b_w], dim=-1)
        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)
        
        layer_loss=torch.frobenius_norm(a - b, dim=-1)

   
        loss += layer_loss

    return loss / len(list_attentions_a)


def dirichlet_BCE(y_pred:torch.tensor,y:torch.tensor,**kwargs):
    """
    This loss function compute the MSE error according to 

    https://arxiv.org/pdf/1806.01768.pdf
    https://colab.research.google.com/github/muratsensoy/muratsensoy.github.io/blob/master/uncertainty.ipynb#scrollTo=g9Zb_A8AKOLa
    
    
    def mse_loss(p, alpha, global_step, annealing_step): 
        S = tf.reduce_sum(alpha, axis=1, keep_dims=True) 
        E = alpha - 1
        m = alpha / S
        
        A = tf.reduce_sum((p-m)**2, axis=1, keep_dims=True) 
        B = tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keep_dims=True) 
        
        annealing_coef = tf.minimum(1.0,tf.cast(global_step/annealing_step,tf.float32))
        
        alp = E*(1-p) + 1 
        C =  annealing_coef * KL(alp)
        return (A + B) + C

    
    y: a NON hot vector . This function converts it to the hot vector
        
    """
    #alpha=e+1
    #Convert to one hot vector
    epoch=kwargs["epoch"]
    kl_annealing_step=kwargs["kl_annealing_step"]
    max_hot=kwargs["max_class_id"]
    if not kwargs["raw"]:
        y=torch.nn.functional.one_hot(y,max_hot)
    else:
        
        
            # Compute the argmax
        
        # loss=differentiable_argmax(y_pred,y,cut=0.99,cut_val=0.001)
        # return loss
        
        #Put y in prob form
        #Compute the dirichlet parameters
        alphay=y+1
        #Compute the sum
        Sy = torch.sum(alphay, axis=1, keepdim=True) 
        #Compute probabilities ; m= p ( in the paper)
        y = alphay / Sy# belief probabilities
    
    K=y.shape[1]

    #Compute the dirichlet parameters
    alpha=y_pred+1
    #Compute the sum
    S = torch.sum(alpha, axis=1, keepdim=True) 
    #Compute evidences
    E = alpha - 1 #evidence : the outputs of the neural network
    #Compute probabilities ; m= p ( in the paper)
    m = alpha / S # belief probabilities

   
    #Compute the lef side of equation
    A = torch.digamma(S) - torch.digamma(alpha)
    B = torch.sum(y*A, axis=1, keepdim=True) #Fist side of eq : 3 : Classification part
        #Compute the annealing coeff to give increasing importance to KL divergence
    annealing_coef = torch.tensor(min(1.0,epoch/kl_annealing_step))
    annealing_coef=annealing_coef.to(A.device)


    alp = E*(1-y) + 1 
    C =  annealing_coef * dirichlet_KLDivergence_loss(alp)

    # loss=torch.mean(B+C)
    loss=torch.mean(B)

    return loss



def dirichlet_MSE(y_pred:torch.tensor,y:torch.tensor,**kwargs):
    """
    This loss function compute the MSE error according to 

    https://arxiv.org/pdf/1806.01768.pdf
    https://colab.research.google.com/github/muratsensoy/muratsensoy.github.io/blob/master/uncertainty.ipynb#scrollTo=g9Zb_A8AKOLa
    
    
    def mse_loss(p, alpha, global_step, annealing_step): 
        S = tf.reduce_sum(alpha, axis=1, keep_dims=True) 
        E = alpha - 1
        m = alpha / S
        
        A = tf.reduce_sum((p-m)**2, axis=1, keep_dims=True) 
        B = tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keep_dims=True) 
        
        annealing_coef = tf.minimum(1.0,tf.cast(global_step/annealing_step,tf.float32))
        
        alp = E*(1-p) + 1 
        C =  annealing_coef * KL(alp)
        return (A + B) + C

    
    y: a NON hot vector . This function converts it to the hot vector
        
    """
    #alpha=e+1
    #Convert to one hot vector
    epoch=kwargs["epoch"]
    kl_annealing_step=kwargs["kl_annealing_step"]
    max_hot=kwargs["max_class_id"]
    if not kwargs["raw"]:
        y=torch.nn.functional.one_hot(y,max_hot)
    else:
        
        
            # Compute the argmax
        
        # loss=differentiable_argmax(y_pred,y,cut=0.99,cut_val=0.001)
        # return loss
        
        #Put y in prob form
        #Compute the dirichlet parameters
        alphay=y+1
        #Compute the sum
        Sy = torch.sum(alphay, axis=1, keepdim=True) 
        #Compute probabilities ; m= p ( in the paper)
        y = alphay / Sy# belief probabilities
        

    #Compute the dirichlet parameters
    alpha=y_pred+1
    #Compute the sum
    S = torch.sum(alpha, axis=1, keepdim=True) 
    #Compute evidences
    E = alpha - 1 #evidence : the outputs of the neural network
    #Compute probabilities ; m= p ( in the paper)
    m = alpha / S # belief probabilities


    if kwargs["raw"]:
        #https://neuralnet-pytorch.readthedocs.io/en/latest/manual/metrics.html
        kl_loss = torch.nn.KLDivLoss(reduction="none")
        loss=0.5*(kl_loss(torch.log(m),y) + kl_loss(torch.log(y),m)) #jensen - shannon
        return loss
    
    A = torch.sum((y-m)**2, axis=1, keepdim=True) #Fist side of eq : 3 : Classification part

    B = torch.sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdim=True) #Variance part
    #Compute the annealing coeff to give increasing importance to KL divergence
    annealing_coef = torch.tensor(min(1.0,epoch/kl_annealing_step))
    annealing_coef=annealing_coef.to(A.device)
    #alpha_tilde=evidence*(1-y)+1
    alp = E*(1-y) + 1 
    C =  annealing_coef * dirichlet_KLDivergence_loss(alp)

    if True:
        pure_mse_loss=torch.mean(A)
        variance_mse_loss=torch.mean(B)
        kl_loss=torch.mean(C)

    # print("\nPure MSE loss {p} - Variance Dirichlet {v} - KL Divergence {kl}".format(p=pure_mse_loss.item(),v=variance_mse_loss.item(),kl=kl_loss.item()))



    loss=torch.mean(A+B+C)

    return loss
    

def dirichlet_KLDivergence_loss(alpha):
    """
    This loss compute a KL divergence between a dirichlet and uniform dirichlet
    https://arxiv.org/pdf/1806.01768.pdf
    https://colab.research.google.com/github/muratsensoy/muratsensoy.github.io/blob/master/uncertainty.ipynb#scrollTo=g9Zb_A8AKOLa
    

    --> in tensorFlow
    def KL(alpha):
        beta=tf.constant(np.ones((1,K)),dtype=tf.float32)
        S_alpha = tf.reduce_sum(alpha,axis=1,keep_dims=True)
        S_beta = tf.reduce_sum(beta,axis=1,keep_dims=True)
        lnB = tf.lgamma(S_alpha) - tf.reduce_sum(tf.lgamma(alpha),axis=1,keep_dims=True)
        lnB_uni = tf.reduce_sum(tf.lgamma(beta),axis=1,keep_dims=True) - tf.lgamma(S_beta)
        
        dg0 = tf.digamma(S_alpha)
        dg1 = tf.digamma(alpha)
        
        kl = tf.reduce_sum((alpha - beta)*(dg1-dg0),axis=1,keep_dims=True) + lnB + lnB_uni
        return kl

    """
    #Get the number of classes
    K=alpha.shape[-1]
    #Get the uniform distribution
    beta=torch.ones((1,K),dtype=torch.float32)
    beta=beta.to(alpha.device)
    #Compute the intermediate sums
    S_alpha = torch.sum(alpha,axis=1,keepdim=True)
    S_beta = torch.sum(beta,axis=1,keepdim=True)
    #Compute the lef side of equation
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha),axis=1,keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta),axis=1,keepdim=True) - torch.lgamma(S_beta)
    #Compute right side of equation
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    #Compute KL divergence
    kl = torch.sum((alpha - beta)*(dg1-dg0),axis=1,keepdim=True) + lnB + lnB_uni
    return kl



if __name__=="__main__":

    # n = 5
    # batch_size = 4
    # a = np.array([[[i, 0] for i in range(n)] for b in range(batch_size)])
    # b = np.array([[[i, b + 1] for i in range(n)] for b in range(batch_size)])

    # # Wrap with torch tensors
    # x = torch.tensor(a, dtype=torch.float)
    # y = torch.tensor(b, dtype=torch.float)
    n=10
    a=torch.rand(128,n,requires_grad=True)
    
    a=torch.nn.functional.softmax(a)
    b=torch.rand(128,n,requires_grad=True)
    b=torch.nn.functional.softmax(b)
    # x_axis=torch.linspace(start=0,end=3,steps=n)
    # x_axis=x_axis.repeat(len(a),1).reshape(a.shape)
   
    # a=torch.cat([a,x_axis],axis=2).to("cpu")
    # b=torch.cat([b,x_axis],axis=2).to("cpu")
    tic=time.time()
  
    bs=len(a)
    for i in range(bs):
        emd=greedy_earth_mover_planning(a[i,:],b[i,:])
        print("Greedy Earth Mover Distance {} and {} is {}".format(a[i].item(),b[i],emd))
        if i==0:
            loss=emd/bs
        else: loss+=emd/bs
    
    print("Elapsed time Loop {}".format(time.time()-tic))

      
   
    tic=time.time()
    sinkhorn_loss_func = sinkhorn.SinkhornDistance(eps=1, max_iter=100, reduction=None)
    sinkhorn_loss_func=sinkhorn_loss_func.to("cpu")
    dist, P, C = sinkhorn_loss_func(a, b)
    print("Sinkhorn distances: ", dist)

    # result = Pool(cpu_count()).starmap(earth_movers_distance_pmf, zip(a.tolist(),b.tolist()))
    # print("Elapsed time Pool {}".format(time.time()-tic))
    # tic=time.time()
    # print(np.array(emds))

    y_pred=torch.rand(10,5)
    y=torch.tensor([0,2,3,4,1,0,0,0,0,4])
    # y=torch.nn.functional.one_hot(y)
    # y_pred[1,:]=1
    # a[1,0]=0.56
    kl=dirichlet_KLDivergence_loss(y_pred)
    loss=dirichlet_MSE(y_pred,y,**{"epoch":0,"kl_annealing_step":10})
    print(loss)

    




