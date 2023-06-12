import functools


import os
import sys
# from inclearn.strategies.finetuning import Finetuning
current_dir=os.getcwd()
base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.getcwd())
sys.path.append(base_dir)



import math
from xmlrpc.client import TRANSPORT_ERROR
from sklearn.preprocessing import OneHotEncoder
# from sympy import false

import torch
import torch. nn as nn
from torch.nn import functional as F
from dit.divergences import earth_movers_distance
from scipy.optimize import linprog
import numpy as np
import time
import sinkhorn
# https://pythonot.github.io/index.html
import ot 
#https://stackoverflow.com/questions/72284627/is-this-custom-pytorch-loss-function-differentiable


def mer_loss(new_logits, old_logits):
    """Distillation loss that is less important if the new model is unconfident.

    Reference:
        * Kim et al.
          Incremental Learning with Maximum Entropy Regularization: Rethinking
          Forgetting and Intransigence.

    :param new_logits: Logits from the new (student) model.
    :param old_logits: Logits from the old (teacher) model.
    :return: A float scalar loss.
    """
    new_probs = F.softmax(new_logits, dim=-1)
    old_probs = F.softmax(old_logits, dim=-1)

    return torch.mean(((new_probs - old_probs) * torch.log(new_probs)).sum(-1), dim=0)


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

    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        # shape of (b, n, w, h)
        assert a.shape == b.shape, (a.shape, b.shape)

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

        layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss

    return loss / len(list_attentions_a)


def spatial_pyramid_pooling(
    list_attentions_a,
    list_attentions_b,
    levels=[1, 2],
    pool_type="avg",
    weight_by_level=True,
    normalize=True,
    **kwargs
):
    loss = torch.tensor(0.).to(list_attentions_a[0].device)

    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        # shape of (b, n, w, h)
        assert a.shape == b.shape

        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        for j, level in enumerate(levels):
            if level > a.shape[2]:
                raise ValueError(
                    "Level {} is too big for spatial dim ({}, {}).".format(
                        level, a.shape[2], a.shape[3]
                    )
                )
            kernel_size = level // level

            if pool_type == "avg":
                a_pooled = F.avg_pool2d(a, (kernel_size, kernel_size))
                b_pooled = F.avg_pool2d(b, (kernel_size, kernel_size))
            elif pool_type == "max":
                a_pooled = F.max_pool2d(a, (kernel_size, kernel_size))
                b_pooled = F.max_pool2d(b, (kernel_size, kernel_size))
            else:
                raise ValueError("Invalid pool type {}.".format(pool_type))

            a_features = a_pooled.view(a.shape[0], -1)
            b_features = b_pooled.view(b.shape[0], -1)

            if normalize:
                a_features = F.normalize(a_features, dim=-1)
                b_features = F.normalize(b_features, dim=-1)

            level_loss = torch.frobenius_norm(a_features - b_features, dim=-1).mean(0)
            if weight_by_level:  # Give less importance for smaller cells.
                level_loss *= 1 / 2**j

            loss += level_loss

    return loss


def relative_teacher_distances(features_a, features_b, normalize=False, distance="l2", **kwargs):
    """Distillation loss between the teacher and the student comparing distances
    instead of embeddings.

    Reference:
        * Lu Yu et al.
          Learning Metrics from Teachers: Compact Networks for Image Embedding.
          CVPR 2019.

    :param features_a: ConvNet features of a model.
    :param features_b: ConvNet features of a model.
    :return: A float scalar loss.
    """
    if normalize:
        features_a = F.normalize(features_a, dim=-1, p=2)
        features_b = F.normalize(features_b, dim=-1, p=2)

    if distance == "l2":
        p = 2
    elif distance == "l1":
        p = 1
    else:
        raise ValueError("Invalid distance for relative teacher {}.".format(distance))

    pairwise_distances_a = torch.pdist(features_a, p=p)
    pairwise_distances_b = torch.pdist(features_b, p=p)

    return torch.mean(torch.abs(pairwise_distances_a - pairwise_distances_b))


def perceptual_features_reconstruction(list_attentions_a, list_attentions_b, factor=1.):
    loss = 0.

    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        bs, c, w, h = a.shape

        # a of shape (b, c, w, h) to (b, c * w * h)
        a = a.view(bs, -1)
        b = b.view(bs, -1)

        a = F.normalize(a, p=2, dim=-1)
        b = F.normalize(b, p=2, dim=-1)

        layer_loss = (F.pairwise_distance(a, b, p=2)**2) / (c * w * h)
        loss += torch.mean(layer_loss)

    return factor * (loss / len(list_attentions_a))


def perceptual_style_reconstruction(list_attentions_a, list_attentions_b, factor=1.):
    loss = 0.

    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        bs, c, w, h = a.shape

        a = a.view(bs, c, w * h)
        b = b.view(bs, c, w * h)

        gram_a = torch.bmm(a, a.transpose(2, 1)) / (c * w * h)
        gram_b = torch.bmm(b, b.transpose(2, 1)) / (c * w * h)

        layer_loss = torch.frobenius_norm(gram_a - gram_b, dim=(1, 2))**2
        loss += layer_loss.mean()

    return factor * (loss / len(list_attentions_a))


def gradcam_distillation(gradients_a, gradients_b, activations_a, activations_b, factor=1):
    """Distillation loss between gradcam-generated attentions of two models.

    References:
        * Dhar et al.
          Learning without Memorizing
          CVPR 2019

    :param base_logits: [description]
    :param list_attentions_a: [description]
    :param list_attentions_b: [description]
    :param factor: [description], defaults to 1
    :return: [description]
    """
    attentions_a = _compute_gradcam_attention(gradients_a, activations_a)
    attentions_b = _compute_gradcam_attention(gradients_b, activations_b)

    assert len(attentions_a.shape) == len(attentions_b.shape) == 4
    assert attentions_a.shape == attentions_b.shape

    batch_size = attentions_a.shape[0]

    flat_attention_a = F.normalize(attentions_a.view(batch_size, -1), p=2, dim=-1)
    flat_attention_b = F.normalize(attentions_b.view(batch_size, -1), p=2, dim=-1)

    distances = torch.abs(flat_attention_a - flat_attention_b).sum(-1)

    return factor * torch.mean(distances)


def _compute_gradcam_attention(gradients, activations):
    alpha = F.adaptive_avg_pool2d(gradients, (1, 1))
    return F.relu(alpha * activations)


def mmd(x, y, sigmas=[1, 5, 10], normalize=False):
    """Maximum Mean Discrepancy with several Gaussian kernels."""
    # Flatten:
    x = x.view(x.shape[0], -1)
    y = y.view(y.shape[0], -1)

    if len(sigmas) == 0:
        mean_dist = torch.mean(torch.pow(torch.pairwise_distance(x, y, p=2), 2))
        factors = (-1 / (2 * mean_dist)).view(1, 1, 1)
    else:
        factors = _get_mmd_factor(sigmas, x.device)

    if normalize:
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)

    xx = torch.pairwise_distance(x, x, p=2)**2
    yy = torch.pairwise_distance(y, y, p=2)**2
    xy = torch.pairwise_distance(x, y, p=2)**2

    k_xx, k_yy, k_xy = 0, 0, 0

    div = 1 / (x.shape[1]**2)

    k_xx = div * torch.exp(factors * xx).sum(0).squeeze()
    k_yy = div * torch.exp(factors * yy).sum(0).squeeze()
    k_xy = div * torch.exp(factors * xy).sum(0).squeeze()

    mmd_sq = torch.sum(k_xx) - 2 * torch.sum(k_xy) + torch.sum(k_yy)
    return torch.sqrt(mmd_sq)


@functools.lru_cache(maxsize=1, typed=False)
def _get_mmd_factor(sigmas, device):
    sigmas = torch.tensor(sigmas)[:, None, None].to(device).float()
    sigmas = -1 / (2 * sigmas)
    return sigmas


def similarity_per_class(
    features,
    targets,
    goal_features,
    goal_targets,
    epoch,
    epochs,
    memory_flags,
    old_centroids_features=None,
    old_centroids_targets=None,
    factor=1.,
    scheduled=False,
    apply_centroids=True,
    initial_centroids=False
):
    loss = 0.
    counter = 0

    # We only keep new classes, no classes stored in memory
    indexes = ~memory_flags.bool()
    features = features[indexes]
    targets = targets[indexes].to(features.device)

    for target in torch.unique(targets):
        sub_features = features[targets == target]

        sub_goal_features = goal_features[goal_targets == target]
        if apply_centroids:
            sub_goal_features = sub_goal_features.mean(dim=0, keepdims=True)

        # We want the new real features to be similar to their old alter-ego ghosts:
        similarities = torch.mm(
            F.normalize(sub_features, dim=1, p=2),
            F.normalize(sub_goal_features, dim=1, p=2).T
        )
        loss += torch.clamp((1 - similarities).sum(), min=0.)
        counter += len(sub_features)

        if initial_centroids:
            # But we also want that the new real features stay close to what the
            # trained ConvNet though was best as first initialization:
            sub_centroids = old_centroids_features[old_centroids_targets == target]
            similarities = torch.mm(
                F.normalize(sub_features, dim=1, p=2), F.normalize(sub_centroids.T, dim=1, p=2)
            )
            loss += torch.clamp((1 - similarities).sum(), min=0.)
            counter += len(sub_features)

    if counter == 0:
        return 0.
    loss = factor * (loss / counter)

    if scheduled:
        loss = (1 - epoch / epochs) * loss

    if loss < 0.:
        raise ValueError(f"Negative loss value for PLC! (epoch={epoch}, epochs={epochs})")

    return loss


def semantic_drift_compensation(old_features, new_features, targets, sigma=0.2):
    """Returns SDC drift.

    # References:
        * Semantic Drift Compensation for Class-Incremental Learning
          Lu Yu et al.
          CVPR 2020
    """
    assert len(old_features) == len(new_features)

    with torch.no_grad():
        delta = new_features - old_features

        denominator = 1 / (2 * sigma**2)

        drift = torch.zeros(new_features.shape[1]).float().to(new_features.device)
        summed_w = 0.
        for target in torch.unique(targets):
            indexes = target == targets
            old_features_class = old_features[indexes]

            # Computing w, aka a weighting measuring how much an example
            # is representative based on its distance to the class mean.
            numerator = old_features_class - old_features_class.mean(dim=0)
            numerator = torch.pow(torch.norm(numerator, dim=1), 2)
            w = torch.exp(-numerator / denominator)

            tmp = (w[..., None] * delta[indexes])
            drift = drift + tmp.sum(dim=0)
            summed_w = summed_w + w.sum()
        drift = drift / summed_w

    return drift


def oracle_arena_loss(x:torch.Tensor,
                      targets:torch.Tensor,
                      training_network,
                      previous_classes_oracle_network,
                      next_classes_oracle_network,
                      old_seen_classes:list,
                      new_seen_classes:list):
    """
    https://www.google.com/search?channel=fs&client=ubuntu&q=dirichlet+loss+pytorch+
    https://stackoverflow.com/questions/72284627/is-this-custom-pytorch-loss-function-differentiable
    https://analyticsindiamag.com/how-to-model-uncertainty-with-dempster-shafers-theory/
    https://chcorbi.github.io/posts/2020/11/dirichlet-networks/
    https://arxiv.org/pdf/1806.01768.pdf
    https://colab.research.google.com/github/muratsensoy/muratsensoy.github.io/blob/master/uncertainty.ipynb#scrollTo=g9Zb_A8AKOLa

    28/06/2022 : First Attempt at writing a loss which take in account 3 networks and dirichlet categorical prediction
    The idea behind this loss is to use 2 networks biaised toward new and past data to guide a gdumd initialised network.
    We expect the optimisation to go toward the true minimum we would obtain using cumulative approch.
    
    By default , neural network are too certains of their prediction.
    Another way to see it is that they always give a solution using softmax. 
    We want the network to be able to reason about uncertainties.
    This uncertainties mean that we can have a measure of how sure the network is from is answer,
    but in a CIL setup,it encourage to find stable representation and leave room for new knowledge to come to fill the uncertainties.
    To do so , we use 
    

    #1 Transform y to make it one hot vector

    #Distillation part
    
    #Old classes
        #If target is from old_seen_classes
            # It Dirichlet distribution from current network must match previous_classes_oracle_network distribution for that sample
        #If not
            #It previous_classes_oracle_network prediction must be close to uniform distribution

    #New classes
     #If target is from new_seen_classes
            # It Dirichlet distribution from current network must match next_classes_oracle_network distribution for that sample
        #If not
            #It next_classes_oracle_network prediction must be close to uniform distribution
    
    #Job ( classification ) loss
        Dirichlet distribution of the current network prediction must decrease the MSE with y
        Any unexplainable evidence must lead to uniform dustribution

    """

    return 0

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


def differentiable_argmax(y_pred,y,cut=0.99,cut_val=0.001):
    """
    THis function emulate the behavior of an argmax on a set of vector and compute the mean square error
    We compute from the y_pred the minimax value
    on these minimax data, we compute a fitted 2nd order function  argmaxer=Ax² + Bx + C
    such that argmaxer(1)=1.0;argmaxer(0)=0;argmaxer(cut)=cut_val
    After computing 
    we have :
    C=0
    B=(1-cut_val)/(cut² -cut)
    A=1-B
    """

    C=0
    B=(cut**2-cut_val)/(cut**2 -cut)
    A=1-B
    #Compute the minimax scaling so that the value lie between 0 and 1
    s_y_pred=(y_pred-torch.min(y_pred,dim=1,keepdim=True).values)/(torch.max(y_pred,dim=1,keepdim=True).values-torch.min(y_pred,dim=1,keepdim=True).values)
    s_y=(y-torch.min(y,dim=1,keepdim=True).values)/(torch.max(y,dim=1,keepdim=True).values-torch.min(y,dim=1,keepdim=True).values)
    #Compute the scaling
    m_y_pred=A*s_y_pred**2+B*s_y_pred+C
    m_y=A*s_y**2+B*s_y+C
    #Clip them befor zeros
    m_y_pred=torch.nn.functional.relu(m_y_pred)
    m_y=torch.nn.functional.relu(m_y)
    #Compute the mean square error
    loss=torch.mean((m_y_pred-m_y)**2)

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


def cross_entropy_compatible(y_pred,y,**kwargs):

    return F.cross_entropy(y_pred,y)




def earth_mover_distance(y_pred,y,return_y_onehot=False,**kwargs):
    """
    Compute an earth mover distance between two distributions
    https://pypi.org/project/dit/
    # https://pythonot.github.io/index.html
    """
    #Format y
    max_hot=kwargs["max_class_id"]
    if not kwargs["raw"]:
        y=torch.nn.functional.one_hot(y,max_hot)
    else:
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

    bs=len(m)
    (bs,ch)=m.shape

    
    
    distances=1.0-torch.eye(ch,requires_grad=True).to(m.device)
    # distances=torch.arange(end=ch,requires_grad=False).reshape(-1,1).to(m.device)
    distances=distances/1.0
    distances=torch.cdist(distances,distances)
    loss_=[]
    for i in range(bs):

        x_i=round_probs_to_one(m[i].to(torch.float64))
        y_i=round_probs_to_one(y[i].to(torch.float64))
        
        (emd,log)=ot.emd2(x_i,y_i,distances, log=True,return_matrix=True)
        loss_.append(emd)
        # emd=torch.sum(OT*distances)
        # print("Greedy Earth Mover Distance {} and {} is {}".format(a[i].item(),b[i],emd))
        if i==0:
            # loss=emd.clone()
            loss=emd
        else: 
            loss=torch.vstack([loss,emd])
    
    loss=torch.mean(loss)
    if return_y_onehot:
        return loss, m,y
    else:
        return loss
    


def evidential_emd(y_pred,y,**kwargs):
    #Variables
    epoch=kwargs["epoch"]
    kl_annealing_step=kwargs["kl_annealing_step"]
    max_hot=kwargs["max_class_id"]

    

    annealing_coef = torch.tensor(min(1.0,epoch/kl_annealing_step))
    annealing_coef=annealing_coef.to(y_pred.device)

    #Compute emd between target and output
    if False:
        emd_loss,m,y_true=earth_mover_distance(y_pred,y,True,**kwargs)
    else:
        if not kwargs["raw"]:
            y_true=torch.nn.functional.one_hot(y,max_hot)
        else:
            # loss=differentiable_argmax(y_pred,y,cut=0.99,cut_val=0.001)
            # return loss
            
            #Put y in prob form
            #Compute the dirichlet parameters
            alphay=y+1
            #Compute the sum
            Sy = torch.sum(alphay, axis=1, keepdim=True) 
            #Compute probabilities ; m= p ( in the paper)
            y_true = alphay / Sy# belief probabilities
    

    #Compute standard ce loss
    ce_loss=dirichlet_BCE(y_pred,y,**kwargs)
    #Remove the evidence for the true class
    alp = y_pred*(1-y_true)
    # alp=alp 1e-12+torch.sum(alp,axis=1)).reshape(-1,1)
    #Compute the sum
    mask_alp_y_true=torch.zeros(alp.shape).to(alp.device)
    mask_alp_y_true[:,0]=1.0*torch.sum(alp,axis=1)
    
    #Compute abstentions evidences
    
    #All other outputs should be mapped to the abstention class
    #Evidential learning implies 
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    alp_loss=kl_loss(torch.log(alp+1.0),1.0+mask_alp_y_true)




    # alp_loss =  alp_loss*annealing_coef
    if torch.isnan(alp_loss):
        raise ValueError

    # loss=alp_loss+emd_loss+ce_loss
    # return (ce_loss+annealing_coef*emd_loss)/(1.0+annealing_coef) #+ annealing_coef*(alp_loss) 
    return ce_loss #+ annealing_coef*(alp_loss) 





def round_probs_to_one(prob:np.array):
    """
    This utils to round a prob list to one exactly
    """
    rest=1.0-torch.sum(prob)
    if rest!=0.0:
        #share the rest over everybody
        prob+=rest/len(prob)
    prob[-1]=1.0-torch.sum(prob[:-1])
    assert torch.isclose(torch.sum(prob),torch.tensor(1.0,dtype=torch.double))
    return prob




def greedy_earth_mover(y_pred:torch.Tensor,y:torch.Tensor,distances=None,**kwargs):
    """
    A greedy plan for em calculus
    We fill the closest holes first
    We fill the biggest holes in priority
    x: y_pred
    y: y

    """

    #Format y
    max_hot=kwargs["max_class_id"]
    if not kwargs["raw"]:
        y=torch.nn.functional.one_hot(y,max_hot)
    else:
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

    bs=len(m)
    for i in range(bs):
        
        emd=greedy_earth_mover_planning(m[i,:],y[i,:])
        # print("Greedy Earth Mover Distance {} and {} is {}".format(a[i].item(),b[i],emd))
        if i==0:
            loss=emd/bs
        else: loss+=emd/bs
    return loss


def greedy_earth_mover_planning(x:torch.Tensor,y:torch.Tensor,distances=None):
    

    #Round x and y so that it makes exactly 1 on each sum
    x=round_probs_to_one(x.to(torch.float64))
    y=round_probs_to_one(y.to(torch.float64))
    ##Compute the closest holes filling 
    fill_plan=x-torch.maximum(x-x,x-y) #MAx between 0 and x-y
    plan_matrix=torch.diag(fill_plan.squeeze())
    #Compute the plan matrix
    #Get the remaining piles
    piles=torch.maximum(x-x,x-y)
    holes=fill_plan.clone()-y


    while not torch.isclose(sum(piles),sum(holes)):

        #Find biggest hole
        h_index=torch.argmin(holes)
        # Find biggest piles
        p_index=torch.argmax(piles)
        # Pour maximum amount of pile in hole
        poor=piles[p_index]+holes[h_index]
        update=piles[p_index]-torch.maximum(poor-poor,poor)
        #Update plan matrix
        plan_matrix[h_index,p_index]=update
        #Updates piles
        piles[p_index]-=update
        # Update holes 
        holes[h_index]+=update

    
    # print("Cost of that path {}".format(np.sum(plan_matrix*distances)))
    n = len(x)
    # if distances is None:

    #     x_axis=torch.arange(end=n,requires_grad=False).reshape(-1,1).to(x.device)
    #     # x_axis=x_axis.repeat(n,1)

    #     d_x=x_axis+x.reshape(-1,1)
    #     d_y=x_axis+y.reshape(-1,1)
    # sinkhorn_loss_func = sinkhorn.SinkhornDistance(eps=1, max_iter=100, reduction=None)
    # loss=sinkhorn_loss_func(x,y)
    distances=1-torch.eye(n).to(x.device)
    Wd = pot.emd2(x, y, distances) # exact linear program
    reg=1e-3
    Wd_reg = pot.sinkhorn2(x, y, distances, reg,verbose=True) # entropic regularized OT
    # distances=categorical_distances(n)
    
    # distances=torch.cdist(x.reshape(-1,1),y.reshape(-1,1))
    
    loss=torch.sum(plan_matrix*distances)
            


    return loss


    


def categorical_distances(n):
    """
    Construct a categorical distances matrix.

    Parameters
    ----------
    n : int
        The size of the matrix.
    
    Returns
    -------
    ds : np.ndarray
        The matrix of distances.
    """
    return 1 - np.eye(n)

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

    




