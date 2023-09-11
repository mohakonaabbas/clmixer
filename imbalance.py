import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm
class BiC(nn.Module):
    def __init__(self, lr,
                 lr_decay_factor,
                   weight_decay, 
                   batch_size, epochs):
        super(BiC, self).__init__()
        self.beta = torch.nn.Parameter(torch.ones(1))  #.cuda()
        self.gamma = torch.nn.Parameter(torch.zeros(1))  #.cuda()
        self.lr = lr
        # self.scheduling = scheduling
        self.lr_decay_factor = lr_decay_factor
        self.weight_decay = weight_decay
        self.class_specific = False
        self.batch_size = batch_size
        self.epochs = epochs
        self.bic_flag = False

    def reset(self, lr=None, scheduling=None, lr_decay_factor=None, weight_decay=None, n_classes=-1):
        with torch.no_grad():
            if lr is None:
                lr = self.lr
            # if scheduling is None:
            #     scheduling = self.scheduling
            if lr_decay_factor is None:
                lr_decay_factor = self.lr_decay_factor
            if weight_decay is None:
                weight_decay = self.weight_decay
            if self.class_specific:
                assert n_classes != -1
                self.beta = torch.nn.Parameter(torch.ones(n_classes).cuda())
                self.gamma = torch.nn.Parameter(torch.zeros(n_classes).cuda())
            else:
                self.beta = torch.nn.Parameter(torch.ones(1).cuda())
                self.gamma = torch.nn.Parameter(torch.zeros(1).cuda())
            self.optimizer = torch.optim.SGD([self.beta, self.gamma], lr=lr, momentum=0.9, weight_decay=weight_decay)
            # self.scheduler = CosineAnnealingLR(self.optimizer, 10)
            # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, scheduling, gamma=lr_decay_factor)

    def extract_preds_and_targets(self, model, loader):
        preds, targets = [], []
        with torch.no_grad():
            for (x, y) in loader:
                preds.append(model(x.cuda())['logits'])
                targets.append(y.cuda())
        return torch.cat((preds)), torch.cat((targets))

    def update(self, model, loader, loss_criterion=None):
        # if task_size == 0:
            # logger.info("no new task for BiC!")
            # return
        if loss_criterion is None:
            loss_criterion = F.cross_entropy

        # self.bic_flag = True
        # logger.info("Begin BiC ...")
        model.eval()

        preds_, targets_ = self.extract_preds_and_targets(model, loader)

        for epoch in tqdm(range(self.epochs)):
            
            order = np.arange(preds_.shape[0])
            np.random.shuffle(order)

            preds, targets = preds_.clone(), targets_.clone()
            preds, targets = preds[order], targets[order]
            _loss = 0.0
            _correct = 0
            _count = 0
            for start in range(0, preds.shape[0], self.batch_size):
                if start + self.batch_size < preds.shape[0]:
                    outputs = preds[start:start + self.batch_size, :].clone()
                    lbls = targets[start:start + self.batch_size]
                else:
                    outputs = preds[start:, :].clone()
                    lbls = targets[start:]
                    
                outputs = outputs * self.beta + self.gamma
                loss = loss_criterion(outputs, lbls)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                _, pred = outputs.max(1)
                _correct += (pred == lbls).sum()
                _count += lbls.size(0)
                _loss += loss.item() * outputs.shape[0]
            # logger.info("epoch {} loss {:4f} acc {:4f}".format(epoch, _loss / preds.shape[0], _correct / _count))

            # self.scheduler.step()
        # logger.info("beta {:.4f} gamma {:.4f}".format(self.beta.cpu().item(), self.gamma.cpu().item()))

    @torch.no_grad()
    def post_process(self, preds):
        # if self.class_specific is False:
        #     if task_size != 0:
        #         preds[:, -task_size:] = preds[:, -task_size:] * self.beta + self.gamma
        # else:
        preds = preds * self.beta + self.gamma
        return preds


class WA(object):
    def __init__(self):
        self.gamma = None

    @torch.no_grad()
    def update(self, classifier, old_mask,new_mask):
        """
        Task_mask masks out the newly introduced task classes. This can includes old classes.
        """
        EPSILON=1e-32
        # old_mask = task_mask
        if min(old_mask) == True:
            self.gamma = 1.0
            return self.gamma

        # new_mask = list(map(lambda x: not x, task_mask))
        try:
            old_weight_norm = torch.norm(classifier.weight[old_mask], p=2, dim=1)
            new_weight_norm = torch.norm(classifier.weight[new_mask], p=2, dim=1)
        except:
            old_weight_norm = torch.norm(classifier.module.weight[old_mask], p=2, dim=1)
            new_weight_norm = torch.norm(classifier.module.weight[new_mask], p=2, dim=1)
        self.gamma = old_weight_norm.mean() / (EPSILON + new_weight_norm.mean())
       
       
    @torch.no_grad()
    def post_process(self, preds, new_mask):
        # new_mask=[not x for x in task_mask]
        preds[:,new_mask] = preds[:, new_mask] * self.gamma
        return preds
