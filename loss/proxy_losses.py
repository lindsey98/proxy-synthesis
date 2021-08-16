'''
proxy-synthesis
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

import sklearn


def pairwise_distance(a, squared=False):
    """Computes the pairwise distance matrix with numerical stability."""
    pairwise_distances_squared = torch.add(
        a.pow(2).sum(dim=1, keepdim=True).expand(a.size(0), -1),
        torch.t(a).pow(2).sum(dim=0, keepdim=True).expand(a.size(0), -1)
    ) - 2 * (
        torch.mm(a, torch.t(a))
    )

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(
        pairwise_distances_squared, min=0.0
    )

    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)
    #print(error_mask.sum())
    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(
            pairwise_distances_squared + error_mask.float() * 1e-16
        )

    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(
        pairwise_distances,
        (error_mask == False).float()
    )

    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - torch.eye(
        *pairwise_distances.size(),
        device=pairwise_distances.device
    )
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances


def binarize_and_smooth_labels(T, nb_classes, smoothing_const = 0):
    import sklearn.preprocessing
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T).cuda()

    return T

def proxy_synthesis(input_l2, proxy_l2, target, ps_alpha, ps_mu):
    '''
    input_l2: [batch_size, dims] l2-normalized embedding features
    proxy_l2: [n_classes, dims] l2-normalized proxy parameters
    target: [batch_size] Note that adjacent labels should be different (e.g., [0,1,2,3,4,5,...])
    ps_alpha: alpha for beta distribution
    ps_mu: generation ratio (# of synthetics / batch_size)
    '''

    input_list = [input_l2]
    proxy_list = [proxy_l2]
    target_list = [target]

    ps_rate = np.random.beta(ps_alpha, ps_alpha)

    input_aug = ps_rate * input_l2 + (1.0 - ps_rate) * torch.roll(input_l2, 1, dims=0)
    proxy_aug = ps_rate * proxy_l2[target,:] + (1.0 - ps_rate) * torch.roll(proxy_l2[target,:], 1, dims=0)
    input_list.append(input_aug)
    proxy_list.append(proxy_aug)
    
    n_classes = proxy_l2.shape[0]
    pseudo_target = torch.arange(n_classes, n_classes + input_l2.shape[0]).cuda()
    target_list.append(pseudo_target)

    embed_size = int(input_l2.shape[0] * (1.0 + ps_mu))
    proxy_size = int(n_classes + input_l2.shape[0] * ps_mu)
    input_large = torch.cat(input_list, dim=0)[:embed_size,:]
    proxy_large = torch.cat(proxy_list, dim=0)[:proxy_size,:]
    target = torch.cat(target_list, dim=0)[:embed_size]
    
    input_l2 = F.normalize(input_large, p=2, dim=1)
    proxy_l2 = F.normalize(proxy_large, p=2, dim=1)

    return input_l2, proxy_l2, target


class Norm_SoftMax(nn.Module):
    def __init__(self, input_dim, n_classes, scale=23.0, ps_mu=0.0, ps_alpha=0.0):
        super(Norm_SoftMax, self).__init__()
        self.scale = scale
        self.n_classes = n_classes
        self.ps_mu = ps_mu
        self.ps_alpha = ps_alpha
        self.proxy = Parameter(torch.Tensor(n_classes, input_dim))
        
        init.kaiming_uniform_(self.proxy, a=math.sqrt(5))
        

    def forward(self, input, target):
        input_l2 = F.normalize(input, p=2, dim=1)
        proxy_l2 = F.normalize(self.proxy, p=2, dim=1)
        
        if self.ps_mu > 0.0:
            input_l2, proxy_l2, target = proxy_synthesis(input_l2, proxy_l2, target,
                                                         self.ps_alpha, self.ps_mu)

        sim_mat = input_l2.matmul(proxy_l2.t())
        
        logits = self.scale * sim_mat
        
        loss = F.cross_entropy(logits, target)
        
        return loss


class Proxy_NCA(nn.Module):
    def __init__(self, input_dim, n_classes, scale=10.0, ps_mu=0.0, ps_alpha=0.0):
        super(Proxy_NCA, self).__init__()
        self.scale = scale
        self.n_classes = n_classes
        self.ps_mu = ps_mu
        self.ps_alpha = ps_alpha
        self.proxy = Parameter(torch.Tensor(n_classes, input_dim))
        
        init.kaiming_uniform_(self.proxy, a=math.sqrt(5))
    
    
    def forward(self, input, target):
        input_l2 = F.normalize(input, p=2, dim=1)
        proxy_l2 = F.normalize(self.proxy, p=2, dim=1)

        if self.ps_mu > 0.0:
            input_l2, proxy_l2, target = proxy_synthesis(input_l2, proxy_l2, target,
                                                         self.ps_alpha, self.ps_mu)
 
        dist_mat = torch.cdist(input_l2, proxy_l2) ** 2
        dist_mat *= self.scale
        pos_target = F.one_hot(target, dist_mat.shape[1]).float()
        loss = torch.mean(torch.sum(-pos_target * F.log_softmax(-dist_mat, -1), -1))

        return loss
    
    
class ProxyNCA_prob(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale, **kwargs):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8)
        self.scale = scale
     
    def forward(self, X, T):
        P = self.proxies
        #note: self.scale is equal to sqrt(1/T)
        # in the paper T = 1/9, therefore, scale = sart(1/(1/9)) = sqrt(9) = 3
        #  we need to apply sqrt because the pairwise distance is calculated as norm^2
       
        P = self.scale * F.normalize(P, p = 2, dim = -1)
        X = self.scale * F.normalize(X, p = 2, dim = -1)
        
        D = pairwise_distance(
            torch.cat(
                [X, P]
            ),
            squared = True
        )[:X.size()[0], X.size()[0]:]

        T = binarize_and_smooth_labels(
            T = T, nb_classes = len(P), smoothing_const = 0
        )

        loss = torch.sum(- T * F.log_softmax(-D, -1), -1)
        loss = loss.mean()
        return loss
