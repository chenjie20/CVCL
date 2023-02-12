import time
import torch
import torch.nn as nn
import numpy as np
import math
from metrics import *
import torch.nn.functional as F
from torch.nn.functional import normalize
from dataprocessing import *


class DeepMVCLoss(nn.Module):
    def __init__(self, num_samples, num_clusters):
        super(DeepMVCLoss, self).__init__()
        self.num_samples = num_samples
        self.num_clusters = num_clusters

        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        mask = mask.bool()

        return mask

    def forward_prob(self, q_i, q_j):
        q_i = self.target_distribution(q_i)
        q_j = self.target_distribution(q_j)

        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = (p_i * torch.log(p_i)).sum()

        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = (p_j * torch.log(p_j)).sum()

        entropy = ne_i + ne_j

        return entropy

    def forward_label(self, q_i, q_j, temperature_l, normalized=False):

        q_i = self.target_distribution(q_i)
        q_j = self.target_distribution(q_j)

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.num_clusters
        q = torch.cat((q_i, q_j), dim=0)

        if normalized:
            sim = (self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / temperature_l).to(q.device)
        else:
            sim = (torch.matmul(q, q.T) / temperature_l).to(q.device)

        sim_i_j = torch.diag(sim, self.num_clusters)
        sim_j_i = torch.diag(sim, -self.num_clusters)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


    def target_distribution(self, q):
        weight = (q ** 2.0) / torch.sum(q, 0)
        return (weight.t() / torch.sum(weight, 1)).t()
