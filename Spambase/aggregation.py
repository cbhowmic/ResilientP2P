import numpy as np
from copy import deepcopy
import torch
# from cooperation import *
import scipy.spatial.distance

def getWeight(ego_k, A, Para_ex, Para_last, batch_x, batch_y, Accumulated_Loss, rule, attackers):
    # ego_k: agent id whose weights are to be calculated
    # A is the list of neighbors of ego_k
    Weight = []
    gamma = 0.05
    n = len(A)
    f = len(attackers)

    # Proposed aggregation: adaptive  aggregation using
    if rule == "loss":
        Weight = np.zeros((n,))
        reversed_Loss = np.zeros((n,))
        loss = A[ego_k].getLoss(batch_x, batch_y, A[ego_k].net)  # calculate the loss of agent k itself
        Accumulated_Loss[ego_k, ego_k] = (1 - gamma) * Accumulated_Loss[ego_k, ego_k] + gamma * loss  # soft update
        for l in range(0, n):
            if not l == ego_k:
                loss = A[ego_k].getLoss(batch_x, batch_y, A[l].net)  # find the loss using l's network but k's data
                Accumulated_Loss[ego_k, l] = (1 - gamma) * Accumulated_Loss[ego_k, l] + gamma * loss
            if Accumulated_Loss[ego_k, l] <= Accumulated_Loss[ego_k, ego_k]:
                reversed_Loss[l] = 1. / Accumulated_Loss[ego_k, l]
        sum_reversedLoss = sum(reversed_Loss)
        for l in range(0, n):
            if Accumulated_Loss[ego_k, l] <= Accumulated_Loss[ego_k, ego_k]:
                weight = reversed_Loss[l] / sum_reversedLoss
                Weight[l] = weight

    # average aggregation
    elif rule == 'average':
        for l in range(0, n):
            if not l == ego_k:
                weight = 1 / n
            else:
                weight = 1 - (n - 1) / n
            Weight.append(weight)

    # No-cooperation
    elif rule == "no-coop":
        for l in range(0, n):
            if l == ego_k:
                weight = 1
            else:
                weight = 0
            Weight.append(weight)

    # Krum/multi-Krum aggregation
    elif rule in ["krum", "mKrum"]:
        Weight = np.zeros((n,))
        m = n - f - 2
        x = torch.from_numpy(np.array(Para_ex))
        cdist = torch.cdist(x, x, p=2)
        nbhDist, nbh = torch.topk(cdist, m + 1, largest=False)
        if rule == 'krum':
            i_star = np.argmin(nbhDist.sum(1))
            Weight[i_star] = 1
        elif rule == "mKrum":
            i_loss, i_star = torch.topk(nbhDist.sum(1), m + 1, largest=False)
            for l in i_star:
                Weight[l] = 1/m
        # return

    # Medoid aggregation
    elif rule == "medoid":
        Weight = np.zeros((n,))
        dist = scipy.spatial.distance_matrix(Para_ex, Para_ex, p=2)
        i_star = np.argmin(dist.sum(1))
        Weight[i_star] = 1
    else:
        return Weight, Accumulated_Loss




    return Weight, Accumulated_Loss


def aggregate_parameters(ego_k, A, Parameters_exchange, Para_ex, Para_last, batch_x, batch_y, Accumulated_Loss, rule, attackers):
    N = len(A)
    # N_a = len(attackers)
    Parameters = deepcopy(Parameters_exchange)
    a = A[ego_k]

    if rule == 'average':
        Weight, Accumulated_Loss = getWeight(ego_k, A, Para_ex, Para_last, batch_x, batch_y, Accumulated_Loss, rule, attackers)

        for name, param in a.net.named_parameters():
            Parameters[ego_k][name] = 0. * Parameters[ego_k][name]
            for l in range(0, N):
                if param.requires_grad:
                    Parameters[ego_k][name] += Parameters_exchange[l][name] * Weight[l]

        return Parameters, Accumulated_Loss

    # # Aggregation rule: Average
    # if rule == 'average':
    #     # Parameters = deepcopy(Parameters_exchange)
    #     # a = A[ego_k]
    #     Weight = []
    #     for l in range(0, N):
    #         if not l == ego_k:
    #             weight = 1 / N
    #         else:
    #             weight = 1 - (N - 1) / N
    #         Weight.append(weight)
    #     # print('Weights', Weight)
    #     for name, param in a.net.named_parameters():
    #         Parameters[ego_k][name] = 0. * Parameters[ego_k][name]
    #         for l in range(0, N):
    #         #     if not l == ego_k:
    #         #         weight = 1 / N
    #         #     else:
    #         #         weight = 1 - (N - 1) / N
    #             if param.requires_grad:
    #                 Parameters[ego_k][name] += Parameters_exchange[l][name] * Weight[l]
    #     return Parameters, Accumulated_Loss

    # # Aggregation rule: Coordinate-wise median
    # if rule == 'median':
    #     # Parameters = deepcopy(Parameters_exchange)
    #     # a = A[ego_k]
    #     for name, param in a.net.named_parameters():
    #         param_list = [Parameters_exchange[l][name] for l in range(0, N)]
    #         if param.requires_grad:
    #             Parameters[ego_k][name] = torch.stack(param_list).median(dim=0)[0]
    #     return Parameters, Accumulated_Loss
    #
    # # Aggregation rule: Trimmed mean
    # if rule == 'TM':
    #     # Parameters = deepcopy(Parameters_exchange)
    #     # a = A[ego_k]
    #     for name, param in a.net.named_parameters():
    #         param_list = [Parameters_exchange[l][name] for l in range(0, N)]
    #         if param.requires_grad:
    #             Parameters[ego_k][name] = torch.stack(param_list).sort(dim=0).values[N_a:-N_a].mean(dim=0)
    #     return Parameters, Accumulated_Loss
    #
    # # Aggregation rule: Medoid
    # if rule == 'medoid':
    #     for name, param in a.net.named_parameters():
    #         param_list = torch.stack([Parameters_exchange[l][name] for l in range(0, N)])
    #         if param.requires_grad:
    #             cdist = torch.cdist(param_list, param_list, p=2)
    #             print('cdist', cdist.shape)
    #             i_star = torch.argmin(cdist.sum(0))
    #             print('i star', i_star)
    #             return param_list[i_star][name]
    #     return Parameters, Accumulated_Loss



# def loss_adaptive(ego_k, A, batch_x, batch_y, Accumulated_Loss):
#     gamma = 0.05
#     N = len(A)
#     Weight = np.zeros((N,))
#     reversed_Loss = np.zeros((N,))
#     loss = A[ego_k].getLoss(batch_x, batch_y, A[ego_k].net)  # calculate the loss of agent k itself
#     Accumulated_Loss[ego_k, ego_k] = (1 - gamma) * Accumulated_Loss[ego_k, ego_k] + gamma * loss  # soft update
#     for l in range(0, N):
#         if not l == ego_k:
#             loss = A[ego_k].getLoss(batch_x, batch_y, A[l].net)  # find the loss using l's network but k's data
#             Accumulated_Loss[ego_k, l] = (1 - gamma) * Accumulated_Loss[ego_k, l] + gamma * loss
#         if Accumulated_Loss[ego_k, l] <= Accumulated_Loss[ego_k, ego_k]:
#             reversed_Loss[l] = 1. / Accumulated_Loss[ego_k, l]
#     sum_reversedLoss = sum(reversed_Loss)
#     for l in range(0, N):
#         if Accumulated_Loss[ego_k, l] <= Accumulated_Loss[ego_k, ego_k]:
#             weight = reversed_Loss[l] / sum_reversedLoss
#             Weight[l] = weight
#     return Weight, Accumulated_Loss
#
#
# def average(ego_k, A):
#     N = len(A)
#     Weight = []
#     for l in range(0, N):
#         if not l == ego_k:
#             weight = 1 / N
#         else:
#             weight = 1 - (N - 1) / N
#         Weight.append(weight)
#     return Weight


