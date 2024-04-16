import numpy as np
from copy import deepcopy
import random
from aggregation import *
import torch
from collections import defaultdict

def cooperation(A, A_last, Batch_X, Batch_Y, Accumulated_Loss, rule, attackers, attack_type):
    Parameters_last = []
    Parameters_exchange = []
    N = len(A)
    f = len(attackers)
    sigma = 1      # parameter for sign-flip attack

    if attack_type in ['empire', 'little']:
        Para_stack, Para_stack_last  = defaultdict(list), defaultdict(list)
        Para_avg, Para_avg_last = {}, {}

        for k in range(0, N):
            a = A[k]
            a_last = A_last[k]
            for name, param in a.net.named_parameters():
                Para_stack[name].append(param.data)
            for name, param in a_last.net.named_parameters():
                Para_stack_last[name].append(param.data)
        a0 = A[0]
        a0_last = A_last[0]
        for name, _ in a0.net.named_parameters():
            temp = torch.stack(Para_stack[name])
            Para_avg[name] = temp.mean(dim=0)
        for name, _ in a0_last.net.named_parameters():
            temp = torch.stack(Para_stack_last[name])
            Para_avg_last[name] = temp.mean(dim=0)

    for k in range(0, N):
        Parameters_last.append({})
        Parameters_exchange.append({})
        a_last = A_last[k]
        a = A[k]
        for name, param in a.net.named_parameters():
            if param.requires_grad:
                if k in attackers:
                    # a.net.named_parameters()[name] = param.data * random.random() * 0.1
                    if attack_type == 'arbitrary':
                        Parameters_exchange[k][name] = param.data * random.random() * 0.1
                    elif attack_type == 'sign_flip':
                        Parameters_exchange[k][name] = - sigma * param.data
                    elif attack_type == 'label_flip':
                        Parameters_exchange[k][name] = param.data
                    elif attack_type == 'empire':
                        Parameters_exchange[k][name] = Para_avg[name].neg()
                    elif attack_type == 'little':
                        Parameters_exchange[k][name] = torch.stack(Para_stack[name]).var(dim=0).sqrt_()
                else:
                    Parameters_exchange[k][name] = param.data
        for name, param in a_last.net.named_parameters():
            if param.requires_grad:
                if k in attackers:
                    # a_last.net.named_parameters()[name] = param.data * random.random() * 0.1
                    if attack_type == 'arbitrary':
                        Parameters_last[k][name] = param.data * random.random() * 0.1
                    elif attack_type == 'sign_flip':
                        Parameters_last[k][name] = - sigma * param.data
                    elif attack_type == 'label_flip':
                        Parameters_last[k][name] = param.data
                    elif attack_type == 'empire':
                        Parameters_last[k][name] = Para_avg_last[name].neg()
                    elif attack_type == 'little':
                        Parameters_last[k][name] = torch.stack(Para_stack_last[name]).var(dim=0).sqrt_()
                else:
                    Parameters_last[k][name] = param.data

    Para_ex = []
    Para_last = []
    for k in range(0, N):
        para_ex_k = np.hstack([v.flatten().tolist() for v in Parameters_exchange[k].values()])
        para_last_k = np.hstack([v.flatten().tolist() for v in Parameters_last[k].values()])
        Para_ex.append(para_ex_k)
        Para_last.append(para_last_k)
    Parameters = deepcopy(Parameters_exchange)      # dictionary
    # print('Parameter size', np.shape(Para_ex[0]), )

    # print('para ex', np.shape(Para_ex[0]))
    for k in range(0, N):
        a = A[k]
        if k not in attackers:
            # print('agent', k)
            batch_x, batch_y = Batch_X[k], Batch_Y[k]
            # Parameters, Accumulated_Loss = aggregate_parameters(k, A, Parameters_exchange, Para_ex, Para_last, batch_x, batch_y, Accumulated_Loss, rule, attackers)

            # for name, param in a.net.named_parameters():
            #     Parameters[k][name] = 0. * Parameters[k][name]
            #     for l in range(0, N):
            #         if param.requires_grad:
            #             Parameters[k][name], Accumulated_Loss = aggregate_parameters(k, A, Parameters_exchange, Para_ex, Para_last, batch_x, batch_y, Accumulated_Loss, rule)
            #
            #             Parameters[k][name] += Parameters_exchange[l][name] * Weight[l]

            if rule in ["loss", "average", "krum", "mKrum", "medoid"]:
                Weight, Accumulated_Loss = getWeight(k, A, Para_ex, Para_last, batch_x, batch_y, Accumulated_Loss, rule, attackers)

                for name, param in a.net.named_parameters():
                    Parameters[k][name] = 0. * Parameters[k][name]
                    for l in range(0, N):
                        if param.requires_grad:
                            Parameters[k][name] += Parameters_exchange[l][name] * Weight[l]
            elif rule == "CM":
                for name, param in a.net.named_parameters():
                    Parameters[k][name] = 0. * Parameters[k][name]
                    if param.requires_grad:
                        paras = torch.stack([Parameters_exchange[l][name] for l in range(0, N)])
                        med_paras = paras.median(dim=0)[0]
                        Parameters[k][name] = med_paras
            elif rule == "TM":
                for name, param in a.net.named_parameters():
                    Parameters[k][name] = 0. * Parameters[k][name]
                    if param.requires_grad:
                        paras = torch.stack([Parameters_exchange[l][name] for l in range(0, N)])
                        # print('paras', type(paras), paras.shape)
                        if f == 0:
                            f = int(0.1*N)
                        paras_sorted, _ = torch.sort(paras, dim=0)
                        med_paras = paras_sorted[f:-f].mean(dim=0)
                        # print('para med', type(med_paras), med_paras.shape)
                        Parameters[k][name] = med_paras



    # Assign the aggregated parameters to each agent
    for k in range(0, N):
        a = A[k]
        for name, param in a.net.named_parameters():
            param.data = Parameters[k][name]

    return A, Accumulated_Loss