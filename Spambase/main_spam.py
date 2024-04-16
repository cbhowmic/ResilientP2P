# Import the packages
import math
import os
import random
import time
from copy import deepcopy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import sys

# Import other codeblocks
from agent import agent
from dataset import *
from network import *
from cooperation import *


def process_commandline():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--seed",
                        type=int,
                        default=-1,
                        help="Fixed seed to use for reproducibility purpose, negative for random seed")
    parser.add_argument("--epoch",
                        type=int,
                        default=200,
                        help="Number of (additional) training steps to do, negative for no limit" )
    parser.add_argument("--nb-workers",
                        type=int,
                        default=10,
                        help="Total number of worker machines")
    parser.add_argument("--nb-attackers",
                        type=int,
                        default=3,
                        help="Number of Byzantine worker(s)")
    parser.add_argument("--gar",
                        type=str,
                        default="krum",
                        help="(Byzantine-resilient) aggregation rule to use")
    parser.add_argument("--temperature",
                        type=int,
                        default=1000,
                        help="Temperature value used in SM/TSM aggregation")
    parser.add_argument("--attack",
                        type=str,
                        default="arbitrary",
                        help="Attack to use")
    parser.add_argument("--model",
                        type=str,
                        default="MLP_1layer",
                        help="Model to train")
    parser.add_argument("--loss",
                        type=str,
                        default="nll",
                        help="Loss to use")
    parser.add_argument("--batch-size",
                        type=int,
                        default=20,
                        help="Batch-size to use for training")
    parser.add_argument("--batch-size-test",
                        type=int,
                        default=5,
                        help="Batch-size to use for testing")
    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.005,  # 0.01
                        help="Learning rate to use for training")
    parser.add_argument("--learning-rate-decay",
                        type=int,
                        default=100,        # 50
                        help="Learning rate hyperbolic half-decay time, non-positive for no decay")
    parser.add_argument("--learning-rate-decay-delta",
                        type=int,
                        default=1,
                        help="How many steps between two learning rate updates, must be a positive integer")
    parser.add_argument("--dataset",
                        type=str,
                        default='spambase',
                        help="dataset in use")
    parser.add_argument("--data_distribution",
                        type=str,
                        default='niid_practical',     # 'niid_practical',      # niid_pathological
                        help="iid or non-iid (pathological or practical) distribution")
    parser.add_argument("--save-data",
                        type=bool,
                        default=True,
                        help="Whether or not to save data")
    parser.add_argument("--evaluation-delta",
                        type=int,
                        default=10,
                        help="How frequently to evaluate the algorithm")
    return parser.parse_args(sys.argv[1:])


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')  # don't have GPU
    return device


def gaussian(x, mean, stddev):
    noise = Variable(x.new(x.size()).normal_(mean, stddev))
    return x + noise


torch.manual_seed(1)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)

args = process_commandline()

# Create directory for saving results
if args.save_data:
    if args.nb_attackers > 0:
        if args.gar in {'SM', 'TSM'}:
            args.result_directory = 'results_spambase_%s_%s/%s_attack/%s_%dagents_%dattacker_T%d' % (args.data_distribution, args.model,
                                   args.attack, args.gar, args.nb_workers, args.nb_attackers, args.temperature)
        else:
            args.result_directory = 'results_spambase_%s_%s/%s_attack/%s_%dagents_%dattacker' % (args.data_distribution, args.model,
                                    args.attack, args.gar, args.nb_workers, args.nb_attackers)
    else:
        if args.gar in {'SM', 'TSM'}:
            args.result_directory = 'results_spambase_%s_%s/noAttack/%s_%dagents_%dattacker_T%d' % (args.data_distribution, args.model,
                                                                            args.gar, args.nb_workers, args.nb_attackers, args.temperature)
        else:
            args.result_directory = 'results_spambase_%s_%s/noAttack/%s_%dagents_%dattacker' % (args.data_distribution, args.model,
                                                                            args.gar, args.nb_workers, args.nb_attackers)
else:
    args.result_directory = None

if args.dataset == 'spambase':
    # Read dataset
    # train_data, test_data = readData_spambase()
    train_input, test_input, train_labels, test_labels = readData_spambase()
    print('train size', len(train_labels), 'test size', len(test_labels))
# train_split_len = int(len(train_data) / args.nb_workers)
# test_split_len = int(len(test_data) / args.nb_workers)
train_split_len = int(len(train_labels) / args.nb_workers)
test_split_len = int(len(test_labels) / args.nb_workers)


Agents = []
Parameters = []
Train_loader, Test_loader = [], []
# Val_loader_iter, Val_loader = [], []
Accumulated_Loss = np.ones((args.nb_workers, args.nb_workers))

average_train_loss, average_train_acc = [], []
average_test_loss, average_test_acc = [], []
individual_average_train_loss, individual_average_train_acc = np.zeros((args.epoch, args.nb_workers)), np.zeros((args.epoch, args.nb_workers))
individual_average_test_loss, individual_average_test_acc = np.zeros((args.epoch, args.nb_workers)), np.zeros((args.epoch, args.nb_workers))

# The attackers
attackers = random.sample(range(0, args.nb_workers), args.nb_attackers)
print('The attackers:', attackers)
normalAgents = [x for x in range(args.nb_workers) if x not in attackers]
# Saving the normal agents
try:
    os.makedirs('%s' % args.result_directory)
except OSError:
    print("Creation of the result directory failed")
np.save('%s/normalAgents.npy' % args.result_directory, normalAgents)


for k in range(0, args.nb_workers):
    if args.model == 'MLP_1layer':
        net = MLPclassification()
    if args.model == 'MLP_2layer':
        net = MLP()
    a = agent(net, args.learning_rate)
    Agents.append(a)
    if args.attack == 'label_flip' and k in attackers:
        attack_LF = True
    else:
        attack_LF = False

    if args.data_distribution == 'iid':
        train_loader_no, test_loader_no = generateData_spambase_iid(train_input, test_input, train_labels, test_labels,
                                         train_split_len, test_split_len, k, args.batch_size, args.batch_size_test, attack_LF)
        Train_loader.append(train_loader_no)
        Test_loader.append(test_loader_no)


if args.data_distribution == 'niid_pathological':
    if args.attack == 'label_flip' and args.nb_attackers > 0:
        LF_attackers = attackers
    else:
        LF_attackers = []
    Train_loader, Test_loader = generateData_spambase_niid_pathological(train_input, test_input, train_labels, test_labels,
                                              args.nb_workers, args.batch_size, args.batch_size_test, LF_attackers)


if args.data_distribution == 'niid_practical':
    if args.attack == 'label_flip' and args.nb_attackers > 0:
        LF_attackers = attackers
    else:
        LF_attackers = []
    train_loader_list, test_loader_list = generateData_spambase_niid_practical(train_input, test_input, train_labels, test_labels,
                                              args.nb_workers, args.batch_size, args.batch_size_test, LF_attackers)
    Train_loader, Test_loader = train_loader_list, test_loader_list

    # for k in range(0, args.nb_workers):
    #     Train_loader.append(iter(train_loader_list[k]))
    #     Test_loader.append(test_loader_list[k])


print("Training...")
start_time = time.time()

for round in range(args.epoch):
    print('epoch {}'.format(round + 1))
    Train_loader_iter = []
    Test_loader_iter = []
    total_train_loss = 0.
    total_train_acc = 0.
    total_eval_loss = 0.
    total_eval_acc = 0.
    Count = np.zeros((args.nb_workers,))
    ave_train_loss = 0.
    ave_train_acc = 0.
    ave_eval_loss = 0.
    ave_eval_acc = 0.
    nanCount = 0

    for k in range(0, args.nb_workers):
        a = Agents[k]
        a.train_loss = 0.
        a.train_acc = 0.

        Train_loader_iter.append(iter(Train_loader[k]))

    try:
        while True:
            Agents_last = deepcopy(Agents)
            Batch_X, Batch_Y = {}, {}
            for k in range(0, args.nb_workers):
                # print('next(Train_loader_iter[k])', type(next(Train_loader_iter[k])))
                batch_x, batch_y = next(Train_loader_iter[k])
                Batch_X[k] = batch_x.to(device)
                Batch_Y[k] = batch_y.to(device)
                if k in attackers:
                    continue

                a = Agents[k]
                # print('k:', k)
                loss, acc = a.optimize(batch_x.to(device), batch_y.to(device))  # the agent networks are update using standard Ml algorithm (step before aggregation)
                # if math.isnan(loss) or math.isnan(acc):
                #     continue
                # print('agent', k, 'loss', loss)
                total_train_loss += loss
                total_train_acc += acc
                Count[k] += len(batch_x)

            # the aggregation step: the returned A is the updated agent parameters after the aggregation step
            Agents, Accumulated_Loss = cooperation(Agents, Agents_last, Batch_X, Batch_Y, Accumulated_Loss, args.gar, attackers, args.attack)

    except StopIteration:
        Eval_count = np.zeros((args.nb_workers,))
        for k in range(0, args.nb_workers):
            if k in attackers:
                continue
            print('Agent: {:d}, Train Loss: {:.6f}, Acc: {:.6f}'.format(k, Agents[k].train_loss / Count[k], Agents[k].train_acc / Count[k]))
            individual_average_train_loss[round, k] = Agents[k].train_loss / Count[k]
            individual_average_train_acc[round, k] = Agents[k].train_acc / Count[k]

            if not (math.isnan(Agents[k].train_loss / Count[k]) or math.isnan(Agents[k].train_acc / Count[k])):
                ave_train_loss += Agents[k].train_loss / Count[k]
                ave_train_acc += Agents[k].train_acc / Count[k]
            else:
                nanCount += 1

            # evaluation--------------------------------
            Agents[k].net.eval()
            eval_loss = 0.
            eval_acc = 0.
            for batch_x, batch_y in Test_loader[k]:
                batch_x, batch_y = Variable(batch_x, volatile=True).to(device), Variable(batch_y, volatile=True).to(device)
                out = Agents[k].net(batch_x).squeeze(1)
                loss_func = torch.nn.CrossEntropyLoss()
                loss = loss_func(out, batch_y)
                pred = torch.max(out, 1)[1]
                num_correct = (pred == batch_y).sum()
                if math.isnan(loss) or math.isnan(num_correct):
                    continue
                eval_loss += loss.item()
                total_eval_loss += loss.item()
                eval_acc += num_correct.item()
                total_eval_acc += num_correct.item()
                Eval_count[k] += len(batch_x)

            if not (math.isnan(eval_loss / Eval_count[k]) or math.isnan(eval_acc / Eval_count[k])):
                ave_eval_loss += eval_loss / Eval_count[k]
                ave_eval_acc += eval_acc / Eval_count[k]
            print('Agent: {:d}, Test Loss: {:.6f}, Acc: {:.6f}'.format(k, eval_loss / Eval_count[k], eval_acc / Eval_count[k]))
            individual_average_test_loss[round, k] = eval_loss / Eval_count[k]
            individual_average_test_acc[round, k] = eval_acc / Eval_count[k]

    print('Total Average Train Loss: {:.6f}, Train Acc: {:.6f}'.format(
        ave_train_loss / (args.nb_workers - args.nb_attackers),
        ave_train_acc / (args.nb_workers - args.nb_attackers)))
    average_train_loss.append(ave_train_loss / (args.nb_workers - nanCount - args.nb_attackers))
    average_train_acc.append(ave_train_acc / (args.nb_workers - nanCount - args.nb_attackers))
    print('Total Average Test Loss: {:.6f}, Test Acc: {:.6f}'.format(
        ave_eval_loss / (args.nb_workers - args.nb_attackers),
        ave_eval_acc / (args.nb_workers - args.nb_attackers)))

    print('Training time by far: {:.2f}s'.format(time.time() - start_time))
    average_test_loss.append(ave_eval_loss / (args.nb_workers - nanCount - args.nb_attackers))
    average_test_acc.append(ave_eval_acc / (args.nb_workers - nanCount - args.nb_attackers))

    # saving data
    if round % args.evaluation_delta == 0 or round == args.epoch - 1:
        try:
            os.makedirs('%s' % args.result_directory)
        except OSError:
            print("Creation of the result directory failed")
        np.save('%s/average_train_loss.npy' % args.result_directory, average_train_loss)
        np.save('%s/average_train_acc.npy' % args.result_directory, average_train_acc)
        np.save('%s/average_test_loss.npy' % args.result_directory, average_test_loss)
        np.save('%s/average_test_acc.npy' % args.result_directory, average_test_acc)
        np.save('%s/individual_average_train_loss.npy' % args.result_directory, individual_average_train_loss)
        np.save('%s/individual_average_train_acc.npy' % args.result_directory, individual_average_train_acc)
        np.save('%s/individual_average_test_loss.npy' % args.result_directory, individual_average_test_loss)
        np.save('%s/individual_average_test_acc.npy' % args.result_directory, individual_average_test_acc)

print('Training complete!')
print('Total training time: {:.2f}s'.format(time.time() - start_time))