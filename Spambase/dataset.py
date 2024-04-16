import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, Dataset


def readData_spambase():
    data = pd.read_csv('./Spambase Dataset/spambase.data', names=[x for x in range(58)])

    # First let's shuffle the dataset
    data = data.sample(frac=1).reset_index(drop=True)

    # Now lets split dataset into features and target
    Y = data[57]
    del data[57]
    X = data

    inputs = X.to_numpy()
    labels = Y.to_numpy()

    labels = torch.tensor(labels)
    inputs = torch.tensor(inputs)

    # splitting training and test data
    pct_test = 0.2

    train_labels = labels[:-int(len(labels) * pct_test)]
    train_inputs = inputs[:-int(len(labels) * pct_test)]

    test_labels = labels[-int(len(labels) * pct_test):]
    test_inputs = inputs[-int(len(labels) * pct_test):]

    train_data, test_data = [], []

    for i in range(len(train_inputs)):
        train_data.append([train_inputs[i], train_labels[i]])
    for i in range(len(test_inputs)):
        test_data.append([test_inputs[i], test_labels[i]])

    # return train_data, test_data
    return train_inputs, test_inputs, train_labels, test_labels



def generateData_spambase_iid(train_data, test_data, train_labels, test_labels, tr_split_len, te_split_len, number, batch_size, batch_size_test, LF_flag):
    train_x_no = train_data[number * tr_split_len: (number + 1) * tr_split_len].float() / 255
    train_y_no = train_labels[number * tr_split_len: (number + 1) * tr_split_len]
    test_x_no = test_data[number * te_split_len: (number + 1) * te_split_len].float() / 255
    test_y_no = test_labels[number * te_split_len: (number + 1) * te_split_len]
    # print(type(train_x_no), train_x_no)

    train_data_no = []
    for i in range(len(train_x_no)):
        if LF_flag:
            train_data_no.append([train_x_no[i].unsqueeze(0), (1 - train_y_no[i])])
        else:
            train_data_no.append([train_x_no[i].unsqueeze(0), train_y_no[i]])

    test_data_no = []
    for i in range(len(test_x_no)):
        test_data_no.append([test_x_no[i].unsqueeze(0), test_y_no[i]])

    train_loader_no = DataLoader(dataset=train_data_no, batch_size=batch_size, shuffle=True)
    test_loader_no = DataLoader(dataset=test_data_no, batch_size=batch_size_test)
    print('k=', number, len(train_data_no), len(test_data_no))
    return train_loader_no, test_loader_no


def generateData_spambase_niid_pathological(train_input, test_input, train_labels, test_labels, num_workers, batch_size, batch_size_test, LF_attackers):
    # Create datasets for each worker based on the selected classes
    train_index_class1 = (train_labels == 0).nonzero().squeeze()
    train_index_class2 = (train_labels == 1).nonzero().squeeze()
    test_index_class1 = (test_labels == 0).nonzero().squeeze()
    test_index_class2 = (test_labels == 1).nonzero().squeeze()
    train_input_class1, train_input_class2 = train_input[train_index_class1], train_input[train_index_class2]
    test_input_class1, test_input_class2 = test_input[test_index_class1], test_input[test_index_class2]
    train_label_class1, train_label_class2 = train_labels[train_index_class1], train_labels[train_index_class2]
    test_label_class1, test_label_class2 = test_labels[test_index_class1], test_labels[test_index_class2]
    class1_tr_len, class2_tr_len = train_index_class1.shape[0]//6, train_index_class2.shape[0]//4
    class1_te_len, class2_te_len = test_index_class1.shape[0] // 6, test_index_class2.shape[0] // 4
    worker_train_loaders, worker_test_loaders = [], []
    # print('partition', class1_tr_len, class2_tr_len, class1_te_len, class2_te_len)
    # print(type(train_index_class1), train_index_class1.shape)
    # print(type(train_index_class2), train_index_class2.shape)
    # print(type(test_index_class1), test_index_class1.shape)
    # print(type(test_index_class2), test_index_class2.shape)

    for k in range(0, 6):
        try:
            train_x_no = train_input_class1[k * class1_tr_len: (k + 1) * class1_tr_len].float()
            train_y_no = train_label_class1[k * class1_tr_len: (k + 1) * class1_tr_len]
            test_x_no = test_input_class1[k * class1_te_len: (k + 1) * class1_te_len].float()
            test_y_no = test_label_class1[k * class1_te_len: (k + 1) * class1_te_len]
        except:
            train_x_no = train_input_class1[k * class1_tr_len:].float()
            train_y_no = train_label_class1[k * class1_tr_len:]
            test_x_no = test_input_class1[k * class1_te_len:].float()
            test_y_no = test_label_class1[k * class1_te_len:]
        train_data_no, test_data_no = [], []
        for i in range(len(train_x_no)):
            train_data_no.append([train_x_no[i].unsqueeze(0), train_y_no[i]])
        for i in range(len(test_x_no)):
            test_data_no.append([test_x_no[i].unsqueeze(0), test_y_no[i]])
        # print('k=', k, len(train_data_no), len(test_data_no))
        print('train_data_no', type(train_data_no[0]))
        worker_train_loaders.append(DataLoader(dataset=train_data_no, batch_size=batch_size, shuffle=True))
        worker_test_loaders.append(DataLoader(dataset=test_data_no, batch_size=batch_size_test, shuffle=True))
    print('worker_train_loaders', type(worker_train_loaders), type(worker_train_loaders[0]))
    for k in range(0, 4):
        try:
            train_x_no = train_input_class2[k * class2_tr_len: (k + 1) * class2_tr_len].float()
            train_y_no = train_label_class2[k * class2_tr_len: (k + 1) * class2_tr_len]
            test_x_no = test_input_class2[k * class2_te_len: (k + 1) * class2_te_len].float()
            test_y_no = test_label_class2[k * class2_te_len: (k + 1) * class2_te_len]
        except:
            train_x_no = train_input_class2[k * class2_tr_len:].float()
            train_y_no = train_label_class2[k * class2_tr_len:]
            test_x_no = test_input_class2[k * class2_te_len:].float()
            test_y_no = test_label_class2[k * class2_te_len:]
        # print('size', train_x_no.shape)
        train_data_no, test_data_no = [], []
        for i in range(len(train_y_no)):
            train_data_no.append([train_x_no[i].unsqueeze(0), train_y_no[i]])
        for i in range(len(test_y_no)):
            test_data_no.append([test_x_no[i].unsqueeze(0), test_y_no[i]])
        print('k=', k, len(train_data_no), len(test_data_no))
        print('train_data_no[0]', type(train_data_no[0]), len(train_data_no[0]), train_data_no[0][0].shape)
        worker_train_loaders.append(DataLoader(dataset=train_data_no, batch_size=batch_size, shuffle=True))
        worker_test_loaders.append(DataLoader(dataset=test_data_no, batch_size=batch_size_test, shuffle=True))
    # print('loaders', len(worker_train_loaders))

    return worker_train_loaders, worker_test_loaders


# class FedDataset(Dataset):
#     def __init__(self, dataset, indx):
#         self.dataset = dataset
#         self.indx = [int(i) for i in indx]
#         # self.label_flip = flip
#
#     def __len__(self):
#         return len(self.indx)
#
#     def __getitem__(self, item):
#         images, label = self.dataset[self.indx[item]]
#         # if self.label_flip == True:
#         #     label = 9 - label
#         # print(label)
#         return torch.tensor(images).clone().detach(), torch.tensor(label).clone().detach()

def generateData_spambase_niid_practical(train_input, test_input, train_labels, test_labels, num_workers, batch_size, batch_size_test, LF_attackers):
    # classes, images = 50, 92
    train_labels = train_labels[:-1]
    classes, images = 20, 184       # 40, 92
    classes_indx = [i for i in range(classes)]
    train_group = {i: np.array([]) for i in range(num_workers)}
    indeces = np.arange(classes * images)
    unsorted_labels = train_labels.numpy()

    indeces_unsortedlabels = np.vstack((indeces, unsorted_labels))
    indeces_labels = indeces_unsortedlabels[:, indeces_unsortedlabels[1, :].argsort()]
    indeces = indeces_labels[0, :]
    # print('indeces', len(indeces))

    for i in range(num_workers):
        np.random.seed(i)
        temp = set(np.random.choice(classes_indx, 2, replace=False))
        classes_indx = list(set(classes_indx) - temp)

        for t in temp:
            train_group[i] = np.concatenate((train_group[i], indeces[t * images:(t + 1) * images]), axis=0)


    classes, images = 20, 46
    classes_indx = [i for i in range(classes)]
    test_group = {i: np.array([]) for i in range(num_workers)}
    indeces = np.arange(classes * images)
    unsorted_labels = test_labels.numpy()

    indeces_unsortedlabels = np.vstack((indeces, unsorted_labels))
    indeces_labels = indeces_unsortedlabels[:, indeces_unsortedlabels[1, :].argsort()]
    indeces = indeces_labels[0, :]

    for i in range(num_workers):
        np.random.seed(i)
        temp = set(np.random.choice(classes_indx, 2, replace=False))
        classes_indx = list(set(classes_indx) - temp)
        for t in temp:
            test_group[i] = np.concatenate((test_group[i], indeces[t * images:(t + 1) * images]), axis=0)

    # for i in range(num_workers):
    #     print(i, 'train_group[i]', train_group[i].shape)
    #     print(i, 'test_group[i]', test_group[i].shape)

    worker_train_loaders, worker_test_loaders = [], []
    for k in range(num_workers):
        train_x_no, train_y_no = train_input[train_group[k]].float(), train_labels[train_group[k]]
        test_x_no, test_y_no = test_input[test_group[k]].float(), test_labels[test_group[k]]
        train_data_no, test_data_no = [], []
        for i in range(len(train_y_no)):
            train_data_no.append([train_x_no[i].unsqueeze(0), train_y_no[i]])
        for i in range(len(test_y_no)):
            test_data_no.append([test_x_no[i].unsqueeze(0), test_y_no[i]])
        # print('k=', k, len(train_data_no), len(test_data_no))
        # print('train_data_no[0]', type(train_data_no[0]), len(train_data_no[0]), train_data_no[0][0].shape)
        worker_train_loaders.append(DataLoader(dataset=train_data_no, batch_size=batch_size, shuffle=True))
        worker_test_loaders.append(DataLoader(dataset=test_data_no, batch_size=batch_size_test, shuffle=True))

    return worker_train_loaders, worker_test_loaders