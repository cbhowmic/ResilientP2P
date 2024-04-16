import torchvision
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import torch

def readData_mnist():
    train_data = torchvision.datasets.MNIST(
        './mnist', train=True, transform=torchvision.transforms.ToTensor(), download=True
    )
    test_data = torchvision.datasets.MNIST(
        './mnist', train=False, transform=torchvision.transforms.ToTensor()
    )
    print("train_data:", train_data.train_data.size())
    print("train_labels:", train_data.train_labels.size())
    print("test_data:", test_data.test_data.size())
    print("train_labels:", test_data.test_labels.size())
    return train_data, test_data


def generateData_mnist_iid(train_data, test_data, tr_split_len, te_split_len, number, LF_flag):
    train_x_no = train_data.train_data[number * tr_split_len: (number + 1) * tr_split_len].float() / 255
    train_y_no = train_data.train_labels[number * tr_split_len: (number + 1) * tr_split_len]
    test_x_no = test_data.test_data[number * te_split_len: (number + 1) * te_split_len].float() / 255
    test_y_no = test_data.test_labels[number * te_split_len: (number + 1) * te_split_len]

    train_data_no = []
    if LF_flag:
        # wrong label
        for i in range(len(train_x_no)):
            train_data_no.append([train_x_no[i].unsqueeze(0), (9 - train_y_no[i])])
    else:
        for i in range(len(train_x_no)):
            train_data_no.append([train_x_no[i].unsqueeze(0), train_y_no[i]])

    test_data_no = []
    for i in range(len(test_x_no)):
        test_data_no.append([test_x_no[i].unsqueeze(0), test_y_no[i]])

    train_loader_no = DataLoader(dataset=train_data_no, batch_size=64, shuffle=True)
    test_loader_no = DataLoader(dataset=test_data_no, batch_size=64)

    return train_loader_no, test_loader_no


def generateData_mnist_niid_pathological(train_data, test_data, num_workers, LF_attackers):
    # print('LF attackers', LF_attackers)
    # Define the classes to include for each worker
    if num_workers == 10:
        classes_per_worker = [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
            [0, 2],
            [4, 6],
            [8, 0],
            [2, 4],
            [6, 8]
        ]

    # Create datasets for each worker based on the selected classes
    worker_train = [Subset(train_data, [i for i in range(len(train_data))
                                                if train_data.targets[i] in classes]) for classes in classes_per_worker]
    worker_test = [Subset(test_data, [i for i in range(len(test_data))
                                                if test_data.targets[i] in classes]) for classes in classes_per_worker]
    # print(worker_train[0].dataset.targets)
    for k in LF_attackers:
        worker_train[k].dataset.targets = 9 - worker_train[k].dataset.targets

    # Convert each worker dataset to DataLoader
    worker_train_loaders = [DataLoader(dataset, batch_size=64, shuffle=True) for dataset in worker_train]
    worker_test_loaders = [DataLoader(dataset, batch_size=64, shuffle=True) for dataset in worker_test]

    return worker_train_loaders, worker_test_loaders


class FedDataset(Dataset):
    def __init__(self, dataset, indx):
        self.dataset = dataset
        self.indx = [int(i) for i in indx]
        # self.label_flip = flip

    def __len__(self):
        return len(self.indx)

    def __getitem__(self, item):
        images, label = self.dataset[self.indx[item]]
        # if self.label_flip == True:
        #     label = 9 - label
        # print(label)
        return torch.tensor(images).clone().detach(), torch.tensor(label).clone().detach()

def generateData_mnist_niid_practical(train_data, test_data, num_workers, batch_size, batch_size_test, LF_attackers):
    classes, images = 200, 300
    classes_indx = [i for i in range(classes)]
    train_group = {i: np.array([]) for i in range(num_workers)}
    indeces = np.arange(classes * images)
    unsorted_labels = train_data.train_labels.numpy()

    indeces_unsortedlabels = np.vstack((indeces, unsorted_labels))
    indeces_labels = indeces_unsortedlabels[:, indeces_unsortedlabels[1, :].argsort()]
    indeces = indeces_labels[0, :]

    for i in range(num_workers):
        np.random.seed(i)
        temp = set(np.random.choice(classes_indx, 2, replace=False))
        classes_indx = list(set(classes_indx) - temp)
        for t in temp:
            train_group[i] = np.concatenate((train_group[i], indeces[t * images:(t + 1) * images]), axis=0)

    classes, images = 20, 500
    classes_indx = [i for i in range(classes)]
    test_group = {i: np.array([]) for i in range(num_workers)}
    indeces = np.arange(classes * images)
    unsorted_labels = test_data.train_labels.numpy()

    indeces_unsortedlabels = np.vstack((indeces, unsorted_labels))
    indeces_labels = indeces_unsortedlabels[:, indeces_unsortedlabels[1, :].argsort()]
    indeces = indeces_labels[0, :]

    for i in range(num_workers):
        np.random.seed(i)
        temp = set(np.random.choice(classes_indx, 2, replace=False))
        classes_indx = list(set(classes_indx) - temp)
        for t in temp:
            test_group[i] = np.concatenate((test_group[i], indeces[t * images:(t + 1) * images]), axis=0)

    worker_train_loaders, worker_test_loaders = [], []
    for inx in range(num_workers):
        trainset_ind_list, testset_ind_list = list(train_group[inx]), list(test_group[inx])

        worker_train_loaders.append(DataLoader(FedDataset(train_data, trainset_ind_list), batch_size=batch_size, shuffle=True))
        worker_test_loaders.append(DataLoader(FedDataset(test_data, testset_ind_list), batch_size=batch_size_test, shuffle=True))

    return worker_train_loaders, worker_test_loaders


def readData_synthetic_digits():
    transformtrain = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    transformtest = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = torchvision.datasets.ImageFolder('synthetic_digits/synthetic_digits/imgs_train',
                                                  transform=transformtrain)
    test_data = torchvision.datasets.ImageFolder('synthetic_digits/synthetic_digits/imgs_valid',
                                                 transform=transformtest)

    return train_data, test_data


def generateData_synthetic_digits(remaining_tr, remaining_te, tr_split_len, te_split_len):
    part_tr, part_tr2 = torch.utils.data.random_split(remaining_tr, [tr_split_len, len(remaining_tr) - tr_split_len])
    part_te, part_te2 = torch.utils.data.random_split(remaining_te, [te_split_len, len(remaining_te) - te_split_len])

    train_loader_no = DataLoader(part_tr, batch_size=128, shuffle=True)
    test_loader_no = DataLoader(part_te, batch_size=128, shuffle=False)

    return train_loader_no, test_loader_no, part_tr2, part_te2