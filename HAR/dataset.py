import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def readData_har():
    features = pd.read_csv('./UCI HAR Dataset/features.txt', sep='\s+', index_col=0, header=None)
    # train_data = pd.read_csv('./UCI HAR Dataset/train/X_train.txt', sep='\s+',
    #                          names=list(features.values.ravel()))
    # test_data = pd.read_csv('./UCI HAR Dataset/test/X_test.txt', sep='\s+',
    #                         names=list(features.values.ravel()))
    train_data = pd.read_csv('UCI HAR Dataset/train/X_train.txt', delim_whitespace=True, header=None)
    test_data = pd.read_csv('UCI HAR Dataset/test/X_test.txt', delim_whitespace=True, header=None)

    train_label = pd.read_csv('./UCI HAR Dataset/train/y_train.txt', sep='\s+', header=None)
    test_label = pd.read_csv('./UCI HAR Dataset/test/y_test.txt', sep='\s+', header=None)

    train_subject = pd.read_csv('./UCI HAR Dataset/train/subject_train.txt', sep='\s+', header=None)
    test_subject = pd.read_csv('./UCI HAR Dataset/test/subject_test.txt', sep='\s+', header=None)

    label_name = pd.read_csv('UCI HAR Dataset/activity_labels.txt', sep='\s+', header=None, index_col=0)

    train_data['label'] = train_label
    test_data['label'] = test_label

    train_data['subject'] = train_subject
    test_data['subject'] = test_subject

    def get_label_name(num):
        return label_name.iloc[num - 1, 0]

    train_data['label_name'] = train_data['label'].map(get_label_name)
    test_data['label_name'] = test_data['label'].map(get_label_name)

    train_data['label'] = train_data['label'] - 1
    test_data['label'] = test_data['label'] - 1

    np.random.shuffle(train_data.values)
    np.random.shuffle(test_data.values)

    return train_data, test_data



def generateData_har(train_data, test_data, subject, batch_size, LF_flag):
    x_train = [d[:-3] for d in train_data.values if d[-2] == subject]
    y_train = [d[-3] for d in train_data.values if d[-2] == subject]
    if LF_flag:
        y_train = [5 - x for x in y_train]
    x_test = [d[:-3] for d in test_data.values if d[-2] == subject]
    y_test = [d[-3] for d in test_data.values if d[-2] == subject]

    all_x_data = x_train + x_test
    all_y_data = y_train + y_test

    x_tensor = torch.FloatTensor(all_x_data)
    y_tensor = torch.LongTensor(all_y_data)

    all_data = []
    for i in range(len(x_tensor)):
        all_data.append([x_tensor[i], y_tensor[i]])

    np.random.shuffle(all_data)

    train_data_subject, val_data_subject, test_data_subject = all_data[:len(all_data) // 4 * 3], \
                                                              all_data[
                                                              len(all_data) // 4 * 3: len(all_data) // 8 * 7], all_data[
                                                                                                               len(
                                                                                                                   all_data) // 4 * 3:]


    train_loader_subject = DataLoader(dataset=train_data_subject, batch_size=batch_size, shuffle=True)
    test_loader_subject = DataLoader(dataset=test_data_subject, batch_size=batch_size, shuffle=True)
    val_loader_subject = DataLoader(dataset=val_data_subject, batch_size=batch_size, shuffle=True)

    return train_loader_subject, val_loader_subject, test_loader_subject