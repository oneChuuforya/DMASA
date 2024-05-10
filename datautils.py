import os

import numpy as np
import torch
from scipy.io import arff
from sklearn.preprocessing._data import StandardScaler
from sr.sr_evalue import sr
import pandas as pd
import RevIN

def padding_varying_length(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j, :][np.isnan(data[i, j, :])] = 0
    return data


def load_UCR(Path='data/', folder='Cricket'):
    train_path = Path + folder + '/' + folder + '_TRAIN.arff'
    test_path = Path + folder + '/' + folder + '_TEST.arff'
    TRAIN_DATA = []
    TRAIN_LABEL = []
    label_dict = {}
    label_index = 0
    with open(train_path, encoding='UTF-8', errors='ignore') as f:
        data, meta = arff.loadarff(f)
        f.close()
    if type(data[0][0]) == np.ndarray:  # multivariate
        for index in range(data.shape[0]):
            raw_data = data[index][0]
            raw_label = data[index][1]
            if label_dict.__contains__(raw_label):
                TRAIN_LABEL.append(label_dict[raw_label])
            else:
                label_dict[raw_label] = label_index
                TRAIN_LABEL.append(label_index)
                label_index += 1
            raw_data_list = raw_data.tolist()
            # print(raw_data_list)
            TRAIN_DATA.append(np.array(raw_data_list).astype(np.float32).transpose(-1, 0))

        TEST_DATA = []
        TEST_LABEL = []
        with open(test_path, encoding='UTF-8', errors='ignore') as f:
            data, meta = arff.loadarff(f)
            f.close()
        for index in range(data.shape[0]):
            raw_data = data[index][0]
            raw_label = data[index][1]
            TEST_LABEL.append(label_dict[raw_label])
            raw_data_list = raw_data.tolist()
            TEST_DATA.append(np.array(raw_data_list).astype(np.float32).transpose(-1, 0))

        index = np.arange(0, len(TRAIN_DATA))
        np.random.shuffle(index)

        TRAIN_DATA = padding_varying_length(np.array(TRAIN_DATA))
        TEST_DATA = padding_varying_length(np.array(TEST_DATA))

        return [np.array(TRAIN_DATA)[index], np.array(TRAIN_LABEL)[index]], \
               [np.array(TRAIN_DATA)[index], np.array(TRAIN_LABEL)[index]], \
               [np.array(TEST_DATA), np.array(TEST_LABEL)]

    else:  # univariate
        for index in range(data.shape[0]):
            raw_data = np.array(list(data[index]))[:-1]
            raw_label = data[index][-1]
            if label_dict.__contains__(raw_label):
                TRAIN_LABEL.append(label_dict[raw_label])
            else:
                label_dict[raw_label] = label_index
                TRAIN_LABEL.append(label_index)
                label_index += 1
            TRAIN_DATA.append(np.array(raw_data).astype(np.float32).reshape(-1, 1))

        TEST_DATA = []
        TEST_LABEL = []
        with open(test_path, encoding='UTF-8', errors='ignore') as f:
            data, meta = arff.loadarff(f)
            f.close()
        for index in range(data.shape[0]):
            raw_data = np.array(list(data[index]))[:-1]
            raw_label = data[index][-1]
            TEST_LABEL.append(label_dict[raw_label])
            TEST_DATA.append(np.array(raw_data).astype(np.float32).reshape(-1, 1))

        TRAIN_DATA = padding_varying_length(np.array(TRAIN_DATA))
        TEST_DATA = padding_varying_length(np.array(TEST_DATA))

        return [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [
            np.array(TEST_DATA), np.array(TEST_LABEL)]

def load_mat(Path='data/AUSLAN/'):
    if 'UWave' in Path:
        train = torch.load(Path + 'train_new.pt')
        test = torch.load(Path + 'test_new.pt')
    else:
        train = torch.load(Path + 'train.pt')
        test = torch.load(Path + 'test.pt')
    TRAIN_DATA = train['samples'].float()
    TRAIN_LABEL = (train['labels'] - 1).long()
    TEST_DATA = test['samples'].float()
    TEST_LABEL = (test['labels'] - 1).long()
    print('data loaded')

    return [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [
        np.array(TEST_DATA), np.array(TEST_LABEL)]

def load_HAR(Path='data/HAR/'):
    train = torch.load(Path + 'train.pt')
    val = torch.load(Path + 'val.pt')
    test = torch.load(Path + 'test.pt')
    TRAIN_DATA = train['samples'].transpose(1, 2).float()
    TRAIN_LABEL = train['labels'].long()
    VAL_DATA = val['samples'].transpose(1, 2).float()
    VAL_LABEL = val['labels'].long()
    TEST_DATA = test['samples'].transpose(1, 2).float()
    TEST_LABEL = test['labels'].long()

    ALL_TRAIN_DATA = torch.cat([TRAIN_DATA, VAL_DATA])
    ALL_TRAIN_LABEL = torch.cat([TRAIN_LABEL, VAL_LABEL])
    print('data loaded')

    return [np.array(ALL_TRAIN_DATA), np.array(ALL_TRAIN_LABEL)], [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [
        np.array(TEST_DATA), np.array(TEST_LABEL)]
def load_SMD(Path='data/SMD'):
    winsize=128
    retio=0.5
    data = np.load(Path + "/SMD_train.npy")
    test_data = np.load(Path + "/SMD_test.npy")
    test_labels = np.load(Path+ "/SMD_test_label.npy")
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    test_data = scaler.transform(test_data)
    srlist = sr(data)
    srlist = scaler.fit_transform(srlist)
    data = data + srlist
    datalist = []
    for head in range(0,data.shape[0] - winsize + 1,winsize):
        datalist.append(data[head:head + winsize])
    ALL_TRAIN_DATA = np.stack(datalist, axis=0).astype(np.float32)
    ALL_TRAIN_LABEL = np.zeros(ALL_TRAIN_DATA.shape[0], ).astype(np.float32)

    VAL_DATA = ALL_TRAIN_DATA[:int(ALL_TRAIN_DATA.shape[0] * retio)].astype(np.float32)
    VAL_LABEL = ALL_TRAIN_LABEL[:int(ALL_TRAIN_DATA.shape[0] * retio)]
    datalist = []
    labellist = []
    for head in range(0,test_data.shape[0] - winsize + 1,winsize):
        datalist.append(test_data[head:head + winsize])
        labellist.append(test_labels[head:head + winsize])
    TEST_DATA=np.stack(datalist, axis=0).astype(np.float32)
    #TEST_LABEL = test_labels[winsize - 1:]
    TEST_LABEL = np.stack(labellist,axis=0).astype(np.float32)

    return [ALL_TRAIN_DATA, ALL_TRAIN_LABEL], [VAL_DATA, VAL_LABEL], [
        TEST_DATA, TEST_LABEL]
    #torch.save(testdict, os.path.join(".\\data\\SMD\\test.pt"))
def load_SMD1(Path='data/SMD/'):
    # file_list = os.listdir(os.path.join('data/SMD/', "train"))
    # i=0
    # for filename in file_list:
    #     if filename.endswith('.pt'):
    #         train =torch.load(os.path.join(Path, "train",filename))
    #         TRAIN_DATA = train['samples'].transpose(1, 2).float()
    #         TRAIN_LABEL = train['labels'].long()
    #         if i==0:
    #             ALL_TRAIN_DATA = TRAIN_DATA
    #             ALL_TRAIN_LABEL = TRAIN_LABEL
    #             i=1
    #         else:
    #             ALL_TRAIN_DATA = torch.cat([ALL_TRAIN_DATA,TRAIN_DATA])
    #             ALL_TRAIN_LABEL = torch.cat([ALL_TRAIN_LABEL,TRAIN_LABEL])
    train = torch.load(Path + '/train/machine-1-1_train.pt')
    val = torch.load(Path + '/eval/machine-1-1_eval.pt')
    test = torch.load(Path + '/test/machine-1-1_test.pt')
    TRAIN_DATA = train['samples'].transpose(1, 2).float()
    TRAIN_LABEL = train['labels'].long()
    VAL_DATA = val['samples'].transpose(1, 2).float()
    VAL_LABEL = val['labels'].long()
    TEST_DATA = test['samples'].transpose(1, 2).float()
    TEST_LABEL = test['labels'].long()

    ALL_TRAIN_DATA = TRAIN_DATA
    ALL_TRAIN_LABEL = TRAIN_LABEL
    print('data loaded')

    return [np.array(ALL_TRAIN_DATA), np.array(ALL_TRAIN_LABEL)], [np.array(VAL_DATA), np.array(VAL_LABEL)], [
        np.array(TEST_DATA), np.array(TEST_LABEL)]
def load_SWAT(Path='data/SWAT/'):
    train = torch.load(Path + '/train.pt')
    val = torch.load(Path + '/eval.pt')
    test = torch.load(Path + '/test.pt')
    TRAIN_DATA = train['samples'].transpose(1, 2).float()
    TRAIN_LABEL = train['labels'].long()
    VAL_DATA = val['samples'].transpose(1, 2).float()
    VAL_LABEL = val['labels'].long()
    TEST_DATA = test['samples'].transpose(1, 2).float()
    TEST_LABEL = test['labels'].long()

    ALL_TRAIN_DATA = TRAIN_DATA
    ALL_TRAIN_LABEL = TRAIN_LABEL
    print('data loaded')

    return [np.array(ALL_TRAIN_DATA), np.array(ALL_TRAIN_LABEL)], [np.array(VAL_DATA), np.array(VAL_LABEL)], [
        np.array(TEST_DATA), np.array(TEST_LABEL)]
def load_MSL(Path='data/MSL/'):
    train = torch.load(Path + '/train.pt')
    val = torch.load(Path + '/eval.pt')
    test = torch.load(Path + '/test.pt')
    TRAIN_DATA = train['samples'].transpose(1, 2).float()
    TRAIN_LABEL = train['labels'].long()
    VAL_DATA = val['samples'].transpose(1, 2).float()
    VAL_LABEL = val['labels'].long()
    TEST_DATA = test['samples'].transpose(1, 2).float()
    TEST_LABEL = test['labels'].long()

    ALL_TRAIN_DATA = TRAIN_DATA
    ALL_TRAIN_LABEL = TRAIN_LABEL
    print('data loaded')

    return [np.array(ALL_TRAIN_DATA), np.array(ALL_TRAIN_LABEL)], [np.array(VAL_DATA), np.array(VAL_LABEL)], [
        np.array(TEST_DATA), np.array(TEST_LABEL)]

def load_PSM(Path='data/PSM'):
    train = torch.load(Path + '/train.pt')
    val = torch.load(Path + '/eval.pt')
    test = torch.load(Path + '/test.pt')
    TRAIN_DATA = train['samples'].transpose(1, 2).float()
    TRAIN_LABEL = train['labels'].long()
    VAL_DATA = val['samples'].transpose(1, 2).float()
    VAL_LABEL = val['labels'].long()
    TEST_DATA = test['samples'].transpose(1, 2).float()
    TEST_LABEL = test['labels'].long()

    ALL_TRAIN_DATA = TRAIN_DATA
    ALL_TRAIN_LABEL = TRAIN_LABEL
    print('data loaded')

    return [np.array(ALL_TRAIN_DATA), np.array(ALL_TRAIN_LABEL)], [np.array(VAL_DATA), np.array(VAL_LABEL)], [
        np.array(TEST_DATA), np.array(TEST_LABEL)]
def load_SMAP(Path='data/SMAP'):
    train = torch.load(Path + '/train.pt')
    val = torch.load(Path + '/eval.pt')
    test = torch.load(Path + '/test.pt')
    TRAIN_DATA = train['samples'].transpose(1, 2).float()
    TRAIN_LABEL = train['labels'].long()
    VAL_DATA = val['samples'].transpose(1, 2).float()
    VAL_LABEL = val['labels'].long()
    TEST_DATA = test['samples'].transpose(1, 2).float()
    TEST_LABEL = test['labels'].long()

    ALL_TRAIN_DATA = TRAIN_DATA
    ALL_TRAIN_LABEL = TRAIN_LABEL
    print('data loaded')

    return [np.array(ALL_TRAIN_DATA), np.array(ALL_TRAIN_LABEL)], [np.array(VAL_DATA), np.array(VAL_LABEL)], [
        np.array(TEST_DATA), np.array(TEST_LABEL)]
def load_WADI(Path='data/WADI'):
    train = torch.load(Path + '/train.pt')
    val = torch.load(Path + '/eval.pt')
    test = torch.load(Path + '/test.pt')
    TRAIN_DATA = train['samples'].transpose(1, 2).float()
    TRAIN_LABEL = train['labels'].long()
    VAL_DATA = val['samples'].transpose(1, 2).float()
    VAL_LABEL = val['labels'].long()
    TEST_DATA = test['samples'].transpose(1, 2).float()
    TEST_LABEL = test['labels'].long()

    ALL_TRAIN_DATA = TRAIN_DATA
    ALL_TRAIN_LABEL = TRAIN_LABEL
    print('data loaded')

    return [np.array(ALL_TRAIN_DATA), np.array(ALL_TRAIN_LABEL)], [np.array(VAL_DATA), np.array(VAL_LABEL)], [
        np.array(TEST_DATA), np.array(TEST_LABEL)]
def load_SYN(Path='data/SYN'):
    train = torch.load(Path + '/4test.pt')
    val = torch.load(Path + '/4test.pt')
    test = torch.load(Path + '/4test.pt')
    TRAIN_DATA = train['samples'].transpose(1, 2).float()
    TRAIN_LABEL = train['labels'].long()
    VAL_DATA = val['samples'].transpose(1, 2).float()
    VAL_LABEL = val['labels'].long()
    TEST_DATA = test['samples'].transpose(1, 2).float()
    TEST_LABEL = test['labels'].long()

    ALL_TRAIN_DATA = TRAIN_DATA
    ALL_TRAIN_LABEL = TRAIN_LABEL
    print('data loaded')

    return [np.array(ALL_TRAIN_DATA), np.array(ALL_TRAIN_LABEL)], [np.array(VAL_DATA), np.array(VAL_LABEL)], [
        np.array(TEST_DATA), np.array(TEST_LABEL)]
def load_GECCO(Path='data/NIPS_TS_GECCO'):
    train = torch.load(Path + '/train.pt')
    val = torch.load(Path + '/eval.pt')
    test = torch.load(Path + '/test.pt')
    TRAIN_DATA = train['samples'].transpose(1, 2).float()
    TRAIN_LABEL = train['labels'].long()
    VAL_DATA = val['samples'].transpose(1, 2).float()
    VAL_LABEL = val['labels'].long()
    TEST_DATA = test['samples'].transpose(1, 2).float()
    TEST_LABEL = test['labels'].long()

    ALL_TRAIN_DATA = TRAIN_DATA
    ALL_TRAIN_LABEL = TRAIN_LABEL
    print('data loaded')

    return [np.array(ALL_TRAIN_DATA), np.array(ALL_TRAIN_LABEL)], [np.array(VAL_DATA), np.array(VAL_LABEL)], [
        np.array(TEST_DATA), np.array(TEST_LABEL)]
def load_Swan(Path='data/NIPS_TS_Swan'):
    train = torch.load(Path + '/train.pt')
    val = torch.load(Path + '/eval.pt')
    test = torch.load(Path + '/test.pt')
    TRAIN_DATA = train['samples'].transpose(1, 2).float()
    TRAIN_LABEL = train['labels'].long()
    VAL_DATA = val['samples'].transpose(1, 2).float()
    VAL_LABEL = val['labels'].long()
    TEST_DATA = test['samples'].transpose(1, 2).float()
    TEST_LABEL = test['labels'].long()

    ALL_TRAIN_DATA = TRAIN_DATA
    ALL_TRAIN_LABEL = TRAIN_LABEL
    print('data loaded')

    return [np.array(ALL_TRAIN_DATA), np.array(ALL_TRAIN_LABEL)], [np.array(VAL_DATA), np.array(VAL_LABEL)], [
        np.array(TEST_DATA), np.array(TEST_LABEL)]
