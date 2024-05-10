import ast
import csv
import os
import sys
from pickle import dump
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing._data import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sr.sr_evalue import sr
output_folder = 'processed_csv'
os.makedirs(output_folder, exist_ok=True)


def load_and_save(category, filename, dataset, dataset_folder):
    os.makedirs(os.path.join(output_folder, filename.split('.')[0]), exist_ok=True)
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float32,
                         delimiter=',')
    # print(dataset, category, filename, temp.shape)
    fea_len = len(temp[0, :])
    header_list = []
    for i in range(fea_len):
        header_list.append("col_%d"%i)
    data = pd.DataFrame(temp, columns=header_list).reset_index()
    data.rename(columns={'index': 'timestamp'}, inplace=True)
    if category == "test":
        temp1 = np.genfromtxt(os.path.join(dataset_folder, "test_label", filename),
                         dtype=np.float32,
                         delimiter=',')
        data1 = pd.DataFrame(temp1, columns=["label"]).reset_index()
        data1.rename(columns={'index': 'timestamp'}, inplace=True)
        data = pd.merge(data, data1, how="left", on='timestamp')

    print(dataset, category, filename, temp.shape)
    data.to_csv(os.path.join(output_folder,  category, dataset + "_" + category + ".csv"), index=False)

def load_data(dataset):
    if dataset == 'SMD':
        dataset_folder = './data/ServerMachineDataset'
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith('.txt'):
                load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
                load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
    elif dataset == 'SMAP' or dataset == 'MSL':
        dataset_folder = 'data'
        with open(os.path.join(dataset_folder, 'labeled_anomalies.csv'), 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        label_folder = os.path.join(dataset_folder, 'test_label')
        os.makedirs(label_folder, exist_ok=True)
        data_info = [row for row in res if row[1] == dataset and row[0] != 'P-2']
        labels = []
        for row in data_info:
            anomalies = ast.literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.int)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = 1
            labels.extend(label)
        labels = np.asarray(labels)
        print(dataset, 'test_label', labels.shape)

        labels = pd.DataFrame(labels, columns=["label"]).reset_index()
        labels.rename(columns={'index': 'timestamp'}, inplace=True)

        def concatenate_and_save(category):
            data = []
            for row in data_info:
                filename = row[0]
                print(os.path.join(dataset_folder, category, filename + '.npy'))
                temp = np.load(os.path.join(dataset_folder, category, filename + '.npy'))
                data.extend(temp)
            data = np.asarray(data)
            print(dataset, category, data.shape)

            fea_len = len(data[0, :])
            header_list = []
            for i in range(fea_len):
                header_list.append("col_%d" % i)
            data = pd.DataFrame(data, columns=header_list).reset_index()
            data.rename(columns={'index': 'timestamp'}, inplace=True)

            if category == "test":
                data = pd.merge(data, labels, how="left", on='timestamp')
            print(dataset, category, filename, temp.shape)
            data.to_csv(os.path.join(output_folder,  dataset + "_" + category + ".csv"),
                        index=False)

        for c in ['train', 'test']:
            concatenate_and_save(c)
def npy2pt(dataset_folder,winsize,retio):
    # data = np.load(dataset_folder + "/SMD_train.npy")
    # test_data = np.load(dataset_folder + "/SMD_test.npy")
    # test_labels = np.load(dataset_folder + "/SMD_test_label.npy")
    # datalist = []
    # for head in range(data.shape[0] - winsize + 1):
    #     #traindata = np.stack([traindata,data[head:head + winsize]])
    #     datalist.append(data[head:head + winsize])
    # #traindata = np.array(traindata).astype(float)
    # traindata = np.stack(datalist, axis=0)
    # traindata = np.transpose(traindata, (0, 2, 1))
    # trainlabel = np.zeros(traindata.shape[0], ).astype(float)
    # evaldata = traindata[:int(traindata.shape[0] * retio)]
    # evallabel = trainlabel[:int(traindata.shape[0] * retio)]
    # datadict = dict()
    # datadict["samples"] = torch.from_numpy(traindata)
    # datadict["labels"] = torch.from_numpy(trainlabel)
    # torch.save(datadict, os.path.join(".\\data\\SMD","train.pt"))
    # evaldict = dict()
    # evaldict["samples"] = torch.from_numpy(evaldata)
    # evaldict["labels"] = torch.from_numpy(evallabel)
    # torch.save(evaldict, os.path.join(".\\data\\SMD\\eval.pt"))
    # datalist = []
    # for head in range(test_data.shape[0] - winsize + 1):
    #     datalist.append(test_data[head:head+winsize])
    # testdata=np.stack(datalist, axis=0)
    # testdata = np.transpose(testdata, (0, 2, 1))
    # label = test_labels[winsize - 1:]
    # testdict = dict()
    # testdict["samples"] = torch.from_numpy(testdata)
    # testdict["labels"] = torch.from_numpy(label)
    # torch.save(testdict, os.path.join(".\\data\\SMD\\test.pt"))
    # winsize=128
    # retio=0.5
    step = 128
    train = np.load(dataset_folder + "/SMD_train.npy")
    train = (train - train.min(axis=0)) / (train.ptp(axis=0) + 1e-4)
    test = np.load(dataset_folder + "/SMD_test.npy")
    #test = (test - test.min(axis=0)) / (test.ptp(axis=0) + 1e-4)
    labels = np.load(dataset_folder+ "/SMD_test_label.npy")
    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    srlist = sr(train)
    srlist = scaler.fit_transform(srlist)
    train = train + srlist
    traindata = []
    t = train.shape[0]
    # for head in range(0, (train.shape[0] - winsize) // step + 1, step):
    for head in range(0,train.shape[0] - winsize + 1,winsize):
        traindata.append(train[head:head + winsize])
    traindata = np.array(traindata).astype(float)
    traindata = np.transpose(traindata,(0,2,1))
    trainlabel = np.zeros(traindata.shape[0], ).astype(float)
    traindict = dict()
    traindict["samples"] = torch.from_numpy(traindata)
    traindict["labels"] = torch.from_numpy(trainlabel)
    torch.save(traindict, os.path.join(".\\data\\SMD\\train.pt"))
    evaldata = traindata[:int(traindata.shape[0] * retio)]
    evallabel = trainlabel[:int(traindata.shape[0] * retio)]
    evaldict = dict()
    evaldict["samples"] = torch.from_numpy(evaldata)
    evaldict["labels"] = torch.from_numpy(evallabel)
    torch.save(evaldict, os.path.join(".\\data\\SMD\\eval.pt"))
    testdata = []
    testlabel = []
    labels = np.asarray(labels)
    #testlabel = np.asarray(labels[winsize-1:])
    # for head in range(0, (test.shape[0] - winsize) // step + 1, step):
    step = winsize
    for head in range(0,test.shape[0] - winsize + 1,winsize):
        testdata.append(test[head:head + winsize])
        testlabel.append(labels[head:head + winsize])
        # testlabel.append(labels[head + winsize // 2])
    testdata = np.transpose(testdata, (0, 2, 1))
    testdata = np.array(testdata).astype(float)
    testlabel = np.array(testlabel).astype(float)
    datadict = dict()
    datadict["samples"] = torch.from_numpy(testdata)
    datadict["labels"] = torch.from_numpy(testlabel)
    torch.save(datadict, os.path.join(".\\data\\SMD\\test.pt"))
def csv2pt(dataset_folder,mode,winsize,retio):
    if mode=="train":
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith('.csv'):
                df = pd.read_csv(os.path.join(dataset_folder,"train",filename),header=None)
                df = df.drop(df.columns[0], axis=1)
                df = df.drop(df.index[0], axis=0)
                data = np.array(df).astype(float)
                traindata = []
                for head in range(data.shape[0]-winsize+1):
                    traindata.append(data[head:head+winsize])
                traindata = np.array(traindata).astype(float)
                traindata = np.transpose(traindata,(0,2,1))
                trainlabel = np.zeros(traindata.shape[0],).astype(float)
                evaldata = traindata[:int(traindata.shape[0]*retio)]
                evallabel= trainlabel[:int(traindata.shape[0]*retio)]
                datadict = dict()
                datadict["samples"] = torch.from_numpy(traindata)
                datadict["labels"] = torch.from_numpy(trainlabel)
                torch.save(datadict,os.path.join(".\\data\\SMD\\train",filename.split(".")[0]+".pt"))
                evaldict = dict()
                evaldict["samples"] = torch.from_numpy(evaldata)
                evaldict["labels"] = torch.from_numpy(evallabel)
                torch.save(evaldict,os.path.join(".\\data\\SMD\\eval",filename.split("train")[0]+"eval.pt"))
                print(eval)
    else:
        file_list = os.listdir(os.path.join(dataset_folder, "test"))
        for filename in file_list:
            if filename.endswith('.csv'):
                df = pd.read_csv(os.path.join(dataset_folder,"test",filename),header=None)
                df = df.drop(df.columns[0], axis=1)
                df = df.drop(df.index[0], axis=0)
                data = np.array(df).astype(float)
                label = data[:,-1]
                data = data[:,:-1]
                testdata=[]
                for head in range(data.shape[0]-winsize+1):
                    testdata.append(data[head:head+winsize])
                testdata = np.array(testdata).astype(float)
                testdata = np.transpose(testdata,(0,2,1))
                label = label[winsize-1:]
                datadict = dict()
                datadict["samples"] = torch.from_numpy(testdata)
                datadict["labels"] = torch.from_numpy(label)
                torch.save(datadict,os.path.join(".\\data\\SMD\\test",filename.split(".")[0]+".pt"))
def swatpre(dataset_folder,winsize,retio):

    train = pd.read_csv(os.path.join(dataset_folder, 'SWaT_Dataset_Normal_v1.csv'))
    test = pd.read_csv(os.path.join(dataset_folder, 'SWaT_Dataset_Attack_v0.csv'), sep=";")
    del train['Normal/Attack']
    del train['Timestamp']
    for i in list(train):
        train[i] = train[i].apply(lambda x: str(x).replace(",", "."))
    train = train.astype(float)
    train = train[train.columns[0:]].values[::1, :]
    train = (train - train.min(0)) / (train.ptp(0) + 1e-4)
    labels = [float(label != 'Normal') for label in test["Normal/Attack"].values]
    del test['Timestamp']
    del test['Normal/Attack']
    for i in list(test):
        test[i] = test[i].apply(lambda x: str(x).replace(",", "."))
    test = test.astype(float)
    test = test[test.columns[0:]].values[::1, :]
    test = (test - test.min(0)) / (test.ptp(0) + 1e-4)
    labels = labels[::1]
    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    # srlist = sr(train)
    # srlist = scaler.fit_transform(srlist)
    # train = train + srlist
    traindata = []
    for head in range(0,train.shape[0] - winsize + 1,winsize):
        traindata.append(train[head:head + winsize])
    traindata = np.array(traindata).astype(float)
    traindata = np.transpose(traindata,(0,2,1))


    trainlabel = np.zeros(traindata.shape[0], ).astype(float)
    traindict = dict()
    traindict["samples"] = torch.from_numpy(traindata)
    traindict["labels"] = torch.from_numpy(trainlabel)
    torch.save(traindict, os.path.join(".\\data\\SWAT\\train.pt"))
    evaldata = traindata[:int(traindata.shape[0] * retio)]
    evallabel = trainlabel[:int(traindata.shape[0] * retio)]
    evaldict = dict()
    evaldict["samples"] = torch.from_numpy(evaldata)
    evaldict["labels"] = torch.from_numpy(evallabel)
    torch.save(evaldict, os.path.join(".\\data\\SWAT\\eval.pt"))
    testdata = []
    testlabel = []
    labels = np.asarray(labels)
    #testlabel = np.asarray(labels[winsize-1:])
    for head in range(0,test.shape[0] - winsize + 1,winsize):
        testdata.append(test[head:head + winsize])
        testlabel.append(labels[head:head + winsize])
        # testlabel.append(labels[head + winsize//2])
    testdata = np.transpose(testdata, (0, 2, 1))
    testdata = np.array(testdata).astype(float)
    testlabel = np.array(testlabel).astype(float)
    datadict = dict()
    datadict["samples"] = torch.from_numpy(testdata)
    datadict["labels"] = torch.from_numpy(testlabel)
    torch.save(datadict, os.path.join(".\\data\\SWAT\\test.pt"))
# def swatpre1(dataset_folder,winsize,retio):
#
#     train = pd.read_csv(os.path.join(dataset_folder, 'SWaT_Dataset_Normal_v1.csv'))
#     test = pd.read_csv(os.path.join(dataset_folder, 'SWaT_Dataset_Attack_v0.csv'), sep=";")
#     del train['Normal/Attack']
#     del train['Timestamp']
#     for i in list(train):
#         train[i] = train[i].apply(lambda x: str(x).replace(",", "."))
#     train = train.astype(float)
#     train = train[train.columns[0:]].values[::10, :]
#     train = (train - train.min(0)) / (train.ptp(0) + 1e-4)
#     labels = [float(label != 'Normal') for label in test["Normal/Attack"].values]
#     del test['Timestamp']
#     del test['Normal/Attack']
#     for i in list(test):
#         test[i] = test[i].apply(lambda x: str(x).replace(",", "."))
#     test = test.astype(float)
#     test = test[test.columns[0:]].values[::10, :]
#     test = (test - test.min(0)) / (test.ptp(0) + 1e-4)
#     labels = labels[::10]
#
#     traindata = []
#     for head in range(train.shape[0] - winsize + 1):
#         traindata.append(train[head:head + winsize])
#     traindata = np.array(traindata).astype(float)
#     traindata = np.transpose(traindata,(0,2,1))
#     trainlabel = np.zeros(traindata.shape[0], ).astype(float)
#     traindict = dict()
#     traindict["samples"] = torch.from_numpy(traindata)
#     traindict["labels"] = torch.from_numpy(trainlabel)
#     torch.save(traindict, os.path.join(".\\data\\SWAT\\train.pt"))
#     evaldata = traindata[:int(traindata.shape[0] * retio)]
#     evallabel = trainlabel[:int(traindata.shape[0] * retio)]
#     evaldict = dict()
#     evaldict["samples"] = torch.from_numpy(evaldata)
#     evaldict["labels"] = torch.from_numpy(evallabel)
#     torch.save(evaldict, os.path.join(".\\data\\SWAT\\eval.pt"))
#     testdata = []
#     testlabel = np.asarray(labels[winsize-1:])
#     for head in range(test.shape[0] - winsize + 1):
#         testdata.append(test[head:head + winsize])
#     testdata = np.transpose(testdata, (0, 2, 1))
#     testdata = np.array(testdata).astype(float)
#     datadict = dict()
#     datadict["samples"] = torch.from_numpy(testdata)
#     datadict["labels"] = torch.from_numpy(testlabel)
#     torch.save(datadict, os.path.join(".\\data\\SWAT\\test.pt"))
def wadipre(dataset_folder,winsize,retio):
    dataset_folder = 'data/WADI'
    ls = pd.read_csv(os.path.join(dataset_folder, 'WADI_attacklabels.csv'))
    train = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days.csv'), skiprows=1000, nrows=2e5)
    test = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdata.csv'))
    train.dropna(how='all', inplace=True);
    test.dropna(how='all', inplace=True)
    train.fillna(0, inplace=True);
    test.fillna(0, inplace=True)
    test['Time'] = test['Time'].astype(str)
    test['Time'] = pd.to_datetime(test['Date'] + ' ' + test['Time'])
    labels = test.copy(deep=True)
    for i in test.columns.tolist()[3:]: labels[i] = 0
    for i in ['Start Time', 'End Time']:
        ls[i] = ls[i].astype(str)
        ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i])
    for index, row in ls.iterrows():
        to_match = row['Affected'].split(', ')
        matched = []
        for i in test.columns.tolist()[3:]:
            for tm in to_match:
                if tm in i:
                    matched.append(i);
                    break
        st, et = str(row['Start Time']), str(row['End Time'])
        labels.loc[(labels['Time'] >= st) & (labels['Time'] <= et), matched] = 1
    train, test, labels = convertNumpy(train), convertNumpy(test), convertNumpy(labels)
    labels = np.sum(labels,axis=1)
    labels = [1 if labels[i]>0 else 0 for i in range(labels.shape[0])]
    labels = np.asarray(labels)
    print(train.shape, test.shape, labels.shape)
    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    # srlist = sr(train)
    # srlist = scaler.fit_transform(srlist)
    # train = train + srlist
    traindata = []
    for head in range(0,train.shape[0] - winsize + 1,winsize):
        traindata.append(train[head:head + winsize])
    traindata = np.array(traindata).astype(float)
    traindata = np.transpose(traindata,(0,2,1))
    trainlabel = np.zeros(traindata.shape[0], ).astype(float)
    traindict = dict()
    traindict["samples"] = torch.from_numpy(traindata)
    traindict["labels"] = torch.from_numpy(trainlabel)
    torch.save(traindict, os.path.join(".\\data\\WADI\\train.pt"))
    evaldata = traindata[:int(traindata.shape[0] * retio)]
    evallabel = trainlabel[:int(traindata.shape[0] * retio)]
    evaldict = dict()
    evaldict["samples"] = torch.from_numpy(evaldata)
    evaldict["labels"] = torch.from_numpy(evallabel)
    torch.save(evaldict, os.path.join(".\\data\\WADI\\eval.pt"))
    testdata = []
    testlabel = []
    labels = np.asarray(labels)
    #np.savetxt("very.csv", labels, delimiter=",", newline=",")
    for head in range(0,test.shape[0] - winsize + 1,winsize):
        testdata.append(test[head:head + winsize])
        testlabel.append(labels[head:head + winsize])
        # testlabel.append(labels[head + winsize//2])
    testdata = np.transpose(testdata, (0, 2, 1))
    testdata = np.array(testdata).astype(float)
    testlabel = np.array(testlabel).astype(float)
    datadict = dict()
    datadict["samples"] = torch.from_numpy(testdata)
    datadict["labels"] = torch.from_numpy(testlabel)
    torch.save(datadict, os.path.join(".\\data\\WADI\\test.pt"))

def mslpre(dataset_folder,winsize,retio):
    # winsize=128
    # retio=0.5
    train = np.load(dataset_folder + "/MSL_train.npy")
    train = (train - train.min(axis=0)) / (train.ptp(axis=0) + 1e-4)
    test = np.load(dataset_folder + "/MSL_test.npy")
    #test = (test - test.min(axis=0)) / (test.ptp(axis=0) + 1e-4)
    labels = np.load(dataset_folder+ "/MSL_test_label.npy")
    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    # srlist = sr(train)
    # srlist = scaler.fit_transform(srlist)
    # train = train + srlist
    traindata = []
    # for head in range(0,train.shape[0] - winsize + 1,winsize):
    #     traindata.append(train[head:head + winsize])
    step = 128
    for head in range(0,(train.shape[0] - winsize) + 1,step):
        traindata.append(train[head:head + winsize])
    traindata = np.array(traindata).astype(float)
    traindata = np.transpose(traindata,(0,2,1))
    trainlabel = np.zeros(traindata.shape[0], ).astype(float)
    traindict = dict()
    traindict["samples"] = torch.from_numpy(traindata)
    traindict["labels"] = torch.from_numpy(trainlabel)
    torch.save(traindict, os.path.join(".\\data\\MSL\\train.pt"))
    evaldata = traindata[:int(traindata.shape[0] * retio)]
    evallabel = trainlabel[:int(traindata.shape[0] * retio)]
    evaldict = dict()
    evaldict["samples"] = torch.from_numpy(evaldata)
    evaldict["labels"] = torch.from_numpy(evallabel)
    torch.save(evaldict, os.path.join(".\\data\\MSL\\eval.pt"))
    testdata = []
    testlabel = []
    labels = np.asarray(labels)
    #testlabel = np.asarray(labels[winsize-1:])
    # for head in range(0,test.shape[0] - winsize + 1,winsize):
    step = winsize
    for head in range(0, (test.shape[0] - winsize)+ 1, step):
        testdata.append(test[head:head + winsize])
        testlabel.append(labels[head:head + winsize])
        # testlabel.append(labels[head + winsize // 2])
    testdata = np.transpose(testdata, (0, 2, 1))
    testdata = np.array(testdata).astype(float)
    testlabel = np.array(testlabel).astype(float)
    datadict = dict()
    datadict["samples"] = torch.from_numpy(testdata)
    datadict["labels"] = torch.from_numpy(testlabel)
    torch.save(datadict, os.path.join(".\\data\\MSL\\test.pt"))
def smdpre(dataset_folder,winsize,retio):
    file_list = os.listdir(os.path.join(dataset_folder, "train"))
    for filename in file_list:
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(dataset_folder, "train", filename), header=None)
            df = df.drop(df.columns[0], axis=1)
            df = df.drop(df.index[0], axis=0)
            train = np.array(df).astype(float)
            df = pd.read_csv(os.path.join(dataset_folder, "test", filename.split("train")[0]+"test.csv"), header=None)
            df = df.drop(df.columns[0], axis=1)
            df = df.drop(df.index[0], axis=0)
            data = np.array(df).astype(float)
            label = data[:, -1]
            test = data[:, :-1]
            #标准化过程
            scaler = StandardScaler()
            scaler.fit(train)
            train = scaler.transform(train)
            test = scaler.transform(test)
            srlist = sr(train)
            srlist = scaler.fit_transform(srlist)
            train = train + srlist
            traindata = []
            for head in range(0, train.shape[0] - winsize + 1, winsize):
                traindata.append(train[head:head + winsize])
            traindata = np.array(traindata).astype(float)
            traindata = np.transpose(traindata, (0, 2, 1))
            trainlabel = np.zeros(traindata.shape[0], ).astype(float)
            traindict = dict()
            traindict["samples"] = torch.from_numpy(traindata)
            traindict["labels"] = torch.from_numpy(trainlabel)
            torch.save(traindict, os.path.join(".\\data\\SMD\\train", filename.split(".")[0] + ".pt"))
            evaldata = traindata[:int(traindata.shape[0] * retio)]
            evallabel = trainlabel[:int(traindata.shape[0] * retio)]
            evaldict = dict()
            evaldict["samples"] = torch.from_numpy(evaldata)
            evaldict["labels"] = torch.from_numpy(evallabel)
            torch.save(evaldict, os.path.join(".\\data\\SMD\\eval", filename.split("train")[0] + "eval.pt"))
            testdata = []
            testlabel = []
            for head in range(0, test.shape[0] - winsize + 1, winsize):
                testdata.append(test[head:head + winsize])
                testlabel.append(label[head:head + winsize])
            testdata = np.array(testdata).astype(float)
            testlabel = np.array(testlabel).astype(float)
            testdata = np.transpose(testdata, (0, 2, 1))
            testdict = dict()
            testdict["samples"] = torch.from_numpy(testdata)
            testdict["labels"] = torch.from_numpy(testlabel)
            torch.save(testdict, os.path.join(".\\data\\SMD\\test", filename.split("train")[0] + "test.pt"))
def convertNumpy(df):
	x = df[df.columns[3:]].values[::1, :]
	return (x - x.min(0)) / (x.ptp(0) + 1e-4)
def psmpre(dataset_folder,winsize,retio):
    # winsize = 32
    # retio = 0.5
    step = 32
    train = pd.read_csv(dataset_folder + '\\train.csv')
    train = train.values[:, 1:]
    train = np.nan_to_num(train)
    train = (train - train.min(axis=0)) / (train.ptp(axis=0) + 1e-4)
    test = pd.read_csv(dataset_folder + '\\test.csv')
    test = test.values[:, 1:]
    test = np.nan_to_num(test)
    #test = (test - test.min(axis=0)) / (test.ptp(axis=0) + 1e-4)
    labels = pd.read_csv(dataset_folder + '\\test_label.csv').values[:, 1:]
    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    # srlist = sr(train)
    # srlist = scaler.fit_transform(srlist)
    # train = train + srlist
    traindata = []
    # for head in range(0, (train.shape[0] - winsize) // step + 1, step):
    for head in range(0,train.shape[0] - winsize + 1,32):
        traindata.append(train[head:head + winsize])
    traindata = np.array(traindata).astype(float)
    traindata = np.transpose(traindata,(0,2,1))
    trainlabel = np.zeros(traindata.shape[0], ).astype(float)
    traindict = dict()
    traindict["samples"] = torch.from_numpy(traindata)
    traindict["labels"] = torch.from_numpy(trainlabel)
    torch.save(traindict, os.path.join(".\\data\\PSM\\train.pt"))
    evaldata = traindata[:int(traindata.shape[0] * retio)]
    evallabel = trainlabel[:int(traindata.shape[0] * retio)]
    evaldict = dict()
    evaldict["samples"] = torch.from_numpy(evaldata)
    evaldict["labels"] = torch.from_numpy(evallabel)
    torch.save(evaldict, os.path.join(".\\data\\PSM\\eval.pt"))
    testdata = []
    testlabel = []
    labels = np.asarray(labels)
    #testlabel = np.asarray(labels[winsize-1:])
    step = winsize
    # for head in range(0, (test.shape[0] - winsize) // step + 1, step):
    for head in range(0,test.shape[0] - winsize + 1,128):
        testdata.append(test[head:head + winsize])
        testlabel.append(labels[head:head + winsize])
        # testlabel.append(labels[head + winsize])
    testdata = np.transpose(testdata, (0, 2, 1))
    testdata = np.array(testdata).astype(float)
    testlabel = np.array(testlabel).astype(float)
    datadict = dict()
    datadict["samples"] = torch.from_numpy(testdata)
    datadict["labels"] = torch.from_numpy(testlabel)
    torch.save(datadict, os.path.join(".\\data\\PSM\\test.pt"))
def smappre(dataset_folder,winsize,retio):
    # winsize=128
    # retio=0.5
    step = 128
    train = np.load(dataset_folder + "/SMAP_train.npy")
    train = (train - train.min(axis=0)) / (train.ptp(axis=0) + 1e-4)
    test = np.load(dataset_folder + "/SMAP_test.npy")
    #test = (test - test.min(axis=0)) / (test.ptp(axis=0) + 1e-4)
    labels = np.load(dataset_folder+ "/SMAP_test_label.npy")
    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    # srlist = sr(train)
    # srlist = scaler.fit_transform(srlist)
    # train = train + srlist
    traindata = []
    t = train.shape[0]
    # for head in range(0, (train.shape[0] - winsize) // step + 1, step):
    for head in range(0,train.shape[0] - winsize + 1,winsize):
        traindata.append(train[head:head + winsize])
    traindata = np.array(traindata).astype(float)
    traindata = np.transpose(traindata,(0,2,1))
    trainlabel = np.zeros(traindata.shape[0], ).astype(float)
    traindict = dict()
    traindict["samples"] = torch.from_numpy(traindata)
    traindict["labels"] = torch.from_numpy(trainlabel)
    torch.save(traindict, os.path.join(".\\data\\SMAP\\train.pt"))
    evaldata = traindata[:int(traindata.shape[0] * retio)]
    evallabel = trainlabel[:int(traindata.shape[0] * retio)]
    evaldict = dict()
    evaldict["samples"] = torch.from_numpy(evaldata)
    evaldict["labels"] = torch.from_numpy(evallabel)
    torch.save(evaldict, os.path.join(".\\data\\SMAP\\eval.pt"))
    testdata = []
    testlabel = []
    labels = np.asarray(labels)
    #testlabel = np.asarray(labels[winsize-1:])
    # for head in range(0, (test.shape[0] - winsize) // step + 1, step):
    step = winsize
    for head in range(0,test.shape[0] - winsize + 1,winsize):
        testdata.append(test[head:head + winsize])
        testlabel.append(labels[head:head + winsize])
        # testlabel.append(labels[head + winsize // 2])
    testdata = np.transpose(testdata, (0, 2, 1))
    testdata = np.array(testdata).astype(float)
    testlabel = np.array(testlabel).astype(float)
    datadict = dict()
    datadict["samples"] = torch.from_numpy(testdata)
    datadict["labels"] = torch.from_numpy(testlabel)
    torch.save(datadict, os.path.join(".\\data\\SMAP\\test.pt"))
def GECCOpre(dataset_folder,winsize,retio):
    # winsize=128
    # retio=0.5
    train = np.load(dataset_folder + "/NIPS_TS_Water_train.npy")
    train = (train - train.min(axis=0)) / (train.ptp(axis=0) + 1e-4)
    test = np.load(dataset_folder + "/NIPS_TS_Water_test.npy")
    #test = (test - test.min(axis=0)) / (test.ptp(axis=0) + 1e-4)
    labels = np.load(dataset_folder+ "/NIPS_TS_Water_test_label.npy")
    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    srlist = sr(train)
    srlist = scaler.fit_transform(srlist)
    train = train + srlist
    traindata = []
    step = 128
    for head in range(0,(train.shape[0] - winsize) + 1,step):
        traindata.append(train[head:head + winsize])
    traindata = np.array(traindata).astype(float)
    traindata = np.transpose(traindata,(0,2,1))
    trainlabel = np.zeros(traindata.shape[0], ).astype(float)
    traindict = dict()
    traindict["samples"] = torch.from_numpy(traindata)
    traindict["labels"] = torch.from_numpy(trainlabel)
    torch.save(traindict, os.path.join(".\\data\\NIPS_TS_GECCO\\train.pt"))
    evaldata = traindata[:int(traindata.shape[0] * retio)]
    evallabel = trainlabel[:int(traindata.shape[0] * retio)]
    evaldict = dict()
    evaldict["samples"] = torch.from_numpy(evaldata)
    evaldict["labels"] = torch.from_numpy(evallabel)
    torch.save(evaldict, os.path.join(".\\data\\NIPS_TS_GECCO\\eval.pt"))
    testdata = []
    testlabel = []
    labels = np.asarray(labels)
    step = winsize
    for head in range(0, (test.shape[0] - winsize)+ 1, step):
        testdata.append(test[head:head + winsize])
        testlabel.append(labels[head:head + winsize])
    testdata = np.transpose(testdata, (0, 2, 1))
    testdata = np.array(testdata).astype(float)
    testlabel = np.array(testlabel).astype(float)
    datadict = dict()
    datadict["samples"] = torch.from_numpy(testdata)
    datadict["labels"] = torch.from_numpy(testlabel)
    torch.save(datadict, os.path.join(".\\data\\NIPS_TS_GECCO\\test.pt"))
def Swanpre(dataset_folder,winsize,retio):
    # winsize=128
    # retio=0.5
    train = np.load(dataset_folder + "/NIPS_TS_Swan_train.npy")
    train = (train - train.min(axis=0)) / (train.ptp(axis=0) + 1e-4)
    test = np.load(dataset_folder + "/NIPS_TS_Swan_test.npy")
    #test = (test - test.min(axis=0)) / (test.ptp(axis=0) + 1e-4)
    labels = np.load(dataset_folder+ "/NIPS_TS_Swan_test_label.npy")
    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    srlist = sr(train)
    srlist = scaler.fit_transform(srlist)
    train = train + srlist
    traindata = []
    step = 128
    for head in range(0,(train.shape[0] - winsize) + 1,step):
        traindata.append(train[head:head + winsize])
    traindata = np.array(traindata).astype(float)
    traindata = np.transpose(traindata,(0,2,1))
    trainlabel = np.zeros(traindata.shape[0], ).astype(float)
    traindict = dict()
    traindict["samples"] = torch.from_numpy(traindata)
    traindict["labels"] = torch.from_numpy(trainlabel)
    torch.save(traindict, os.path.join(".\\data\\NIPS_TS_Swan\\train.pt"))
    evaldata = traindata[:int(traindata.shape[0] * retio)]
    evallabel = trainlabel[:int(traindata.shape[0] * retio)]
    evaldict = dict()
    evaldict["samples"] = torch.from_numpy(evaldata)
    evaldict["labels"] = torch.from_numpy(evallabel)
    torch.save(evaldict, os.path.join(".\\data\\NIPS_TS_Swan\\eval.pt"))
    testdata = []
    testlabel = []
    labels = np.asarray(labels)
    step = winsize
    for head in range(0, (test.shape[0] - winsize)+ 1, step):
        testdata.append(test[head:head + winsize])
        testlabel.append(labels[head:head + winsize])
    testdata = np.transpose(testdata, (0, 2, 1))
    testdata = np.array(testdata).astype(float)
    testlabel = np.array(testlabel).astype(float)
    datadict = dict()
    datadict["samples"] = torch.from_numpy(testdata)
    datadict["labels"] = torch.from_numpy(testlabel)
    torch.save(datadict, os.path.join(".\\data\\NIPS_TS_Swan\\test.pt"))
def syn(dataset_folder,winsize,retio):
    file_list = os.listdir(os.path.join(dataset_folder))
    for filename in file_list:
        if filename.endswith('4test.csv'):
            # df = pd.read_csv(os.path.join(dataset_folder, filename), header=None)
            # df = df.drop(df.columns[-1], axis=1)
            # df = df.drop(df.index[0], axis=0)
            # train = np.array(df).astype(float)
            df = pd.read_csv(os.path.join(dataset_folder, filename), header=None)
            df = df.drop(df.index[0], axis=0)
            data = np.array(df).astype(float)
            label = data[:, -1]
            data = data[:, :-1]
            train = np.repeat(data, 1, axis=1)

            scaler = StandardScaler()
            scaler.fit(train)
            train = scaler.transform(train)
            # srlist = sr(train)
            # srlist = scaler.fit_transform(srlist)
            # train = train + srlist
            traindata = []
            for head in range(0, train.shape[0] - winsize + 1, winsize):
                traindata.append(train[head:head + winsize])
            traindata = np.array(traindata).astype(float)
            traindata = np.transpose(traindata, (0, 2, 1))
            trainlabel = np.zeros(traindata.shape[0], ).astype(float)
            evaldata = traindata[:int(traindata.shape[0] * retio)]
            evallabel = trainlabel[:int(traindata.shape[0] * retio)]
            datadict = dict()
            datadict["samples"] = torch.from_numpy(traindata)
            datadict["labels"] = torch.from_numpy(trainlabel)
            # torch.save(datadict, os.path.join(dataset_folder,"train.pt"))
            torch.save(datadict, os.path.join(".\\data\\SYN", "4train.pt"))
            evaldict = dict()
            evaldict["samples"] = torch.from_numpy(evaldata)
            evaldict["labels"] = torch.from_numpy(evallabel)
            torch.save(evaldict, os.path.join(dataset_folder,"eval.pt"))
        if filename.endswith('4test.csv'):
            df = pd.read_csv(os.path.join(dataset_folder,filename),header=None)
            df = df.drop(df.index[0], axis=0)
            data = np.array(df).astype(float)
            label = data[:,-1]
            data = data[:,:-1]
            test = np.repeat(data, 1, axis=1)
            scaler = StandardScaler()
            scaler.fit(test)
            test = scaler.transform(test)
            testdata=[]
            testlabel = []
            for head in range(0, test.shape[0] - winsize + 1, winsize):
                testdata.append(test[head:head + winsize])
                testlabel.append(label[head:head + winsize])
            testdata = np.transpose(testdata, (0, 2, 1))
            testdata = np.array(testdata).astype(float)
            testlabel = np.array(testlabel).astype(float)
            datadict = dict()
            datadict["samples"] = torch.from_numpy(testdata)
            datadict["labels"] = torch.from_numpy(testlabel)
            torch.save(datadict, os.path.join(".\\data\\SYN",filename.split(".")[0]+".pt"))
if __name__ == '__main__':
    #datasets = ['SMD', 'SMAP', 'MSL']
    #datasets = ['SMD']
    #load_data('SMD')
    #csv2pt(".\\processed_csv","train",winsize=128,retio=0.2)
    #csv2pt(".\\processed_csv", "test", winsize=128, retio=0.2)

    # npy2pt('.\\data\\SMD',winsize=128,retio=1)#SMD anomaly trans

    smdpre(".\\data\\SMD",winsize=128,retio=0.8)
    # # #
    # mslpre(".\\data\\MSL", winsize=128, retio=1)
    # psmpre(".\\data\\PSM", winsize=128, retio=1)
    # smappre(".\\data\\SMAP", winsize=128, retio=1)
    # swatpre(".\\data\\SWAT", winsize=128, retio=0.8)  # swat的
    # wadipre(".\\data\\WADI", winsize=128, retio=0.8)
    # syn(".\\data\\SYN", winsize=128, retio=1)


    # GECCOpre(".\\data\\NIPS_TS_GECCO", winsize=128, retio=1)
    # Swanpre(".\\data\\NIPS_TS_Swan", winsize=128, retio=1)