import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import lr_scheduler
from tqdm import tqdm

from loss import CE, Align, Reconstruct
from torch.optim.lr_scheduler import LambdaLR
from classification import fit_lr, get_rep_with_label
from RevIN import RevIN
import os
from sr.sr_evalue import sr
import matplotlib.pyplot as plt
from sklearn.preprocessing._data import StandardScaler
from sklearn.preprocessing import MinMaxScaler


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, model, path):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score > self.best_score + self.delta :
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        # torch.save(model.state_dict(), self.save_path + '/pretrain_model.pkl')
        self.val_loss_min = val_loss

class Trainer():
    def __init__(self, args, model, train_loader, train_linear_loader, test_loader, verbose=False):
        self.args = args
        self.verbose = verbose
        self.device = args.device
        # self.print_process(self.device)
        self.model = model.to(torch.device(self.device))
        # self.model = model.cuda()
        # print('model cuda')

        self.train_loader = train_loader
        self.train_linear_loader = train_linear_loader
        self.test_loader = test_loader
        # self.lr_decay = args.lr_decay_rate
        # self.lr_decay_steps = args.lr_decay_steps

        # self.cr = CE(self.model)
        # self.alpha = args.alpha
        # self.beta = args.beta

        # self.test_cr = torch.nn.CrossEntropyLoss()
        # self.num_epoch = args.num_epoch
        self.num_epoch_pretrain = args.num_epoch_pretrain
        # self.eval_per_steps = args.eval_per_steps
        self.save_path = args.save_path
        # if self.num_epoch:
        #     self.result_file = open(self.save_path + '/result.txt', 'w')
        #     self.result_file.close()

        # self.step = 0
        # self.best_metric = -1e9
        # self.metric = 'acc'
        self.dataset = args.dataset
    def pretrain(self):

        print('training')
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        # eval_acc = 0
        align = Align()
        # reconstruct = Reconstruct()
        # self.model.copy_weight()
        # if self.num_epoch_pretrain:
        #     result_file = open(self.save_path + '/pretrain_result.txt', 'w')
        #     result_file.close()
        #     result_file = open(self.save_path + '/linear_result.txt', 'w')
        #     result_file.close()
        early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name=self.dataset)
        for epoch in range(self.num_epoch_pretrain):
            self.model.train()
            tqdm_dataloader = tqdm(self.train_loader)
            loss_sum = 0
            loss_mse = 0
            # loss_ce = 0
            # hits_sum = 0
            # NDCG_sum = 0
            for idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                optimizer.zero_grad()
                [ori, rep]= self.model.pretrain_forward(batch[0])
                align_loss = align.compute(ori, rep)
                loss_mse += align_loss.item()
                loss = align_loss
                loss.backward()
                optimizer.step()
                # scheduler.step()
                # print('lr:',optimizer.state_dict()['param_groups'][0]['lr'])
                # self.model.momentum_update()
                loss_sum += loss.item()
            print('pretrain epoch{0}, loss{1}'.format(epoch + 1, loss_sum / (idx + 1)))
            result_file = open(self.save_path + '/train_result.txt', 'a+')
            print('pretrain epoch{0}, loss{1}'.format(epoch + 1, loss_sum / (idx + 1)),file=result_file)
            result_file.close()
            # torch.save(self.model.state_dict(), os.path.join(self.save_path, str(self.dataset) + '_checkpoint'+str(epoch)+'.pth'))
            #
            vali_loss = self.vali(self.train_linear_loader)
            early_stopping(vali_loss, self.model, self.save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # if (epoch + 1) % 1 == 0:
            #     print('epoch{}:',epoch)
            #     self.test()
            #     self.model.eval()
            #     train_rep, train_label = get_rep_with_label(self.model, self.train_linear_loader)
            #     test_rep, test_label = get_rep_with_label(self.model, self.test_loader)
            #     clf = fit_lr(train_rep, train_label)
            #     acc = clf.score(test_rep, test_label)
            #     print(acc)
            #     result_file = open(self.save_path + '/linear_result.txt', 'a+')
            #     print('epoch{0}, acc{1}'.format(epoch, acc), file=result_file)
            #     result_file.close()
            #     if acc > eval_acc:
            #         eval_acc = acc
            #         torch.save(self.model.state_dict(), self.save_path + '/pretrain_model.pkl')
    def vali(self, vali_loader):
        self.model.eval()
        tqdm_data_loader = tqdm(vali_loader)
        mseloss = nn.L1Loss()
        losslist = []
        labellist = []
        retlist = []
        loss_sum = 0
        with torch.no_grad():
            for idx, batch in enumerate(tqdm_data_loader):
                batch = [x.to(self.device) for x in batch]
                seqs, label = batch
                labellist.append(label.cpu())
                revin = RevIN()
                seqs = revin(seqs, mode='norm')
                ori,ret = self.model(seqs)
                # ret = revin(ret, mode='denorm')
                retlist.append(ret.cpu())  ###############################################
                loss = mseloss(ret, seqs)
                loss_sum += loss.cpu().item()

        return loss_sum
    def test(self):
        if self.args.load_pretrained_model:
            self.model.load_state_dict(
                torch.load(
                    os.path.join(str(self.save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50
        tqdm_data_loader = tqdm(self.test_loader)
        print("======================TEST MODE======================")
        mseloss = nn.L1Loss(reduce=False)
        losslist = []
        labellist = []
        retlist = []
        seqlist =[]
        trainretlist = []
        trainlosslist = []
        with torch.no_grad():
            # for idx, batch in enumerate(self.train_loader):
            #     batch = [x.to(self.device) for x in batch]
            #     seqs, label = batch
            #     revin = RevIN()
            #     seqs = revin(seqs, mode='norm')
            #     ori, ret = self.model(seqs)
            #     # ret = revin(ret,mode = 'denorm')
            #     trainretlist.append(ret.cpu())  ###############################################
            #     loss = mseloss(ret, seqs)
            #     loss_each = torch.mean(loss, -1)
            #     trainlosslist.append(loss_each.cpu())
            for idx, batch in enumerate(tqdm_data_loader):
                batch = [x.to(self.device) for x in batch]
                seqs, label = batch
                seqlist.append(seqs.cpu())
                labellist.append(label.cpu())
                revin = RevIN()
                seqs = revin(seqs,mode='norm')
                ori,ret = self.model(seqs)
                # ret = revin(ret,mode = 'denorm')
                retlist.append(ret.cpu())  ###############################################
                loss = mseloss(ret,seqs)
                loss_each = torch.mean(loss, -1)
                losslist.append(loss_each.cpu())
        losslist = np.concatenate(losslist, axis=0).reshape(-1)
        # trainlosslist = np.concatenate(trainlosslist, axis=0).reshape(-1)
        t = retlist[0].size()[-1]
        # t1 = trainretlist[0].size()[-1]
        seqlist = np.concatenate(seqlist).reshape(-1, t)
        seqlist = seqlist[:,0]
        losslist1 = np.expand_dims(losslist, axis=1)
        temp11 = np.repeat(losslist1, 2,axis=1)
        retlist = np.concatenate(retlist).reshape(-1, t)
        # trainretlist = np.concatenate(trainretlist).reshape(-1, t1)

        srlist = sr(retlist)#原来是retlist
        srlist = np.mean(srlist,axis=1)      # -》》》》》》》》》》》》》》》》》》srlist
        srlist = (srlist - np.mean(srlist)) / np.std(srlist)

        # trainsrlist = sr(trainretlist)  # 原来是retlist
        # trainsrlist = np.mean(trainsrlist, axis=1)
        # trainsrlist = (trainsrlist - np.mean(trainsrlist)) / np.std(trainsrlist)
        temploss = losslist
        # trainlosslist = trainlosslist + trainsrlist
        losslist = losslist+srlist      #-》》》》》》》》》》》》》》》》》》》》sr
        # combined_loss = np.concatenate([trainlosslist, losslist], axis=0)
        name = 'anomaly1'
        labellist = np.concatenate(labellist, axis=0).reshape(-1).astype(int).tolist()
        best_eval = {'f1': 0}
        best_rate = 0
        best_print_thresh = 0
        best_print_predlabel = []
        rate = self.args.anomaly_ratio
        from metrics.combine_all_scores import combine_all_evaluation_scores
        best_VUS_ROC = 0
        if rate == 100:
            for rate in np.arange(0.9, 1.3, 0.1):
                best_eval,best_rate,best_print_thresh,best_print_predlabel = self.point_ad(losslist,
                    rate, labellist, best_eval,best_rate,best_print_thresh,best_print_predlabel)
                # thresh = np.percentile(losslist, 100 - rate)
                # pred = (losslist > thresh).astype(int)
                # scores = combine_all_evaluation_scores(np.array(pred), np.array(labellist), np.array(losslist))
                # result_file1 = open(self.save_path + '/vus_result_final.txt', 'a+')
                # for key, value in scores.items():
                #     print("rate:", rate)
                #     print(key, ' : ', value)
                #     print("rate:", rate, file=result_file1)
                #     print(key, ' : ', value, file=result_file1)
                # result_file1.close()


        else:
            best_eval, best_rate, best_print_thresh, best_print_predlabel = self.point_ad(losslist, rate, labellist,
                    best_eval,best_rate,best_print_thresh,best_print_predlabel)
            # for i in range(1):
            #     thresh = np.percentile(losslist, 100 - rate)
            #     pred = (losslist > thresh).astype(int)
            #     scores = combine_all_evaluation_scores(np.array(pred), np.array(labellist), np.array(losslist))
            #     result_file1 = open(self.save_path + '/vus_result3.txt', 'a+')
            #     for key, value in scores.items():
            #         print("rate:", rate)
            #         print(key, ' : ', value)
            #         print("rate:", rate, file=result_file1)
            #         print(key, ' : ', value, file=result_file1)
            #     result_file1.close()
        print('anomaly_ratio', best_rate)
        print('thresh', best_eval['f1'])
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                best_eval['acc'], best_eval['precision'],
                best_eval['recall'] , best_eval['f1']))

        result_file = open(self.save_path + '/test_result.txt', 'a+')
        print('anomaly_ratio', best_rate,file=result_file)
        print('thresh', best_eval['f1'],file=result_file)
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                best_eval['acc'], best_eval['precision'],
                best_eval['recall'], best_eval['f1']),file=result_file)
        result_file.close()
        # self.print_test(best_print_thresh,losslist,labellist,best_print_predlabel,name,seqlist)
        return best_eval
    def series_segmentation(self,data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
    def print_test(self,thresh, losslist ,labellist, predlabel,name,seqlist):
        labellist = np.array(labellist)
        predlabel = np.array(predlabel)
        anomaly_pos = np.array(np.where(labellist==1))[0]
        # losslist = losslist.astype(np.float)
        thresh = (thresh - losslist.min()) / (losslist.max() - losslist.min())
        losslist = (losslist - losslist.min() )/ (losslist.max()-losslist.min())
        l = len(losslist)-1
        plabel = []
        ploss = []
        ppred = []
        pl = 64
        head =0
        tail =0
        flag = 0
        for pos in anomaly_pos:
            head = (pos-pl) if pos>pl else 0
            tail = l if pos+pl>l else (pos+pl)
            tempa = labellist[head:tail]
            tempb = predlabel[head:tail]
            if (tempa==tempb).all():
                plabel = tempa
                ppred = tempb
                ploss = losslist[head:tail]
                # break
                result_file = open('.//pic//print_result.txt', 'a+')
                print('head{0}, tail{1}'.format(head, tail)+name, file=result_file)
                # ploss = losslist
                # ppred = predlabel
                plt.figure(flag)
                plt.plot(range(len(ploss)), ploss)
                # 画出 y=1 这条水平线
                plt.axhline(thresh,c='r',ls='--')
                outlier_idxs = np.where(ppred == 1)[0]
                outcome = self.series_segmentation(outlier_idxs)
                for outlier in outcome:
                    if len(outlier) == 1:
                        plt.plot(outlier, ploss[outlier], 'ro')
                    else:
                        if len(outlier) != 0:
                            plt.axvspan(outlier[0], outlier[-1], color='red', alpha=0.5)
                # plt.savefig('.//pic' + '//'+name+'//'+str(flag)+'test.eps', format='eps')
                # plt.savefig('.//pic' + '//'+name+'//'+str(flag)+'test.svg', format='svg')
                plt.savefig('.//pic' + '//'+name+'//'+str(flag)+'test.png', format='png')
                flag+=1
                plt.figure(flag)
                seq = seqlist[head:tail]
                plt.plot(range(len(seq)), seq)
                outlier_idxs = np.where(plabel == 1)[0]
                outliers = list(seq[outlier_idxs])
                outcome = self.series_segmentation(outlier_idxs)
                for outlier in outcome:
                    if len(outlier) == 1:
                        plt.plot(outlier, seq[outlier], 'ro')
                    else:
                        if len(outlier) != 0:
                            plt.axvspan(outlier[0], outlier[-1], color='red', alpha=0.5)
                # plt.savefig(".//2test.jpg")
                # plt.savefig('.//pic' + '//' + name+'//'+str(flag) + 'ori.eps', format='eps')
                # plt.savefig('.//pic' + '//' + name+'//'+str(flag) + 'ori.svg', format='svg')
                plt.savefig('.//pic' + '//' + name+'//'+str(flag) + 'ori.png', format='png')
                flag+=1
                if flag == 350:
                    break
                # plt.show()
                # plt.show()
    def point_ad(self, losslist, rate, labellist,best_eval,best_rate,best_print_thresh,best_print_predlabel):
        metrics = {'f1': 0}
        thresh = np.percentile(losslist, 100 - rate)
        predlabel = [1 if losslist[l] > thresh else 0 for l in range(len(losslist))]
        # adjust
        anomaly_state = False
        for i in range(len(labellist)):
            if labellist[i] == 1 and predlabel[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if labellist[j] == 0:
                        break
                    else:
                        if predlabel[j] == 0:
                            predlabel[j] = 1
                for j in range(i, len(labellist)):
                    if labellist[j] == 0:
                        break
                    else:
                        if predlabel[j] == 0:
                            predlabel[j] = 1
            elif labellist[i] == 0:
                anomaly_state = False
            if anomaly_state:
                predlabel[i] = 1
        metrics['f1'] = f1_score(y_true=labellist, y_pred=predlabel)  # 认为1正，0负，recall低，正的判为了负的多，就是说实际标签为1的结果判成0了
        metrics['precision'] = precision_score(y_true=labellist, y_pred=predlabel)
        metrics['recall'] = recall_score(y_true=labellist, y_pred=predlabel)
        metrics['acc'] = accuracy_score(y_true=labellist, y_pred=predlabel)
        metrics['test_loss'] = torch.mean(torch.tensor(losslist)) / (len(losslist) + 1)
        # evaluation = f1_score(test_label, estimation, rate, False, False)
        if metrics['f1'] > best_eval['f1']:
            best_eval['f1'] = metrics['f1']
            best_eval['precision'] = metrics['precision']
            best_eval['recall'] = metrics['recall']
            best_eval['acc'] = metrics['acc']
            best_eval['test_loss'] = metrics['test_loss']
            best_rate = rate
            best_print_thresh = thresh
            best_print_predlabel = predlabel
        return best_eval,best_rate,best_print_thresh,best_print_predlabel



    # def finetune(self):
    #     print('finetune')
    #     if self.args.load_pretrained_model:
    #         print('load pretrained model')
    #         state_dict = torch.load(self.save_path + '/pretrain_model.pkl', map_location=self.device)
    #         try:
    #             self.model.load_state_dict(state_dict)
    #         except:
    #             model_state_dict = self.model.state_dict()
    #             for pretrain, random_intial in zip(state_dict, model_state_dict):
    #                 assert pretrain == random_intial
    #                 if pretrain in ['input_projection.weight', 'input_projection.bias', 'predict_head.weight',
    #                                 'predict_head.bias', 'position.pe.weight']:
    #                     state_dict[pretrain] = model_state_dict[pretrain]
    #             self.model.load_state_dict(state_dict)
    #
    #     self.model.eval()
    #     train_rep, train_label = get_rep_with_label(self.model, self.train_linear_loader)
    #     test_rep, test_label = get_rep_with_label(self.model, self.test_loader)
    #     clf = fit_lr(train_rep, train_label)
    #     acc = clf.score(test_rep, test_label)
    #     pred_label = np.argmax(clf.predict_proba(test_rep), axis=1)
    #     f1 = f1_score(test_label, pred_label, average='macro')
    #     print(acc, f1)
    #     result_file = open(self.save_path + '/linear_result.txt', 'a+')
    #     print('epoch{0}, acc{1}, f1{2}'.format(0, acc, f1), file=result_file)
    #     result_file.close()
    #
    #     self.model.linear_proba = False
    #     self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
    #     self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=self.verbose)
    #     for epoch in range(self.num_epoch):
    #         loss_epoch, time_cost = self._train_one_epoch()
    #         self.result_file = open(self.save_path + '/result.txt', 'a+')
    #         self.print_process(
    #             'Finetune epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost))
    #         print('Finetune train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost),
    #               file=self.result_file)
    #         self.result_file.close()
    #     self.print_process(self.best_metric)
    #     return self.best_metric

    # def _train_one_epoch(self):
    #     t0 = time.perf_counter()
    #     self.model.train()
    #     tqdm_dataloader = tqdm(self.train_linear_loader) if self.verbose else self.train_linear_loader
    #
    #     loss_sum = 0
    #     for idx, batch in enumerate(tqdm_dataloader):
    #         batch = [x.to(self.device) for x in batch]
    #
    #         self.optimizer.zero_grad()
    #         loss = self.cr.compute(batch)
    #         loss_sum += loss.item()
    #
    #         loss.backward()
    #         # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
    #         self.optimizer.step()
    #
    #         self.step += 1
    #     # if self.step % self.eval_per_steps == 0:
    #     metric = self.eval_model()
    #     self.print_process(metric)
    #     self.result_file = open(self.save_path + '/result.txt', 'a+')
    #     print('step{0}'.format(self.step), file=self.result_file)
    #     print(metric, file=self.result_file)
    #     self.result_file.close()
    #     if metric[self.metric] >= self.best_metric:
    #         torch.save(self.model.state_dict(), self.save_path + '/model.pkl')
    #         self.result_file = open(self.save_path + '/result.txt', 'a+')
    #         print('saving model of step{0}'.format(self.step), file=self.result_file)
    #         self.result_file.close()
    #         self.best_metric = metric[self.metric]
    #     self.model.train()
    #
    #     return loss_sum / (idx + 1), time.perf_counter() - t0

    # def eval_model(self):
    #     self.model.eval()
    #     tqdm_data_loader = tqdm(self.test_loader) if self.verbose else self.test_loader
    #     metrics = {'acc': 0, 'f1': 0}
    #     pred = []
    #     label = []
    #     test_loss = 0
    #
    #     with torch.no_grad():
    #         for idx, batch in enumerate(tqdm_data_loader):
    #             batch = [x.to(self.device) for x in batch]
    #             ret = self.compute_metrics(batch)
    #             if len(ret) == 2:
    #                 pred_b, label_b = ret
    #                 pred += pred_b
    #                 label += label_b
    #             else:
    #                 pred_b, label_b, test_loss_b = ret
    #                 pred += pred_b
    #                 label += label_b
    #                 test_loss += test_loss_b.cpu().item()
    #     confusion_mat = self._confusion_mat(label, pred)
    #     self.print_process(confusion_mat)
    #     self.result_file = open(self.save_path + '/result.txt', 'a+')
    #     print(confusion_mat, file=self.result_file)
    #     self.result_file.close()
    #     if self.args.num_class == 2:
    #         metrics['f1'] = f1_score(y_true=label, y_pred=pred)
    #         metrics['precision'] = precision_score(y_true=label, y_pred=pred)
    #         metrics['recall'] = recall_score(y_true=label, y_pred=pred)
    #     else:
    #         metrics['f1'] = f1_score(y_true=label, y_pred=pred, average='macro')
    #         metrics['micro_f1'] = f1_score(y_true=label, y_pred=pred, average='micro')
    #     metrics['acc'] = accuracy_score(y_true=label, y_pred=pred)
    #     metrics['test_loss'] = test_loss / (idx + 1)
    #     return metrics

    # def compute_metrics(self, batch):
    #     if len(batch) == 2:
    #         seqs, label = batch
    #         scores = self.model(seqs)
    #     else:
    #         seqs1, seqs2, label = batch
    #         scores = self.model((seqs1, seqs2))
    #     _, pred = torch.topk(scores, 1)
    #     test_loss = self.test_cr(scores, label.view(-1).long())
    #     pred = pred.view(-1).tolist()
    #     return pred, label.tolist(), test_loss

    # def _confusion_mat(self, label, pred):
    #     mat = np.zeros((self.args.num_class, self.args.num_class))
    #     for _label, _pred in zip(label, pred):
    #         mat[_label, _pred] += 1
    #     return mat

    # def print_process(self, *x):
    #     if self.verbose:
    #         print(*x)
