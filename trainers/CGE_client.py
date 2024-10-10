import torch
import numpy as np
import torch.nn as nn
import copy
import sys
import gc
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from data_processing.preprocessing import vec2one
from data_processing.CustomDataset import CustomDataset
from .evaluation import Evaluation
from data_processing.dataloader_manager import gen_client_cv_dl, gen_client_cr_dl

class CGE_client(object):
    def __init__(
        self,
        args,
        criterion,
        re_criterion,
        device,
        model,
        dataloader,
        client_id
    ):
        self.args = args
        self.criterion = criterion
        self.re_criterion = re_criterion
        self.device = device
        self.model = model
        self.dataloader = dataloader
        self.client_id = client_id

    def get_parameters(self):
        return self.model.state_dict()
    
    def print_loss(self):
        print(f"loss is {self.result['loss']}")

    def cross_validation(self):
        model_1 = copy.deepcopy(self.model)
        model_2 = copy.deepcopy(self.model)
        dl_1 = gen_client_cv_dl(self.client_id, 0, self.args.vul,)
        dl_2 = gen_client_cv_dl(self.client_id, 1, self.args.vul)
        opt_1 = torch.optim.Adam(model_1.parameters(), self.args.inner_lr)
        opt_2 = torch.optim.Adam(model_2.parameters(), self.args.inner_lr)
        
        model_1.train()
        for e in range(20):
            for graph_data, pattern_data, labels in dl_1:
                opt_1.zero_grad()
                graph_data, pattern_data, labels = graph_data.to(self.device), pattern_data.to(self.device), labels.to(self.device)
                pred = model_1(graph_data, pattern_data)
                labels = labels.long().flatten()
                loss = self.criterion(pred, labels)
                loss.backward()
                opt_1.step()
        
        model_2.train()
        for e in range(20):
            for graph_data, pattern_data, labels in dl_2:
                opt_2.zero_grad()
                graph_data, pattern_data, labels = graph_data.to(self.device), pattern_data.to(self.device), labels.to(self.device)
                pred = model_2(graph_data, pattern_data)
                labels = labels.long().flatten()
                loss = self.criterion(pred, labels)
                loss.backward()
                opt_2.step()
        
        model_1.eval()
        model_2.eval()
        cv_graph = list()
        cv_pattern = list()
        cv_labels = list()
        with torch.no_grad():
            for graph_data, pattern_data, labels in dl_2:
                graph_data, pattern_data, labels = graph_data.to(self.device), pattern_data.to(self.device), labels.to(self.device)
                pred = model_1(graph_data, pattern_data)
                pred = F.softmax(pred, dim=-1)
                outputs = torch.argmax(pred, dim=-1).long()
                labels = labels.long().flatten()
                indices = torch.nonzero(torch.eq(outputs, labels), as_tuple=False).squeeze(dim=1)
                if indices.numel() != 0:
                    cv_graph.append(graph_data[indices])
                    cv_pattern.append(pattern_data[indices])
                    cv_labels.append(labels[indices])

            for graph_data, pattern_data, labels in dl_1:
                graph_data, pattern_data, labels = graph_data.to(self.device), pattern_data.to(self.device), labels.to(self.device)
                pred = model_2(graph_data, pattern_data)
                pred = F.softmax(pred, dim=-1)
                outputs = torch.argmax(pred, dim=-1).long()
                labels = labels.long().flatten()
                indices = torch.nonzero(torch.eq(outputs,labels), as_tuple=False).squeeze(dim=1)
                if indices.numel() != 0:
                    cv_graph.append(graph_data[indices])
                    cv_pattern.append(pattern_data[indices])
                    cv_labels.append(labels[indices])
            
        pure_graph = torch.cat(cv_graph, dim=0)
        pure_pattern = torch.cat(cv_pattern, dim=0)
        pure_labels = torch.cat(cv_labels, dim=0)
        pure_ds = CustomDataset(pure_graph, pure_pattern, pure_labels)
        self.dataloader = DataLoader(dataset=pure_ds, batch_size=16, shuffle=True)

    def CV_train(self):
        torch.autograd.set_detect_anomaly(True)
        # opt = torch.optim.SGD(self.model.parameters(), lr=self.args.inner_lr)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.inner_lr)

        self.result = dict()
        self.result['sample'] = len(self.dataloader)

        self.model.train()
        self.result['loss'] = torch.tensor(0., device=self.device)
        for epoch in range(self.args.local_epoch):
            for graph_data, pattern_data, labels in self.dataloader:
                opt.zero_grad()
                graph_data, pattern_data, labels = graph_data.to(self.device), pattern_data.to(self.device), labels.to(self.device)
                pred = self.model(graph_data, pattern_data)
                labels = labels.long().flatten()
                loss = self.criterion(pred, labels)
                self.result['loss'] = self.result['loss'] + loss.item()
                loss.backward()
                opt.step()

    def RCE_train(self):
        torch.autograd.set_detect_anomaly(True)
        opt = torch.optim.SGD(self.model.parameters(), lr=self.args.inner_lr)
        # opt = torch.optim.Adam(self.model.parameters(), lr=self.args.inner_lr)

        self.result = dict()
        self.result['sample'] = len(self.dataloader)

        self.model.train()
        self.result['loss'] = torch.tensor(0., device=self.device)
        for epoch in range(self.args.local_epoch):
            for graph_data, pattern_data, labels in self.dataloader:
                opt.zero_grad()
                graph_data, pattern_data, labels = graph_data.to(self.device), pattern_data.to(self.device), labels.to(self.device)
                pred = self.model(graph_data, pattern_data)
                labels = labels.long().flatten()
                loss_1 = self.criterion(pred, labels)
                # one_hot_labels = F.one_hot(labels,num_classes=2)
                # pred = F.softmax(pred, dim=-1)
                # loss_2 = self.re_criterion(one_hot_labels,pred)
                # total_loss = self.args.alpha*loss_1 + self.args.beta*loss_2
                # self.result['loss'] = self.result['loss'] + total_loss.item()
                # total_loss.backward()

                self.result['loss'] = self.result['loss'] + loss_1.item()
                loss_1.backward()
                opt.step()

    def CR_train(self):
        torch.autograd.set_detect_anomaly(True)
        opt = torch.optim.SGD(self.model.parameters(), lr=self.args.inner_lr)
        # opt = torch.optim.Adam(self.model.parameters(), lr=self.args.inner_lr)
        self.dataloader, self.pos_weight = gen_client_cr_dl(self.client_id, self.args.vul, self.args.noise_rate)

        self.result = dict()
        self.result['sample'] = len(self.dataloader)
        self.model.train()
        self.result['loss'] = torch.tensor(0., device=self.device)

        self.belta = 0.4

        for epoch in range(self.args.local_epoch):
            for graph_data, pattern_data, labels in self.dataloader:
                opt.zero_grad()
                graph_data, pattern_data, labels = graph_data.to(self.device), pattern_data.to(self.device), labels.to(self.device)
                pred = self.model(graph_data, pattern_data)
                labels = labels.long().flatten()

                loss_1 = self.criterion(pred, labels)
                loss_zero_labels = self.criterion(pred, torch.zeros(labels.shape[0]).long().to(self.device))
                loss_one_labels = self.criterion(pred, torch.ones(labels.shape[0]).long().to(self.device))
                expected_loss = (1-self.pos_weight)*loss_zero_labels + self.pos_weight*loss_one_labels
                loss_total = loss_1 - self.belta*expected_loss
                self.result['loss'] = self.result['loss'] + loss_total.item()

                loss_total.backward()
                opt.step()


class Fed_Avg_client(object):
    def __init__(
      self,
      args,
      criterion,
      model,
      dataloader,
    ):
        self.args = args
        self.criterion = criterion
        self.model = model
        self.dataloader = dataloader
        self.device = args.device

    def get_parameters(self):
        return self.model.state_dict()
    
    def print_loss(self):
        print(f"loss is {self.result['loss']}")

    def cross_validation(self, dl_1, dl_2):
        model_1 = copy.deepcopy(self.model)
        model_2 = copy.deepcopy(self.model)
        opt_1 = torch.optim.Adam(model_1.parameters(), lr=self.args.inner_lr)
        opt_2 = torch.optim.Adam(model_2.parameters(), lr=self.args.inner_lr)

        model_1.train()
        for e in range(20):
            for x1, x2, y in dl_1:
                opt_1.zero_grad()
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                pred = model_1(x1, x2)
                y = y.long().flatten()
                loss = self.criterion(pred, y)
                loss.backward()
                opt_1.step()
        
        model_2.train()
        for e in range(20):
            for x1, x2, y in dl_2:
                opt_2.zero_grad()
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                pred = model_2(x1, x2)
                y = y.long().flatten()
                loss = self.criterion(pred, y)
                loss.backward()
                opt_2.step()
        
        model_1.eval()
        model_2.eval()
        cv_x1 = list()
        cv_x2 = list()
        cv_y = list()
        with torch.no_grad():
            for x1, x2, y in dl_2:
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                outputs = model_1(x1, x2)
                outputs = F.softmax(outputs, dim=-1)
                preds = torch.argmax(outputs, dim=-1).long()
                y = y.long().flatten()
                indices = torch.nonzero(torch.eq(preds, y), as_tuple=False).squeeze(dim=1)
                if indices.numel() != 0:
                    cv_x1.append(x1[indices])
                    cv_x2.append(x2[indices])
                    cv_y.append(y[indices])

            for x1, x2, y in dl_1:
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                outputs = model_2(x1, x2)
                outputs = F.softmax(outputs, dim=-1)
                preds = torch.argmax(outputs, dim=-1).long()
                y = y.long().flatten()
                indices = torch.nonzero(torch.eq(preds, y), as_tuple=False).squeeze(dim=1)
                if indices.numel() != 0:
                    cv_x1.append(x1[indices])
                    cv_x2.append(x2[indices])
                    cv_y.append(y[indices])
        
        pure_x1 = torch.cat(cv_x1, dim=0)
        pure_x2 = torch.cat(cv_x2, dim=0)
        pure_y = torch.cat(cv_y, dim=0)
        pure_ds = TensorDataset(pure_x1, pure_x2, pure_y)
        pure_dl = DataLoader(dataset=pure_ds, batch_size=self.args.batch, shuffle=True)
        return pure_dl
           
    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.cbgru_local_lr)

        self.result = dict()
        device = self.args.device
        self.result['sample'] = len(self.dataloader)

        self.model.train()
        for epoch in range(self.args.cbgru_local_epoch):
            self.result['loss'] = 0
            for x1, x2, y in self.dataloader:
                optimizer.zero_grad()
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                outputs = self.model(x1, x2)
                y = y.flatten().long()
                loss = self.criterion(outputs, y)
                self.result['loss'] = self.result['loss'] + loss.item()
                loss.backward()
                optimizer.step()

                del x1, x2, y, outputs, loss
                torch.cuda.empty_cache()
                gc.collect()


class Fed_ARFL_client(object):
    def __init__(
        self,
        args,
        criterion,
        model,
        dataloader,
        weight,
        num_train_samples,
    ):
        self.args = args
        self.device = args.device
        self.criterion = criterion
        self.model = model
        self.dataloader = dataloader
        self.weight = weight
        self.num_train_samples = num_train_samples
    
    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.inner_lr)
        self.result = dict()
        device = self.device

        for epoch in range(self.args.local_epoch):
            self.model.train()
            self.result['loss'] = 0
            for x1, x2 ,y in self.dataloader:
                optimizer.zero_grad()
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                outputs = self.model(x1, x2)
                y = y.flatten().long()

                loss = self.criterion(outputs, y)

                self.result['loss'] = self.result['loss'] + loss.item()
                loss.backward()
                optimizer.step()
                
                del x1, x2, y, outputs, loss
                torch.cuda.empty_cache()
                gc.collect()
    
    def test(self):
        device = self.device
        
        with torch.no_grad():
            # self.result['test_loss'] = 0
            self.test_loss = 0
            self.model.eval()
            for x1, x2, y in self.dataloader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                outputs = self.model(x1, x2)

                y = y.flatten().long()
                loss = self.criterion(outputs, y)
                # self.result['test_loss'] = self.result['test_loss'] + loss.item()
                self.test_loss += loss.item()
                del x1, x2, y, outputs, loss
                torch.cuda.empty_cache()
                gc.collect()

    def get_model_parameters(self):
        return self.model.state_dict()

    def get_test_loss(self):
        # return self.result['test_loss']
        return self.test_loss
    
    def set_weight(self, weight):
        self.weight = weight


class Fed_Corr_client(Fed_Avg_client):
    def __init__(
        self,
        args,
        criterion,
        model,
        dataloader
    ):
        super().__init__(
            args,
            criterion,
            model,
            dataloader
        )

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.cbgru_local_lr)

        self.result = dict()
        device = self.args.device
        self.result['sample'] = len(self.dataloader)

        self.model.train()
        for epoch in range(self.args.cbgru_local_epoch):
            self.result['loss'] = 0
            for x1, x2, y in self.dataloader:
                optimizer.zero_grad()
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                outputs = self.model(x1, x2)
                y = y.flatten().long()
                loss = self.criterion(outputs, y)
                loss = loss.mean()
                self.result['loss'] = self.result['loss'] + loss.item()
                loss.backward()
                optimizer.step()

                del x1, x2, y, outputs, loss
                torch.cuda.empty_cache()
                gc.collect()

    
    def get_output(self):
        self.model.eval()
        with torch.no_grad():
            for i, (x1, x2, y) in enumerate(self.dataloader):
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                y = y.long()

                outputs = self.model(x1, x2)
                outputs = F.softmax(outputs, dim=1)

                loss = self.criterion(outputs, y)
                if i == 0:
                    outputs_whole = np.array(outputs.cpu())
                    loss_whole = np.array(loss.cpu())
                else:
                    output_whole = np.concatenate((output_whole, outputs.cpu()), axis=0)
                    loss_whole = np.concatenate((loss_whole, loss.cpu()), axis=0)

        return output_whole, loss_whole


class Fed_CLC_client(object):
    def __init__(
        self, 
        args,
        criterion,
        model,
        dataset,
        client_id,
        tao
    ):
        self.args = args
        self.criterion = criterion
        self.model = model
        self.dataset = dataset
        self.client_id = client_id
        self.tao = tao
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch, shuffle=True)

    def get_parameters(self):
        return self.model.state_dict()
    
    def print_loss(self):
        print(f"loss is {self.result['loss']}")

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.cbgru_local_lr)

        self.result = dict()
        device = self.args.device
        self.result['sample'] = len(self.dataloader)

        self.model.train()
        for epoch in range(self.args.cbgru_local_epoch):
            self.result['loss'] = 0
            for x1, x2, y in self.dataloader:
                optimizer.zero_grad()
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                outputs = self.model(x1, x2)
                y = y.flatten().long()
                loss = self.criterion(outputs, y)
                self.result['loss'] = self.result['loss'] + loss.item()
                loss.backward()
                optimizer.step()

                del x1, x2, y, outputs, loss
                torch.cuda.empty_cache()
                gc.collect()

    def sendconf(self):
        confListU, class_nums = self.confidence()
        sys.stdout.write('\r')
        sys.stdout.write('User = [%d/%d]  | confidence is computed '
                         % (self.client_id, self.args.client_num))
        sys.stdout.flush()
        return confListU, class_nums

    def data_holdout(self, conf_score):
        r = self.sfm_Mat.shape[0]
        delta_sort = {}
        naive_num = 0
        self.keys = []
        self.sudo_labels = []
        for idx in range(r):
            if idx == 26:
                debug = True
            softmax = self.sfm_Mat[idx]

            maxPro_Naive = -1
            preIndex_Naive = -1
            maxPro = -1
            preIndex = -1

            for j in range(self.args.num_classes):
                if softmax[j] > maxPro_Naive:
                    preIndex_Naive = j
                    maxPro_Naive = softmax[j]

                if softmax[j] > conf_score[j]:
                    if softmax[j] > maxPro:
                        maxPro = softmax[j]
                        preIndex = j

            label = int(softmax[-1])
            margin = maxPro_Naive - softmax[label]

            if preIndex == -1:
                preIndex = preIndex_Naive
                maxPro = maxPro_Naive
                naive_num += 1
            elif preIndex != label:
                delta_sort[idx] = margin
            
            self.sudo_labels.append(preIndex)
        
        delta_sorted = sorted(delta_sort.items(), key=lambda delta_sort: delta_sort[1], reverse=True)  # 降序
        reserve = []

        for (k, v) in delta_sorted:
            if v > self.tao:
                self.keys.append(k)

        # 对于没有被放入到keys中的样本,保留下来
        for idx in range(r):
            if idx not in self.keys:
                reserve.append(idx)

        for idx in range(r):
            if idx not in self.keys:
                reserve.append(idx)
        # names = np.array(self.dataset.names)
        graphs = np.array(self.dataset.graph_feature)
        patterns = np.array(self.dataset.pattern_feature)
        labels = np.array(self.dataset.labels)
        graphs = graphs[reserve]
        patterns = patterns[reserve]
        self.avai_dataset = copy.deepcopy(self.dataset)
        self.avai_dataset.graph_feature = graphs
        self.avai_dataset.pattern_feature = patterns
        self.avai_dataset.labels = labels
        self.data_loader = DataLoader(self.avai_dataset, batch_size=self.args.batch, shuffle=True)

    def confidence(self):
        outputSofma = self.outputSof()
        r = outputSofma.shape[0]
        c = outputSofma.shape[1]
        prob_everyclass = [[] for i in range(c - 1)]
        class_nums = []
        confList = []

        for i in range(r):
            oriL = outputSofma[i][c - 1]
            oriL = int(oriL)
            pro = outputSofma[i, oriL]
            prob_everyclass[oriL].append(pro)

        for i in range(c - 1):
            confList.append(round(np.mean(prob_everyclass[i], axis=0), 3))
            class_nums.append(len(prob_everyclass[i]))
        self.sfm_Mat = outputSofma

        return confList, class_nums

    def outputSof(self):
        dataset = self.dataset
        s = dataset.labels
        s = np.array(s)

        self.model.eval()
        device = self.args.device
        val_loader = DataLoader(dataset, batch_size=self.args.batch, shuffle=False)
        outputs = []
        with torch.no_grad():
            for x1, x2, labels in val_loader:
                x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
                if len(outputs) == 0:
                    outputs = self.model(x1, x2)
                else:
                    outputs = torch.cat([outputs, self.model(x1, x2)], dim=0)
        
        psx_cv = F.softmax(outputs, dim=1)
        
        psx = psx_cv.cpu().numpy().reshape((-1, self.args.num_classes))
        s = s.reshape([s.size, 1])
        sfm_Mat = np.hstack((psx, s))

        return sfm_Mat

    def data_correct(self):
        self.avai_dataset.labels = self.sudo_labels
        self.data_loader = DataLoader(self.avai_dataset, batch_size=self.args.batch, shuffle=True)


class CGE_Graph_Client(CGE_client):
    def __init__(self, args, criterion, re_criterion, device, model, dataloader, client_id):
        super().__init__(args, criterion, re_criterion, device, model, dataloader, client_id)
    
    def train(self):
        torch.autograd.set_detect_anomaly(True)
        opt = torch.optim.SGD(self.model.parameters(), lr=self.args.inner_lr)

        self.result = dict()
        self.result['sample'] = len(self.dataloader)

        self.model.train()
        self.result['loss'] = torch.tensor(0., device=self.device)
        for epoch in range(self.args.local_epoch):
            for graph_data, _, labels in self.dataloader:
                opt.zero_grad()
                graph_data, labels = graph_data.to(self.device), labels.to(self.device)
                pred = self.model(graph_data)
                labels = labels.long().flatten()
                loss_1 = self.criterion(pred, labels)

                self.result['loss'] = self.result['loss'] + loss_1.item()
                loss_1.backward()
                opt.step()