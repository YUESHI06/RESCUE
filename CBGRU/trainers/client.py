import gc
import torch
import numpy as np
import torch.nn as nn
import copy
import sys
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from data_processing.preprocessing import vec2one
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


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


class Fed_PLE_client(object):
    def __init__(
        self,
        args,
        criterion,
        device,
        inner_model,
        outer_model,
        noise_dataloader,
        pure_dataloader,
        valid_dataloader
    ):
        self.args = args
        self.criterion = criterion
        self.device = device
        self.inner_model = inner_model
        self.outer_model = outer_model
        self.noise_dataloader = noise_dataloader
        self.pure_dataloader = pure_dataloader
        self.valid_dataloader = valid_dataloader

    def get_inner_parameters(self):
        return self.inner_model.state_dict()
    
    def get_outer_parameters(self):
        return self.outer_model.state_dict()
    
    def get_all_parameters(self):
        return self.get_inner_parameters(), self.get_outer_parameters()
    
    def print_loss(self):
        print(f"outer_loss is {self.result['outer_loss']}")

    def warm_up(self):
        torch.autograd.set_detect_anomaly(True)
        inner_optimizer = torch.optim.Adam(self.inner_model.parameters(), lr=self.args.inner_lr)

        self.result = dict()
        self.result['sample'] = len(self.pure_dataloader)
        self.inner_model.train()
        for x1, x2, y in self.pure_dataloader:
            x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
            inner_optimizer.zero_grad()
            predictions = self.inner_model(x1, x2)
            predictions = F.log_softmax(predictions, dim=-1)
            labels = y.long().flatten()
            loss = self.criterion(predictions, labels)
            
            loss.backward()
            inner_optimizer.step()

            del x1, x2, y
            del predictions, loss
            torch.cuda.empty_cache()
            gc.collect()
            

    def meta_train(self):
        torch.autograd.set_detect_anomaly(True)

        inner_model_copy = copy.deepcopy(self.inner_model)
        # outer_optimizer = torch.optim.SGD(self.outer_model.parameters(), lr=self.args.outer_lr)
        # inner_optimizer = torch.optim.SGD(self.inner_model.parameters(), lr=self.args.cbgru_local_lr)
        # inner_copy_opt = torch.optim.SGD(inner_model_copy.parameters(), lr=self.args.cbgru_local_lr)
        outer_optimizer = torch.optim.Adam(self.outer_model.parameters(), lr=self.args.outer_lr)
        inner_optimizer = torch.optim.Adam(self.inner_model.parameters(), lr=self.args.cbgru_local_lr)
        inner_copy_opt = torch.optim.Adam(inner_model_copy.parameters(), lr=self.args.cbgru_local_lr)
        
        self.result = dict()
        self.result['sample'] = len(self.noise_dataloader)
        for epoch in range(self.args.local_epoch):
            outer_loss_total = torch.tensor(0., device=self.device)

            for e in range(1):
                self.result['outer_loss'] = torch.tensor(0., device=self.device)
                for (x1, x2, noise_labels, global_labels), (x1_pure, x2_pure, pure_labels) in zip(self.noise_dataloader, self.pure_dataloader):
                    outer_optimizer.zero_grad()

                    x1, x2 = x1.to(self.device), x2.to(self.device)
                    noise_labels, global_labels = noise_labels.to(self.device), global_labels.to(self.device)
                    x1_pure, x2_pure, pure_labels = x1_pure.to(self.device), x2_pure.to(self.device), pure_labels.to(self.device)

                    inner_model_copy.train()
                    self.outer_model.eval()
                    inner_optimizer.zero_grad()
                    predictions = inner_model_copy(x1, x2)
                    predictions = F.softmax(predictions, dim=-1)

                    # 使用钩子函数获取的中间输出
                    h_x = inner_model_copy.inter_outputs
                    h_x.requires_grad = True
                    gl_one_hot = F.one_hot(global_labels.long().flatten(),num_classes=2)
                    gl_one_hot = gl_one_hot.unsqueeze(1)
                    nl_one_hot = F.one_hot(noise_labels.long().flatten(),num_classes=2)
                    nl_one_hot = nl_one_hot.unsqueeze(1)
                    cat_labels = torch.cat((gl_one_hot, nl_one_hot), dim=1)
                    cat_labels = cat_labels.float()
                    cat_labels.requires_grad = True
                    outer_outputs = self.outer_model(h_x, cat_labels)
                    outer_outputs = torch.squeeze(outer_outputs, dim=1)
                    outer_outputs = torch.softmax(outer_outputs, dim=-1)

                    # 内循环
                    inner_loss = nn.functional.kl_div(predictions, outer_outputs, reduction='batchmean')
                    inner_loss.backward()
                    inner_copy_opt.step()
                
                    # inner_model_copy.eval()
                    inner_model_copy.train()
                    self.outer_model.train()
                    updated_predictions = inner_model_copy(x1_pure, x2_pure)
                    pure_labels = pure_labels.long().flatten()
                    outer_loss = self.criterion(updated_predictions, pure_labels)
                    self.result['outer_loss'] = self.result['outer_loss'] + outer_loss.item()
                    
                    outer_loss.backward()
                    outer_optimizer.step()
                    
                    del x1, x2, noise_labels, global_labels
                    del h_x, gl_one_hot, nl_one_hot, cat_labels
                    del outer_outputs, inner_loss, outer_loss
                    torch.cuda.empty_cache()
                    gc.collect()

            for e in range(1):
                for x1, x2, noise_labels, global_labels in self.noise_dataloader:
                    outer_optimizer.zero_grad()
                    x1, x2 =x1.to(self.device), x2.to(self.device)
                    noise_labels, global_labels = noise_labels.to(self.device), global_labels.to(self.device)

                    self.inner_model.train()
                    self.outer_model.eval()
                    inner_optimizer.zero_grad()
                    predictions = self.inner_model(x1, x2)
                    predictions = F.log_softmax(predictions, dim=-1)

                    h_x = self.inner_model.inter_outputs
                    h_x.requires_grad = True
                    gl_one_hot = F.one_hot(global_labels.long().flatten(),num_classes=2)
                    gl_one_hot = gl_one_hot.unsqueeze(1)
                    nl_one_hot = F.one_hot(noise_labels.long().flatten(),num_classes=2)
                    nl_one_hot = nl_one_hot.unsqueeze(1)
                    cat_labels = torch.cat((gl_one_hot, nl_one_hot), dim=1)
                    cat_labels = cat_labels.float()
                    cat_labels.requires_grad = True
                    outer_outputs = self.outer_model(h_x, cat_labels)
                    outer_outputs = torch.squeeze(outer_outputs, dim=1)
                    outer_outputs = torch.softmax(outer_outputs, dim=-1)

                    inner_loss = nn.functional.kl_div(predictions, outer_outputs, reduction='batchmean')
                    inner_loss.backward()
                    inner_optimizer.step()

                    del x1, x2, noise_labels, global_labels
                    del h_x, gl_one_hot, nl_one_hot, cat_labels
                    del outer_outputs, inner_loss

                    torch.cuda.empty_cache()
                    gc.collect()

    def cross_validation(self):
        model_1 = copy.deepcopy(self.model)
        model_2 = copy.deepcopy(self.model)

    def validation(self):
        self.inner_model.eval()
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for x1, x2, y in self.valid_dataloader:
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                softmax = nn.Softmax(dim=1)
                outputs = self.inner_model(x1, x2)
                pred = torch.argmax(softmax(outputs), dim=-1)
                all_predictions.extend(pred.flatten().tolist())
                all_targets.extend(y.flatten().tolist())

                del x1, x2, y
                torch.cuda.empty_cache()
                gc.collect()

            tn, fp, fn, tp = confusion_matrix(all_targets, all_predictions).ravel()
            self.result['Recall(TPR)'] = tp / (tp + fn)
            self.result['Precision'] = tp / (tp + fp)
            self.result['F1 score'] = (2 * self.result['Precision'] * self.result['Recall(TPR)']) / (self.result['Precision'] + self.result['Recall(TPR)'])
        


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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.cbgru_local_lr)
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


class Fed_KNN_client(Fed_Avg_client):
    def __init__(
      self,
      args,
      criterion,
      model,
      dataloader,
    ):
        super().__init__(args, criterion, model, dataloader)
    
    def relabel_with_pretrained_knn(self, features, y, num_neighbors):
        _labels = y.cpu().numpy()
        _labels = _labels.astype(np.int64)
        knn = KNeighborsClassifier(n_neighbors=num_neighbors, weights='uniform', n_jobs=1)
        

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
        names = np.array(self.dataset.names)
        labels = np.array(self.dataset.labels)
        names = names[reserve]
        labels = labels[reserve]
        self.avai_dataset = copy.deepcopy(self.dataset)
        self.avai_dataset.names = names
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


class Fed_Ablation_client(Fed_PLE_client):
    def __init__(self, args, criterion, device, inner_model, outer_model, noise_dataloader, pure_dataloader):
        super().__init__(args, criterion, device, inner_model, outer_model, noise_dataloader, pure_dataloader)

    def print_loss(self):
        print(f"lcn_loss is {self.result['lcn_loss']}")
        print(f"classifier_loss is {self.result['classifier_loss']}")

    def train(self):
        torch.autograd.set_detect_anomaly(True)

        inner_model_copy = copy.deepcopy(self.inner_model)

        outer_optimizer = torch.optim.Adam(self.outer_model.parameters(), lr=self.args.outer_lr)
        inner_optimizer = torch.optim.Adam(self.inner_model.parameters(), lr=self.args.cbgru_local_lr)
        inner_copy_opt = torch.optim.Adam(inner_model_copy.parameters(), lr=self.args.cbgru_local_lr)
        
        self.result = dict()
        self.result['sample'] = len(self.noise_dataloader)
        for epoch in range(self.args.local_epoch):
            outer_loss_total = torch.tensor(0., device=self.device)

            # 训练概率标签模型
            for e in range(1):
                self.result['lcn_loss'] = torch.tensor(0., device=self.device)
                for (x1, x2, noise_labels, global_labels), (x1_pure, x2_pure, pure_labels) in zip(self.noise_dataloader, self.pure_dataloader):
                    outer_optimizer.zero_grad()
                    
                    x1, x2 = x1.to(self.device), x2.to(self.device)
                    noise_labels, global_labels = noise_labels.to(self.device), global_labels.to(self.device)
                    x1_pure, x2_pure, pure_labels = x1_pure.to(self.device), x2_pure.to(self.device), pure_labels.to(self.device)

                    inner_model_copy.eval()
                    self.outer_model.train()
                    predictions = inner_model_copy(x1, x2)
                    predictions = F.softmax(predictions, dim=-1)

                    # 使用钩子函数获取的中间输出
                    h_x = inner_model_copy.inter_outputs
                    h_x.requires_grad = True
                    gl_one_hot = F.one_hot(global_labels.long().flatten(),num_classes=2)
                    gl_one_hot = gl_one_hot.unsqueeze(1)
                    nl_one_hot = F.one_hot(noise_labels.long().flatten(),num_classes=2)
                    nl_one_hot = nl_one_hot.unsqueeze(1)
                    cat_labels = torch.cat((gl_one_hot, nl_one_hot), dim=1)
                    cat_labels = cat_labels.float()
                    cat_labels.requires_grad = True
                    outer_outputs = self.outer_model(h_x, cat_labels)
                    outer_outputs = torch.squeeze(outer_outputs, dim=1)
                    outer_outputs = torch.softmax(outer_outputs, dim=-1)

                    loss = self.criterion(outer_outputs, pure_labels)
                    self.result['lcn_loss'] = self.result['lcn_loss'] + loss.item()

                    loss.backward()
                    outer_optimizer.step()

                    del x1, x2, noise_labels, global_labels
                    del h_x, gl_one_hot, nl_one_hot, cat_labels
                    del outer_outputs
                    torch.cuda.empty_cache()
                    gc.collect()

            for e in range(1):
                self.result['classifier_loss'] = torch.tensor(0., device=self.device)
                for (x1, x2, noise_labels, global_labels), (x1_pure, x2_pure, pure_labels) in zip(self.noise_dataloader, self.pure_dataloader):
                    outer_optimizer.zero_grad()

                    x1, x2 = x1.to(self.device), x2.to(self.device)
                    noise_labels, global_labels = noise_labels.to(self.device), global_labels.to(self.device)
                    x1_pure, x2_pure, pure_labels = x1_pure.to(self.device), x2_pure.to(self.device), pure_labels.to(self.device)

                    self.inner_model.train()
                    self.outer_model.eval()
                    inner_optimizer.zero_grad()
                    predictions = self.inner_model(x1, x2)
                    predictions = F.softmax(predictions, dim=-1)

                    # 使用钩子函数获取的中间输出
                    h_x = self.inner_model.inter_outputs
                    h_x.requires_grad = True
                    gl_one_hot = F.one_hot(global_labels.long().flatten(),num_classes=2)
                    gl_one_hot = gl_one_hot.unsqueeze(1)
                    nl_one_hot = F.one_hot(noise_labels.long().flatten(),num_classes=2)
                    nl_one_hot = nl_one_hot.unsqueeze(1)
                    cat_labels = torch.cat((gl_one_hot, nl_one_hot), dim=1)
                    cat_labels = cat_labels.float()
                    cat_labels.requires_grad = True
                    outer_outputs = self.outer_model(h_x, cat_labels)
                    outer_outputs = torch.squeeze(outer_outputs, dim=1)
                    outer_outputs = torch.softmax(outer_outputs, dim=-1)

                    # 内循环
                    inner_loss = nn.functional.kl_div(predictions, outer_outputs, reduction='batchmean')
                    inner_loss.backward()
                    inner_optimizer.step()
                
                    self.result['classifier_loss'] = self.result['classifier_loss'] + inner_loss.item()
                    
                    del x1, x2, noise_labels, global_labels
                    del h_x, gl_one_hot, nl_one_hot, cat_labels
                    del outer_outputs, inner_loss
                    torch.cuda.empty_cache()
                    gc.collect()