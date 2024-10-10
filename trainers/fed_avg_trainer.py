import gc
import torch
import numpy as np
import torch.nn as nn
import copy
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_processing.preprocessing import vec2one
from data_processing.LabelDataset import LabelDataset
from .evaluation import Evaluation

class ClientFedAvg(object):
    def __init__(
        self,
        args,
        criterion,
        device,
        inner_model,
        outer_model,
        noise_dataloader,
        pure_dataloader
    ):
        self.args = args
        self.criterion = criterion
        self.device = device
        # the main model
        self.inner_model = inner_model
        # the LCN model
        self.outer_model = outer_model
        self.noise_dataloader = noise_dataloader
        self.pure_dataloader = pure_dataloader

    def get_parameters(self):
        return self.inner_model.state_dict()
    
    def get_parameters_2(self):
        return self.inner_model.state_dict(), self.outer_model.state_dict()
            
    def print_loss(self):
        print(f"outer_loss is {self.result['outer_loss']}")

    def meta_train(self):
        torch.autograd.set_detect_anomaly(True)

        inner_model_copy = copy.deepcopy(self.inner_model)
        # outer_optimizer = torch.optim.SGD(self.outer_model.parameters(), lr=self.args.outer_lr)
        # inner_optimizer = torch.optim.SGD(self.inner_model.parameters(), lr=self.args.inner_lr)
        # # inner_optimizer = torch.optim.Adam(self.inner_model.parameters(), lr=0.001)
        # inner_copy_opt = torch.optim.SGD(inner_model_copy.parameters(), lr=self.args.inner_lr)
        outer_optimizer = torch.optim.Adam(self.outer_model.parameters(), lr=self.args.outer_lr)
        inner_optimizer = torch.optim.Adam(self.inner_model.parameters(), lr=self.args.inner_lr)
        # inner_optimizer = torch.optim.Adam(self.inner_model.parameters(), lr=0.001)
        inner_copy_opt = torch.optim.Adam(inner_model_copy.parameters(), lr=self.args.inner_lr)
        
        self.result = dict()
        self.result['sample'] = len(self.noise_dataloader)
        for epoch in range(self.args.local_epoch):
            # outer_loss_total = 0
            outer_loss_total = torch.tensor(0., device=self.device)

            for e in range(3):
                self.result['outer_loss'] = torch.tensor(0., device=self.device)
                for (graph_data, pattern_data, noise_labels, global_labels), (pure_graph, pure_pattern, pure_labels) in zip(self.noise_dataloader, self.pure_dataloader):
                    outer_optimizer.zero_grad()

                    graph_data, pattern_data = graph_data.to(self.device), pattern_data.to(self.device)
                    noise_labels, global_labels = noise_labels.to(self.device), global_labels.to(self.device)
                    pure_graph, pure_pattern, pure_labels = pure_graph.to(self.device), pure_pattern.to(self.device), pure_labels.to(self.device)

                    inner_model_copy.train()
                    self.outer_model.eval()
                    inner_optimizer.zero_grad()
                    predictions = inner_model_copy(graph_data, pattern_data)
                    predictions = F.log_softmax(predictions, dim=-1)

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
                    inner_loss = nn.functional.kl_div(predictions, outer_outputs, reduction='mean')
                    inner_loss.backward()
                    inner_copy_opt.step()
                
                    outer_optimizer.zero_grad()
                    inner_model_copy.eval()
                    self.outer_model.train()
                    updated_predictions = inner_model_copy(pure_graph, pure_pattern)
                    pure_labels = pure_labels.long().flatten()
                    outer_loss = self.criterion(updated_predictions, pure_labels)
                    self.result['outer_loss'] = self.result['outer_loss'] + outer_loss.item()
                    # self.result['outer_loss'] = self.result['outer_loss'] + outer_loss.item()

                    outer_loss.backward()
                    outer_optimizer.step()
                self.print_loss()

            for e in range(1):
                for graph_data, pattern_data, noise_labels, global_labels in self.noise_dataloader:
                    outer_optimizer.zero_grad()
                    graph_data, pattern_data = graph_data.to(self.device), pattern_data.to(self.device)
                    noise_labels, global_labels = noise_labels.to(self.device), global_labels.to(self.device)

                    self.inner_model.train()
                    self.outer_model.eval()
                    inner_optimizer.zero_grad()
                    predictions = self.inner_model(graph_data, pattern_data)
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

                    inner_loss = nn.functional.kl_div(predictions, outer_outputs, reduction='mean')
                    inner_loss.backward()
                    inner_optimizer.step()
        
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


        
class ClientFedAvg_CBGRU(object):
    def __init__(
        self,
        args,
        criterion,
        device,
        inner_model,
        outer_model,
        noise_dataloader,
        pure_dataloader
    ):
        self.args = args
        self.criterion = criterion
        self.device = device
        self.inner_model = inner_model
        self.outer_model = outer_model
        self.noise_dataloader = noise_dataloader
        self.pure_dataloader = pure_dataloader

    def get_inner_parameters(self):
        return self.inner_model.state_dict()
    
    def get_outer_parameters(self):
        return self.outer_model.state_dict()
    
    def get_all_parameters(self):
        return self.get_inner_parameters(), self.get_outer_parameters()
    
    def print_loss(self):
        print(f"outer_loss is {self.result['outer_loss']}")

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

            for e in range(3):
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
        