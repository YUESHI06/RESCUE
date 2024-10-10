import torch
import numpy as np
import torch.nn as nn
import copy
import torch.nn.functional as F
import gc


class CBGRU_client(object):
    def __init__(
      self,
      args,
      criterion,
      model,
      dataloader      
    ):
        self.args = args
        self.criterion = criterion
        self.model = model
        self.dataloader = dataloader

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

