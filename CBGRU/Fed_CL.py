import copy
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from options import parse_args
from data_processing.dataloader_manager import gen_cbgru_ds, gen_cbgru_valid_dl
from trainers.CLC import CLC
from global_test import global_test


if __name__ == "__main__":
    args = parse_args()

    datasets_dict = dict()
    datasets_dict['train'] = list()
    for i in range(args.client_num):
        train_ds = gen_cbgru_ds(i, args.vul, args.noise_type, args.noise_rate, args.batch)
        datasets_dict['train'].append(train_ds)
    test_dl = gen_cbgru_valid_dl(args.vul)
    
    criterion = nn.CrossEntropyLoss()
    clc = CLC(args, 100, 300, datasets_dict['train'], 0.1)
    clc.holdout_stage()

    global_test(clc.server.global_model, test_dl, criterion, args, 'Fed_CL')
