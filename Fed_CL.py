import copy
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from options import parse_args
from data_processing.dataloader_manager import gen_cge_dataset, gen_test_dataloader
from trainers.CLC import CLC
from CGE_test import CGE_test


if __name__ == "__main__":
    args = parse_args()

    datasets_dict = dict()
    datasets_dict['train'] = list()
    for i in range(args.client_num):
        train_ds = gen_cge_dataset(i, args.vul, args.noise_type, args.noise_rate)
        datasets_dict['train'].append(train_ds)
    test_dl = gen_test_dataloader(args.vul)
    
    criterion = nn.CrossEntropyLoss()
    clc = CLC(args, 100, 300, datasets_dict['train'], 0.1)
    clc.holdout_stage()
    # clc.correct_stage()

    CGE_test(clc.server.global_model, test_dl, criterion, args.device, args, 'Fed_CL')
