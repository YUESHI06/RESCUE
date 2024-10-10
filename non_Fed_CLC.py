import copy
import gc
import random
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

    if args.noise_type == "diff_noise":
        noise_types = random.sample(['fn_noise', 'fn_noise', 'fn_noise', 'fn_noise', 'non_noise', 'non_noise', 'non_noise', 'non_noise'], 8)
        noise_rates = [args.noise_rate]*8
    else:
        noise_types = [args.noise_type]*8
        noise_rates = random.sample([0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3], 8)

    for i in range(args.client_num):
        train_ds = gen_cge_dataset(i, args.vul, noise_types[i], noise_rates[i])
        datasets_dict['train'].append(train_ds)
    test_dl = gen_test_dataloader(args.vul)
    
    criterion = nn.CrossEntropyLoss()
    clc = CLC(args, 100, 300, datasets_dict['train'], 0.1)
    clc.holdout_stage()
    clc.correct_stage()

    if args.noise_type != 'diff_noise':
        args.noise_rate = 2.3
    CGE_test(clc.server.global_model, test_dl, criterion, args.device, args, 'non_Fed_CLC')
