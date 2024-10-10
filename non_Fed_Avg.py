import numpy as np
import copy
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from models.CGE_Variants import CGEVariant
from trainers.CGE_client import CGE_client
from trainers.server import Server
from data_processing.dataloader_manager import gen_client_dataloader, gen_test_dataloader
from models.RCELoss import RCELoss
from options import parse_args
from CGE_test import CGE_test


if __name__ == '__main__':
    args = parse_args()
    args.inner_lr = 0.0005
    criterion = nn.CrossEntropyLoss()
    re_criterion = RCELoss()
    device = args.device

    datloader_dict = dict()
    datloader_dict['train'] = list()

    if args.noise_type == "diff_noise":
        noise_types = random.sample(['fn_noise', 'fn_noise', 'fn_noise', 'fn_noise', 'non_noise', 'non_noise', 'non_noise', 'non_noise'], 8)
        noise_rates = [args.noise_rate]*8
    else:
        noise_types = [args.noise_type]*8
        noise_rates = random.sample([0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3], 8)

    for i in range(args.client_num):
        train_dl = gen_client_dataloader(i, args.vul, noise_types[i], noise_rates[i])
        datloader_dict['train'].append(train_dl)
    
    global_model = CGEVariant()
    global_model = global_model.to(device)
    server = Server(
        args,
        global_model,
        device,
        criterion
    )
    
    client_list = list()
    for i in range(args.client_num):
        model = CGEVariant()
        model = model.to(device)
        client = CGE_client(
            args,
            criterion,
            re_criterion,
            device,
            model,
            datloader_dict['train'][i],
            i
        )
        client_list.append(client)

    for epoch in range(args.epoch):
        server.initialize_epoch_updates(epoch)

        for client_id in range(args.client_num):
            client = client_list[client_id]
            client.model = copy.deepcopy(server.global_model)
            client.RCE_train()
            server.save_train_updates(
                    copy.deepcopy(client.get_parameters()),
                    client.result['sample'],
                    client.result
            )
            print(f"client:{client_id}")
            client.print_loss()
    
        server.average_weights()
        print(epoch)

test_dl = gen_test_dataloader(args.vul)
if args.noise_type != 'diff_noise':
    args.noise_rate = 2.3
CGE_test(server.global_model, test_dl, criterion, device, args, 'non_Fed_CGE')

    