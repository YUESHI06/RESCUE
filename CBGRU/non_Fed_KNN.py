import copy
import gc
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from options import parse_args
from data_processing.dataloader_manager import gen_knn_dl, gen_cbgru_valid_dl
from models.ClassiFilerNet import ClassiFilerNet
from trainers.server import Server
from trainers.client import Fed_Avg_client
from global_test import global_test


if __name__ == "__main__":
    args = parse_args()

    dataloader_dict = dict()
    dataloader_dict['train'] = list()
    input_size, time_steps = 0, 0


    if args.noise_type == "diff_noise":
        noise_types = random.sample(['fn_noise', 'fn_noise', 'fn_noise', 'fn_noise', 'non_noise', 'non_noise', 'non_noise', 'non_noise'], 8)
        noise_rates = [args.noise_rate]*8
    else:
        noise_types = [args.noise_type]*8
        noise_rates = random.sample([0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3], 8)

    for i in range(args.client_num):
        train_dl, input_size, time_steps = gen_knn_dl(i, args.vul, noise_types[i], noise_rates[i], args.batch)
        dataloader_dict['train'].append(train_dl)

    criterion = nn.CrossEntropyLoss()

    global_model = ClassiFilerNet(input_size, time_steps)
    global_model = global_model.to(args.device)

    server = Server(
        args,
        global_model,
        args.device,
        criterion
    )

    test_dl = gen_cbgru_valid_dl(args.vul)
    for epoch in range(args.epoch):
        server.initialize_epoch_updates(epoch)

        for client_id in range(args.client_num):
            client = Fed_Avg_client(args,
                                criterion,
                                None,
                                dataloader_dict['train'][client_id])
            client.model = copy.deepcopy(server.global_model)
            client.train()
            server.save_train_updates(
                    copy.deepcopy(client.get_parameters()),
                    client.result['sample'],
                    client.result
            )
            print(f"client:{client_id}")
            client.print_loss()
            del client
            torch.cuda.empty_cache()
            gc.collect()

        server.average_weights()
        print(epoch)
    
    if args.noise_type != "diff_noise":
        args.noise_rate = 2.3
    global_test(server.global_model, test_dl, criterion, args, 'non_Fed_KNN_3')

