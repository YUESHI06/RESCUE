import numpy as np
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CGE_Variants import CGEVariant
from models.LCN import LCN
from trainers.fed_avg_trainer import ClientFedAvg
from trainers.server import Server
from data_processing.dataloader_manager import gen_client_dataloader, gen_client_noise_dl, gen_client_pure_dl, gen_test_dataloader
from data_processing.preprocessing import get_noise_labels
from options import parse_args
from CGE_test import CGE_test


if __name__ == "__main__":
    args = parse_args()
    criterion = nn.CrossEntropyLoss()
    if args.device != "cpu":
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # 根据client_id取出不同的dataloader,这个dataloader用于给全局模型产生伪标签
    dataloader_dict = dict()
    dataloader_dict['noise'] = dict()
    dataloader_dict['pure'] = dict()
    noise_labels_list = list()

    if args.noise_type == "diff_noise":
        noise_types = random.sample(['fn_noise', 'fn_noise', 'fn_noise', 'fn_noise', 'non_noise', 'non_noise', 'non_noise', 'non_noise'], 8)
        noise_rates = [args.noise_rate]*8
    else:
        noise_types = [args.noise_type]*8
        noise_rates = random.sample([0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3], 8)

    for i in range(args.client_num):
        noise_labels = get_noise_labels(i, args.vul, noise_types[i], noise_rates[i])
        noise_labels_list.append(noise_labels)
        noise_dl = gen_client_dataloader(i, args.vul, noise_types[i], noise_rates[i], noise_labels)
        dataloader_dict['noise'][i] = noise_dl
        pure_dl = gen_client_pure_dl(i, args.vul, noise_types[i], noise_rates[i])
        dataloader_dict['pure'][i] = pure_dl

    
    # 为client和server创建model(pure_dataloader部分需要修改)
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
        inner_model = CGEVariant()
        outer_model = LCN(in_channels=args.input_channels)
        inner_model, outer_model = inner_model.to(device), outer_model.to(device)
        client = ClientFedAvg(
            args,
            criterion,
            device,
            inner_model,
            outer_model,
            None,
            dataloader_dict['pure'][i])
        client_list.append(client)

    for epoch in range(args.warm_up_epoch):
        server.initialize_epoch_updates(epoch)
        for client_id in range(args.client_num):
            client = client_list[client_id]
            client.inner_model = copy.deepcopy(server.global_model)
            client.warm_up()
            server.save_train_updates(
                copy.deepcopy(client.get_parameters()),
                client.result['sample'],
                client.result
            )
            print(f"client:{client_id}")

        server.average_weights()
        print(f"warm up:{epoch}")
    
    # 训练部分
    for epoch in range(args.epoch):
        # 初始化服务器端存储的更新信息
        server.initialize_epoch_updates(epoch)

        for client_id in range(args.client_num):
            global_labels = []
            client = client_list[client_id]
            inner_model = copy.deepcopy(server.global_model)

            with torch.no_grad():
                server.global_model.eval()
                for graph_data, pattern_data, _ in dataloader_dict['noise'][client_id]:
                    graph_data, pattern_data = graph_data.to(device), pattern_data.to(device)
                    outputs = server.global_model(graph_data, pattern_data)
                    outputs = F.softmax(outputs, dim=-1)
                    labels = torch.argmax(outputs, dim=-1)
                    global_labels.append(labels)

            conc_labels = torch.cat(global_labels, dim=0)
            noise_dl = gen_client_noise_dl(client_id, args.vul, noise_types[client_id], conc_labels, noise_labels_list[client_id], noise_rates[client_id])
            client.noise_dataloader = noise_dl
            client.inner_model = inner_model.to(device)
            client.meta_train()
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
    CGE_test(server.global_model, test_dl, criterion, device, args, 'non_wp_Fed_PLE')


    

            

    