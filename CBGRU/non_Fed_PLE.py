import sys
import gc
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ClassiFilerNet import ClassiFilerNet
from models.LCN import LCN
from trainers.client import Fed_PLE_client
from trainers.server import Server
from data_processing.dataloader_manager import gen_cbgru_dl, gen_cbgru_client_pure_dl, gen_cbgru_valid_dl, gen_cbgru_client_noise_dl, get_noise_labels
from options import parse_args
from global_test import global_test


# 需要调整ClassiFilerNet和input_channel，需要读取ClassiFilerNet的中间输出作为outer_model的输入

if __name__ == "__main__":
    args = parse_args()
    criterion = nn.CrossEntropyLoss()
    if args.device != "cpu":
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # 根据client_id生成不同的dataloader
    input_size, time_stamp = 0, 0
    dataloader_dict = dict()
    dataloader_dict["noise"] = list()
    dataloader_dict["pure"] = list()

    if args.noise_type == "diff_noise":
        noise_types = random.sample(['fn_noise', 'fn_noise', 'fn_noise', 'fn_noise', 'non_noise', 'non_noise', 'non_noise', 'non_noise'], 8)
        noise_rates = [args.noise_rate]*8
    else:
        noise_types = [args.noise_type]*8
        noise_rates = random.sample([0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3], 8)

    for i in range(args.client_num):
        noise_dl, input_size, time_stamp = gen_cbgru_dl(client_id=i, vul=args.vul, noise_type=noise_types[i], noise_rate=noise_rates[i], batch=args.batch, shuffle=False)
        dataloader_dict["noise"].append(noise_dl)
        pure_dl = gen_cbgru_client_pure_dl(client_id=i, vul = args.vul, noise_type=noise_types[i], noise_rate=noise_rates[i], batch=args.batch)
        dataloader_dict["pure"].append(pure_dl)
    noise_labels = get_noise_labels(args, noise_types, noise_rates)
    
    # 初始化Server
    global_model = ClassiFilerNet(input_size, time_stamp)
    global_model = global_model.to(device)
    server = Server(
        args,
        global_model,
        device,
        criterion
    )

    # 初始化Client
    client_list = list()
    for i in range(args.client_num):
        inner_model = copy.deepcopy(server.global_model)
        outer_model = LCN(in_channels=args.input_channels)
        inner_model, outer_model = inner_model.to(device), outer_model.to(device)
        client = Fed_PLE_client(
            args,
            criterion,
            device,
            inner_model,
            outer_model,
            None,
            dataloader_dict["pure"][i]
        )
        client_list.append(client)

    # 训练部分
    for epoch in range(args.epoch):
        server.initialize_epoch_updates(epoch)
        
        for client_id in range(args.client_num):
            global_labels = []
            client = client_list[client_id]
            client.inner_model = copy.deepcopy(server.global_model)

            # 生成全局预测标签
            with torch.no_grad():
                server.global_model.eval()
                for x1, x2, _ in dataloader_dict["noise"][client_id]:
                    x1, x2 = x1.to(device), x2.to(device)
                    outputs = server.global_model(x1, x2)
                    outputs = F.softmax(outputs, dim=-1)
                    labels = torch.argmax(outputs, dim=-1)
                    global_labels.append(labels)

                    del x1, x2, outputs, labels
                    torch.cuda.empty_cache()
                    gc.collect()

            conc_labels = torch.cat(global_labels, dim = 0)
            noise_dl = gen_cbgru_client_noise_dl(client_id, args.vul, noise_types[client_id], conc_labels, noise_rates[client_id], args.batch, noise_labels[client_id])
            client.noise_dataloader = noise_dl

            # 本地训练并保存训练结果
            client.meta_train()
            server.save_train_updates(
                copy.deepcopy(client.get_inner_parameters()),
                client.result["sample"],
                client.result
            )
            print(f"client:{client_id} train")
            client.print_loss()
        
        server.average_weights()
        print(epoch)
    
    test_dl = gen_cbgru_valid_dl(args.vul, id=0, batch=args.batch)
    if args.noise_type != "diff_noise":
        args.noise_rate = 2.3
    global_test(server.global_model, test_dl, criterion, args, 'non_Fed_CBGRU_PLE')
            



