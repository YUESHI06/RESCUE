import sys
import gc
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ClassiFilerNet import ClassiFilerNet
from models.LCN import LCN
from trainers.fed_avg_trainer import ClientFedAvg_CBGRU
from trainers.server import Server
from data_processing.dataloader_manager import gen_cbgru_dl, gen_cbgru_client_pure_dl, gen_cbgru_valid_dl, gen_cbgru_client_noise_dl
from options import parse_args
from CGE_test import CBGRU_test


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
    for i in range(args.client_num):
        noise_dl, input_size, time_stamp = gen_cbgru_dl(client_id=i, vul=args.vul, noise_type=args.noise_type, noise_rate=args.noise_rate, batch=args.batch)
        dataloader_dict["noise"].append(noise_dl)
        pure_dl, _, _ = gen_cbgru_client_pure_dl(client_id=i, vul = args.vul, noise_type=args.noise_type, noise_rate=args.noise_rate, batch=args.batch)
        dataloader_dict["pure"].append(pure_dl)
    
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
        client = ClientFedAvg_CBGRU(
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
            noise_dl, _, _ = gen_cbgru_client_noise_dl(client_id, args.vul, args.noise_type, conc_labels, args.noise_rate, args.batch)
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
    CBGRU_test(server.global_model, test_dl, criterion, args, 'Fed_CBGRU_PLE')
            



