import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ClassiFilerNet import ClassiFilerNet
from trainers.client import Fed_Avg_client
from trainers.server import Server
from data_processing.dataloader_manager import gen_cbgru_dl, gen_cbgru_valid_dl, gen_cv_dl, gen_cbgru_client_pure_dl
from options import parse_args
from global_test import global_test


if __name__ == "__main__":
    args = parse_args()
    criterion = nn.CrossEntropyLoss()
    device = args.device

    _, input_size, time_stamp = gen_cbgru_client_pure_dl(client_id=0, vul = args.vul, noise_type=args.noise_type, noise_rate=args.noise_rate, batch=args.batch)
    global_model = ClassiFilerNet(input_size, time_stamp)
    global_model = global_model.to(device)
    server = Server(
        args,
        global_model,
        device,
        criterion
    )

    dataloader_list = list()
    for i in range(args.client_num):
        dl = gen_cbgru_dl(i, args.vul, args.noise_type, args.noise_rate, args.batch)
        client = Fed_Avg_client(
            args,
            criterion,
            copy.deepcopy(server.global_model),
            dl
        )
        dl_1, dl_2 = gen_cv_dl(i, args.vul, args.noise_type, args.noise_rate, args.batch)
        cv_dl = client.cross_validation(dl_1, dl_2)
        dataloader_list.append(cv_dl)
        del client

    for epoch in range(args.epoch):
        server.initialize_epoch_updates(epoch)

        for client_id in range(args.client_num):
            client = Fed_Avg_client(
                args,
                criterion,
                copy.deepcopy(server.global_model),
                dataloader_list[client_id]
            )
            client.train()
            server.save_train_updates(
                copy.deepcopy(client.get_parameters()),
                client.result["sample"],
                client.result
            )
            print(f"client:{client_id} train")
            client.print_loss()
        
        server.average_weights()
        print(epoch)

    test_dl = gen_cbgru_valid_dl(args.vul, id=0, batch=args.batch)
    global_test(server.global_model, test_dl, criterion, args, 'Fed_CBGRU_CV')

