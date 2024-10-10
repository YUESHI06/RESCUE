import torch
import numpy as np
import os
import random
import pandas as pd
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from data_processing.CBGRU_dataset import CBGruDataset, NoiseDataset
from data_processing.preprocessing import get_cbgru_feature, relabel_with_pretrained_knn, read_pretrain_feature, reduced_name_labels
from data_processing.whole_dataset import wholeDataset


# 生成cbgru使用的dataloader
# def gen_cbgru_dl(client_id, vul, noise_type, noise_rate, batch = 16, shuffle=True):
#     embeddings = ['word2vec', 'FastText']
#     file_paths = []

#     if noise_type == 'noise':
#         for emb in embeddings:
#             file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_non_noise_{noise_rate*100:03.0f}_{emb}_fragment_vectors.pkl')
#     elif noise_type == 'fn_noise':
#         # Todo
#         for emb in embeddings:
#             # file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_fn_noise_{noise_rate*100:03.0f}_{emb}_fragment_vectors.pkl')
#             # 更新了fn_noise的读取方式，从纯净数据集读特征，从标签文件读噪声标签
#             file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_pure_{emb}_fragment_vectors.pkl')
#     # 无噪声数据集（与下面的gen_cbgru_client_pure_dl不同，下面的函数用于挑选纯净数据集）
#     elif noise_type == 'pure':
#         for emb in embeddings:
#             file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_pure_{emb}_fragment_vectors.pkl')

#     x1, y = get_cbgru_feature(file_paths[0])
#     x2, _ = get_cbgru_feature(file_paths[1])
#     INPUT_SIZE, TIME_STEPS = x1.shape[1], x1.shape[2]
#     print()

#     # 更新了fn_noise的读取方式，从纯净数据集读特征，从标签文件读噪声标签
#     if noise_type == 'fn_noise':
#         label_path = f"../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_fn_noise_label_train{noise_rate*100:03.0f}.txt"
#         with open(label_path, 'r') as f:
#             lines = f.readlines()
#             labels = list()
#             for line in lines:
#                 label = line.strip()
#                 labels.append(int(label))
#         y = np.array(labels)

#     print(np.count_nonzero(y==0), np.count_nonzero(y==1))
#     x1 = torch.tensor(x1.reshape(-1, 1, 100, 300), dtype=torch.float32)
#     x2 = torch.tensor(x2, dtype=torch.float32)
#     y = torch.tensor(y, dtype=torch.long)
#     y = y.flatten().long()

#     ds = TensorDataset(x1, x2, y)
#     dl = DataLoader(ds, batch_size=batch,shuffle=shuffle)
    
#     return dl, INPUT_SIZE, TIME_STEPS

def gen_cbgru_dl(client_id, vul, noise_type, noise_rate, batch = 16, shuffle=True):
    word2vec_dir = f"./data/cbgru_data/{vul}/word2vec"
    fastText_dir = f"./data/cbgru_data/{vul}/FastText"
    client_dir = f"./data/client_split/{vul}/client_{client_id}/"
    # word2vec_dir = f"./new_dataset/cbgru_data/{vul}/word2vec"
    # fastText_dir = f"./new_dataset/cbgru_data/{vul}/FastText"
    # client_dir = f"./new_dataset/client_split/{vul}/client_{client_id}/"
    names_path = os.path.join(client_dir, "cbgru_contract_name_train.txt")
    labels_path = os.path.join(client_dir, f"non_noise_label_train_000.csv")
    noise_labels = flip_values(names_path, labels_path, noise_rate, noise_type)

    ds = CBGruDataset(word2vec_dir, fastText_dir, labels_path, names_path)
    ds.labels = noise_labels

    # print(f"client {client_id} noise: {len(ds)}")
    dl = DataLoader(ds, batch_size=batch,shuffle=shuffle)
    
    return dl, 100, 300


# def gen_cbgru_client_pure_dl(client_id, vul, noise_type, noise_rate, batch=16):
#     embeddings = ['word2vec', 'FastText']
#     file_paths = []
#     for emb in embeddings:
#         file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_pure_{emb}_fragment_vectors.pkl')
#     x1, y = get_cbgru_feature(file_paths[0])
#     x2, _ = get_cbgru_feature(file_paths[1])
#     input_size, time_stamp = x1.shape[1], x1.shape[2]

#     if noise_type == 'noise':
#         noise_path = f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_non_noise_{noise_rate*100:03.0f}_word2vec_fragment_vectors.pkl'
#     elif noise_type == 'fn_noise':
#         # noise_path = f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_fn_noise_{noise_rate*100:03.0f}_word2vec_fragment_vectors.pkl'
#         noise_path = f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_pure_{emb}_fragment_vectors.pkl'

#     # if noise_type == 'fn_noise':
#     #     label_path = f"../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_fn_noise_label_train{noise_rate*100:03.0f}.txt"
#     #     with open(label_path, 'r') as f:
#     #         lines = f.readlines()
#     #         labels = list()
#     #         for line in lines:
#     #             label = line.strip()
#     #             labels.append(int(label))
#     #     y_noise = np.array(labels)
#     # else:
#     #     _, y_noise = get_cbgru_feature(noise_path)
    

#     # same_indices = np.where(y == y_noisd:\研2\待报销发票\宋恒杰实验室_电子发票.rare)
#     # x1, x2, y = x1[same_indices], x2[same_indices], y[same_indices]

#     x1 = torch.tensor(x1.reshape(-1, 1, 100, 300), dtype=torch.float32)
#     x2 = torch.tensor(x2, dtype=torch.float32)
#     y = torch.tensor(y, dtype=torch.long)
#     y = y.flatten().long()

#     ds = TensorDataset(x1, x2, y)
#     dl = DataLoader(ds, batch_size=batch, shuffle=True)
    
#     return dl, input_size, time_stamp

def gen_cbgru_client_pure_dl(client_id, vul, noise_type, noise_rate, batch=16):
    word2vec_dir = f"./data/cbgru_data/{vul}/word2vec"
    fastText_dir = f"./data/cbgru_data/{vul}/FastText"
    client_dir = f"./data/client_split/{vul}/client_{client_id}/"
    # word2vec_dir = f"./new_dataset/cbgru_data/{vul}/word2vec"
    # fastText_dir = f"./new_dataset/cbgru_data/{vul}/FastText"
    # client_dir = f"./new_dataset/client_split/{vul}/client_{client_id}/"
    names_path = os.path.join(client_dir, "cbgru_contract_name_train.txt")
    labels_path = os.path.join(client_dir, f"non_noise_label_train_000.csv")

    ds = CBGruDataset(word2vec_dir, fastText_dir, labels_path, names_path)
    # print(f"client {client_id} pure: {len(ds)}")
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    
    return dl



def gen_cbgru_valid_dl(vul, id=0, batch=16, frac=1.0):
    embeddings = ['word2vec', 'FastText']
    file_paths = []
    for emb in embeddings:
        file_paths.append(f'../merge_sc_dataset/cbgru_dataset/{vul}/cbgru_valid_dataset_{emb}_fragment_vectors.pkl')
        # file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{id}/cbgru_non_noise_{0.05*100:03.0f}_{emb}_fragment_vectors.pkl')

    x1, y = get_cbgru_feature(file_paths[0])
    x2, _ = get_cbgru_feature(file_paths[1])
    
    x1 = torch.tensor(x1.reshape(-1, 1, 100, 300), dtype=torch.float32)
    x2 = torch.tensor(x2, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    y = y.flatten().long()

    ds = TensorDataset(x1, x2, y)
    dl = DataLoader(ds,batch_size=batch)

    # word2vec_dir = f"./new_dataset/cbgru_data/{vul}/word2vec"
    # fastText_dir = f"./new_dataset/cbgru_data/{vul}/FastText"
    # test_dir = f"./new_dataset/test/{vul}"
    # names_path = os.path.join(test_dir, "cbgru_contract_name_valid.txt")
    # labels_path = os.path.join(test_dir, f"cbgru_label_valid.csv")

    # ds = CBGruDataset(word2vec_dir, fastText_dir, labels_path, names_path)
    # n = len(ds)
    # sub_size = int(n * frac)
    # indices = np.random.permutation(n)[:sub_size]
    # sub_ds = Subset(ds, indices)
    # dl = DataLoader(sub_ds, batch_size=batch,shuffle=False)
    # dl = DataLoader(ds, batch_size=batch,shuffle=False)
    
    return dl
    

# def gen_cbgru_client_noise_dl(client_id, vul, noise_type, global_labels, noise_rate, batch):
#     embeddings = ['word2vec', 'FastText']
#     file_paths = []

#     if noise_type == 'noise':
#         for emb in embeddings:
#             file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_non_noise_{noise_rate*100:03.0f}_{emb}_fragment_vectors.pkl')
#     elif noise_type == 'fn_noise':
#         # Todo
#         for emb in embeddings:
#             file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_pure_{emb}_fragment_vectors.pkl')
#     elif noise_type == 'pure':
#         for emb in embeddings:
#             file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_pure_{emb}_fragment_vectors.pkl')

#     x1, y = get_cbgru_feature(file_paths[0])
#     x2, _ = get_cbgru_feature(file_paths[1])
#     input_size, time_stamp = x1.shape[1], x1.shape[2]
#     if noise_type == 'fn_noise':
#         label_path = f"../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_fn_noise_label_train{noise_rate*100:03.0f}.txt"
#         with open(label_path, 'r') as f:
#             lines = f.readlines()
#             labels = list()
#             for line in lines:
#                 label = line.strip()
#                 labels.append(int(label))
#         y = np.array(labels)

#     x1 = torch.tensor(x1.reshape(-1, 1, 100, 300), dtype=torch.float32)
#     x2 = torch.tensor(x2, dtype=torch.float32)
#     y = torch.tensor(y, dtype=torch.long)
#     global_labels = global_labels.flatten().long()
#     y = y.flatten().long()

#     ds = TensorDataset(x1, x2, y, global_labels)
#     dl = DataLoader(ds, batch, shuffle=True)

#     return dl, input_size, time_stamp


def gen_cbgru_client_valid_dl(client_id, vul, batch, noise_labels, frac):
    word2vec_dir = f"./data/cbgru_data/{vul}/word2vec"
    fastText_dir = f"./data/cbgru_data/{vul}/FastText"
    client_dir = f"./data/client_split/{vul}/client_{client_id}"
    # word2vec_dir = f"./new_dataset/cbgru_data/{vul}/word2vec"
    # fastText_dir = f"./new_dataset/cbgru_data/{vul}/FastText"
    # client_dir = f"./new_dataset/client_split/{vul}/client_{client_id}/"
    names_path = os.path.join(client_dir, "cbgru_contract_name_train.txt")
    labels_path = os.path.join(client_dir, f"non_noise_label_train_000.csv")

    ds = CBGruDataset(word2vec_dir, fastText_dir, labels_path, names_path)
    ds.labels = torch.tensor(np.array(noise_labels))
    n = len(ds)
    sub_size = int(n * frac)
    indices = np.random.permutation(n)[:sub_size]
    sub_ds = Subset(ds, indices)
    dl = DataLoader(sub_ds, batch_size=batch,shuffle=False)
     
    return dl



def gen_cbgru_client_noise_dl(client_id, vul, noise_type, global_labels, noise_rate, batch, noise_labels):
    word2vec_dir = f"./data/cbgru_data/{vul}/word2vec"
    fastText_dir = f"./data/cbgru_data/{vul}/FastText"
    client_dir = f"./data/client_split/{vul}/client_{client_id}"
    # word2vec_dir = f"./new_dataset/cbgru_data/{vul}/word2vec"
    # fastText_dir = f"./new_dataset/cbgru_data/{vul}/FastText"
    # client_dir = f"./new_dataset/client_split/{vul}/client_{client_id}/"
    names_path = os.path.join(client_dir, "cbgru_contract_name_train.txt")
    # labels_path = os.path.join(client_dir, f"{noise_type}_label_train_{noise_rate*100:03.0f}.csv")
    labels_path = os.path.join(client_dir, f"non_noise_label_train_000.csv")

    ds = NoiseDataset(word2vec_dir, fastText_dir, labels_path, names_path, global_labels)
    ds.labels = torch.tensor(np.array(noise_labels))
    dl = DataLoader(ds, batch, shuffle=True)

    return dl

    
def gen_cv_dl(client_id, vul, noise_type, noise_rate, batch):
    embeddings = ['word2vec', 'FastText']
    file_paths = []

    if noise_type == 'noise':
        for emb in embeddings:
            file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_non_noise_{noise_rate*100:03.0f}_{emb}_fragment_vectors.pkl')
    elif noise_type == 'fn_noise':
        # Todo
        for emb in embeddings:
            file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_fn_noise_{noise_rate*100:03.0f}_{emb}_fragment_vectors.pkl')
    # 无噪声数据集（与下面的gen_cbgru_client_pure_dl不同，下面的函数用于挑选纯净数据集）
    elif noise_type == 'pure':
        for emb in embeddings:
            file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_pure_{emb}_fragment_vectors.pkl')

    x1, y = get_cbgru_feature(file_paths[0])
    x2, _ = get_cbgru_feature(file_paths[1])

    cv_bound = y.shape[0] // 2
    print(cv_bound)
    print(x1.shape, x2.shape, y.shape)

    x1_0, x2_0, y_0 = x1[:cv_bound], x2[:cv_bound], y[:cv_bound]
    x1_1, x2_1, y_1 = x1[cv_bound:], x2[cv_bound:], y[cv_bound:]

    x1_0 = torch.tensor(x1.reshape(-1, 1, 100, 300), dtype=torch.float32)
    x2_0 = torch.tensor(x2, dtype=torch.float32)
    y_0 = torch.tensor(y, dtype=torch.long)
    x1_1 = torch.tensor(x1.reshape(-1, 1, 100, 300), dtype=torch.float32)
    x2_1 = torch.tensor(x2, dtype=torch.float32)
    y_1 = torch.tensor(y, dtype=torch.long)

    ds_0 = TensorDataset(x1_0, x2_0, y_0)
    dl_0 = DataLoader(ds_0, batch, shuffle=True)
    ds_1 = TensorDataset(x1_1, x2_1, y_1)
    dl_1 = DataLoader(ds_1, batch, shuffle=True)
    
    return dl_0, dl_1


def gen_arfl_dl(client_id, vul, noise_type, noise_rate, batch):
    embeddings = ['word2vec', 'FastText']
    file_paths = []

    if noise_type == 'noise' or noise_type == 'non_noise':
        for emb in embeddings:
            file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_non_noise_{noise_rate*100:03.0f}_{emb}_fragment_vectors.pkl')
    elif noise_type == 'fn_noise':
        # Todo
        for emb in embeddings:
            file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_fn_noise_{noise_rate*100:03.0f}_{emb}_fragment_vectors.pkl')
    # 无噪声数据集（与下面的gen_cbgru_client_pure_dl不同，下面的函数用于挑选纯净数据集）
    elif noise_type == 'pure':
        for emb in embeddings:
            file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_pure_{emb}_fragment_vectors.pkl')

    x1, y = get_cbgru_feature(file_paths[0])
    x2, _ = get_cbgru_feature(file_paths[1])
    INPUT_SIZE, TIME_STEPS = x1.shape[1], x1.shape[2]
    
    x1 = torch.tensor(x1.reshape(-1, 1, 100, 300), dtype=torch.float32)
    x2 = torch.tensor(x2, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    y = y.flatten().long()

    ds = TensorDataset(x1, x2, y)
    dl = DataLoader(ds, batch_size=batch,shuffle=True)

    return dl, len(ds), INPUT_SIZE, TIME_STEPS
    
    
def gen_whole_dataset(client_num, vul, noise_type, noise_rate):
    embeddings = ['word2vec', 'FastText']
    x1_list = []
    x2_list = []
    y_list = []
    data_indices = []
    input_size, time_steps = 0, 0

    offset = 0
    for client_id in range(client_num):
        file_paths = []

        if isinstance(noise_rate, list):
            nr = noise_rate[client_id]
        else:
            nr = noise_rate

        if isinstance(noise_type, list):
            nt = noise_type[client_id]
        else:
            nt = noise_type

        if nt == 'noise' or nt == 'non_noise':
            for emb in embeddings:
                file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_non_noise_{nr*100:03.0f}_{emb}_fragment_vectors.pkl')
        elif nt == 'fn_noise':
            # Todo
            for emb in embeddings:
                file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_fn_noise_{nr*100:03.0f}_{emb}_fragment_vectors.pkl')
        # 无噪声数据集（与下面的gen_cbgru_client_pure_dl不同，下面的函数用于挑选纯净数据集）
        elif nt == 'pure':
            for emb in embeddings:
                file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_pure_{emb}_fragment_vectors.pkl')

        x1, y = get_cbgru_feature(file_paths[0])
        x2, _ = get_cbgru_feature(file_paths[1])
        input_size, time_steps = x1.shape[1], x1.shape[2]
        
        x1 = torch.tensor(x1.reshape(-1, 1, 100, 300), dtype=torch.float32)
        x2 = torch.tensor(x2, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        y = y.flatten().long()

        x1_list.append(x1)
        x2_list.append(x2)
        y_list.append(y)

        n_data = y.shape[0]
        bound = offset+n_data
        data_indices.append(list(range(offset, bound)))
        offset = bound
    
    x1 = torch.cat(x1_list, dim=0)
    x2 = torch.cat(x2_list, dim=0)
    y = torch.cat(y_list, dim=0)
    # ds = TensorDataset(x1, x2, y)
    ds = wholeDataset(x1, x2, y)
    return ds,input_size, time_steps, data_indices


def gen_knn_dl(client_id, vul, noise_type, noise_rate, batch):
    embeddings = ['word2vec', 'FastText']
    file_paths = []

    if noise_type == 'noise' or noise_type == 'non_noise':
        for emb in embeddings:
            file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_non_noise_{noise_rate*100:03.0f}_{emb}_fragment_vectors.pkl')
    elif noise_type == 'fn_noise':
        # Todo
        for emb in embeddings:
            file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_fn_noise_{noise_rate*100:03.0f}_{emb}_fragment_vectors.pkl')
    # 无噪声数据集（与下面的gen_cbgru_client_pure_dl不同，下面的函数用于挑选纯净数据集）
    elif noise_type == 'pure':
        for emb in embeddings:
            file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_pure_{emb}_fragment_vectors.pkl')
    
    x1, y = get_cbgru_feature(file_paths[0])
    x2, _ = get_cbgru_feature(file_paths[1])
    input_size, time_steps = x1.shape[1], x1.shape[2]

    # Run CNN relable
    feature_dir = f'../merge_sc_dataset/source_code/{vul}/pretrain_feature'
    name_path = f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_contract_name_train.txt'
    relabel_path = f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/{noise_type}_{noise_rate}_relabel.txt'
    if not os.path.exists(relabel_path):
        reduced_names, reduced_labels = reduced_name_labels(name_path, y)
        features = read_pretrain_feature(reduced_names, feature_dir)
        relabels = relabel_with_pretrained_knn(reduced_labels, features, 2, 'uniform', 3, 0.15)
        
        name_relabels = dict()
        for i, name in enumerate(reduced_names):
            name_relabels[name] = relabels[i]

        relabels = list()
        # read name list, map labels
        with open(name_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                name = line.strip()
                relabels.append(name_relabels[name])

    x1 = torch.tensor(x1.reshape(-1, 1, 100, 300), dtype=torch.float32)
    x2 = torch.tensor(x2, dtype=torch.float32)
    print(x1.shape, len(relabels))
    y = torch.tensor(relabels)
    y = y.flatten().long()

    ds = TensorDataset(x1, x2, y)
    dl = DataLoader(ds, batch, shuffle=True)
    return dl, input_size, time_steps
    

def flip_values(names_path, labels_path, noise_rate, noise_type):
    names = []
    with open(names_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            names.append(line.strip())

    with open(labels_path, 'rb') as file:
        df = pd.read_csv(labels_path, header=None)
        labels = df.iloc[:, 0].values

    name_set = set()
    unique_names = []
    unique_labels = []
    for i, name in enumerate(names):
        if name not in name_set:
            unique_labels.append(labels[i])
            unique_names.append(name)
            name_set.add(name)

    for i in range(len(unique_labels)):
        if noise_type == 'fn_noise':
            if unique_labels[i] == 1:
                if random.random() < noise_rate:
                    unique_labels[i] = 1-unique_labels[i]
        elif noise_type == 'non_nosie':
            if random.random() < noise_rate:
                unique_labels[i] = 1-unique_labels[i]
    
    name_label = dict()
    for i in range(len(unique_names)):
        name_label[unique_names[i]] = unique_labels[i]
    
    noise_labels = []
    for name in names:
        noise_labels.append(name_label[name])
    
    return noise_labels


def gen_cbgru_ds(client_id, vul, noise_type, noise_rate, batch = 16, shuffle=True):
    word2vec_dir = f"./data/cbgru_data/{vul}/word2vec"
    fastText_dir = f"./data/cbgru_data/{vul}/FastText"
    client_dir = f"./data/client_split/{vul}/client_{client_id}/"
    names_path = os.path.join(client_dir, "cbgru_contract_name_train.txt")
    # labels_path = os.path.join(client_dir, f"{noise_type}_label_train_{noise_rate*100:03.0f}.csv")
    labels_path = os.path.join(client_dir, f"non_noise_label_train_000.csv")
    noise_labels = flip_values(names_path, labels_path, noise_rate, noise_type)

    ds = CBGruDataset(word2vec_dir, fastText_dir, labels_path, names_path)
    ds.labels = noise_labels

    return ds


def get_noise_labels(args, noise_types, noise_rates):
    noise_labels = []
    for i in range(args.client_num):
        client_dir = f"./data/client_split/{args.vul}/client_{i}"
        # client_dir = f"./new_dataset/client_split/{args.vul}/client_{i}"
        names_path = os.path.join(client_dir, "cbgru_contract_name_train.txt")
        labels_path = os.path.join(client_dir, f"non_noise_label_train_000.csv")
        noise_labels.append(flip_values(names_path, labels_path, noise_rates[i], noise_types[i]))
    return noise_labels

    


    