import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset, TensorDataset
from data_processing.preprocessing import get_graph_feature, get_pattern_feature, get_cbgru_feature, relabel_with_pretrained_knn, read_pretrain_feature, reduced_name_labels, get_noise_labels
from data_processing.CustomDataset import CustomDataset
from data_processing.ClientDataset import ClientDataset
from data_processing.WholeDataset  import wholeDataset


def gen_client_dataloader(client_id, vul, noise_type, noise_rate=0.05, noise_labels=None, shuffle=False):
    # graph_path = f'../smart_contract_data/data/client_split/{vul}/client_{client_id}'
    graph_path = f'./merge_sc_dataset/client_split/{vul}/client_{client_id}'
    graph_train, _, labels_train, _, _ = get_graph_feature(vul, noise_type, graph_path, noise_rate)
    pattern_train, _, _ , _ = get_pattern_feature(vul, graph_path)
    
    if noise_type !='pure':
        if noise_labels == None:
            noise_labels = get_noise_labels(client_id, vul, noise_type, noise_rate)
        labels_train = noise_labels

    dataset = CustomDataset(graph_train, pattern_train, labels_train)
    dl = DataLoader(dataset, batch_size=32, shuffle=shuffle)
    return dl


def gen_test_dataloader(vul):
    # graph_path = f'./data/graph_feature/{vul}'
    graph_path = f'./merge_sc_dataset/graph_feature/{vul}'
    graph_train, graph_test, labels_train, labels_test, _ = get_graph_feature(vul, 'pure', graph_path)
    pattern_train, pattern_test, _, _ = get_pattern_feature(vul, graph_path)
    test_ds = CustomDataset(graph_test, pattern_test, labels_test)
    test_dl = DataLoader(test_ds, batch_size=32)
    return test_dl


# 生成带有全局模型预测标签的dataloader
def gen_client_noise_dl(client_id, vul, noise_type, global_labels, noise_labels, noise_rate=0.05):
    # graph_path = f'../smart_contract_data/data/client_split/{vul}/client_{client_id}'
    graph_path = f'./merge_sc_dataset/client_split/{vul}/client_{client_id}'
    graph_train, _, _, _, _ = get_graph_feature(vul, noise_type, graph_path, noise_rate)
    pattern_train, _, _ , _ = get_pattern_feature(vul, graph_path)
    dataset = ClientDataset(graph_train, pattern_train, noise_labels, global_labels)
    dl = DataLoader(dataset, batch_size=32, shuffle=True)
    return dl


def gen_client_pure_dl(client_id, vul, noise_valid, noise_rate):
    # graph_path = f'../smart_contract_data/data/client_split/{vul}/client_{client_id}'
    graph_path = f'./merge_sc_dataset/client_split/{vul}/client_{client_id}'
    graph_train, _, labels_train, _, _ = get_graph_feature(vul, "pure", graph_path)
    pattern_train, _, _ , _ = get_pattern_feature(vul, graph_path)
    dataset = CustomDataset(graph_train, pattern_train, labels_train)
    dl = DataLoader(dataset,batch_size=32, shuffle=True)
    return dl


def gen_client_cv_dl(client_id, cv_id, vul, noise_rate=0.05):
    graph_path = f'./smart_contract_data/data/client_split/{vul}/client_{client_id}/CV/cv_{cv_id}'
    graph_train, _, labels_train, _, _ = get_graph_feature(vul, True, graph_path, noise_rate)
    pattern_train, _, _, _ = get_pattern_feature(vul, graph_path)
    dataset = CustomDataset(graph_train, pattern_train, labels_train)
    dl = DataLoader(dataset, batch_size=32, shuffle=True)
    return dl


def gen_client_cr_dl(client_id, vul, noise_rate=0.05):
    graph_path = f'./smart_contract_data/data/client_split/{vul}/client_{client_id}'
    # graph_path = f'../FedCorr/data/client_split/{vul}/client_{client_id}/'
    graph_train, _, noise_labels, _, pos_weight = get_graph_feature(vul, True, graph_path, noise_rate)
    pattern_train, _, _, _ = get_pattern_feature(vul, graph_path)
    dataset = CustomDataset(graph_train, pattern_train, noise_labels)
    dl = DataLoader(dataset, batch_size=32, shuffle=True)
    return dl, pos_weight


def gen_arfl_dl(client_id, vul, noise_type, noise_rate=0.05):
    graph_path = f'./merge_sc_dataset/client_split/{vul}/client_{client_id}'
    graph_train, _, labels_train, _, _ = get_graph_feature(vul, noise_type, graph_path, noise_rate)
    pattern_train, _, _ , _ = get_pattern_feature(vul, graph_path)
    dataset = CustomDataset(graph_train, pattern_train, labels_train)
    dl = DataLoader(dataset, batch_size=32, shuffle=False)

    return dl, len(dataset)


def gen_knn_dl(client_id, vul, noise_type, noise_rate, batch=32):
    graph_path = f'./merge_sc_dataset/client_split/{vul}/client_{client_id}'
    graph_train, _, labels_train, _, _ = get_graph_feature(vul, noise_type, graph_path, noise_rate)
    pattern_train, _, _ , _ = get_pattern_feature(vul, graph_path)

    # Run CNN relable
    feature_dir = f'./merge_sc_dataset/source_code/{vul}/pretrain_feature'
    name_path = os.path.join(graph_path, f'contract_name_train.txt')
    relabel_path = f'./merge_sc_dataset/client_split/{vul}/client_{client_id}/{noise_type}_{noise_rate}_relabel.txt'
    if not os.path.exists(relabel_path):
        labels_train = labels_train.detach().cpu().numpy().ravel()
        reduced_names, reduced_labels = reduced_name_labels(name_path, labels_train)
        features = read_pretrain_feature(reduced_names, feature_dir)
        relabels = relabel_with_pretrained_knn(reduced_labels, features, 2, 'uniform', 5, 0.15)
        
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

    relabels = torch.tensor(relabels, dtype=torch.float32)
    dataset = CustomDataset(graph_train, pattern_train, relabels)
    dl = DataLoader(dataset, batch_size=batch, shuffle=False)

    return dl


# 生成cbgru使用的dataloader
def gen_cbgru_dl(client_id, vul, noise_type, noise_rate, batch = 16):
    embeddings = ['word2vec', 'FastText']
    file_paths = []

    if noise_type == 'noise' or noise_type == 'non_noise':
        for emb in embeddings:
            file_paths.append(f'./merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_non_noise_{noise_rate*100:03.0f}_{emb}_fragment_vectors.pkl')
    elif noise_type == 'fn_noise':
        # Todo
        for emb in embeddings:
            file_paths.append(f'./merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_fn_noise_{noise_rate*100:03.0f}_{emb}_fragment_vectors.pkl')
    # 无噪声数据集（与下面的gen_cbgru_client_pure_dl不同，下面的函数用于挑选纯净数据集）
    elif noise_type == 'pure':
        for emb in embeddings:
            file_paths.append(f'./merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_pure_{emb}_fragment_vectors.pkl')

    x1, y = get_cbgru_feature(file_paths[0])
    x2, _ = get_cbgru_feature(file_paths[1])
    INPUT_SIZE, TIME_STEPS = x1.shape[1], x1.shape[2]
    
    x1 = torch.tensor(x1.reshape(-1, 1, 100, 300), dtype=torch.float32)
    x2 = torch.tensor(x2, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    y = y.flatten().long()

    ds = TensorDataset(x1, x2, y)
    dl = DataLoader(ds, batch_size=batch,shuffle=True)
    
    return dl, INPUT_SIZE, TIME_STEPS


def gen_cbgru_client_pure_dl(client_id, vul, noise_type, noise_rate, batch=16):
    embeddings = ['word2vec', 'FastText']
    file_paths = []
    for emb in embeddings:
        file_paths.append(f'./merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_pure_{emb}_fragment_vectors.pkl')

    if noise_type == 'noise' or noise_type == 'non_noise':
        noise_path = f'./merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_non_noise_{noise_rate*100:03.0f}_word2vec_fragment_vectors.pkl'
    elif noise_type == 'fn_noise':
        noise_path = f'./merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_fn_noise_{noise_rate*100:03.0f}_word2vec_fragment_vectors.pkl'
    
    x1, y = get_cbgru_feature(file_paths[0])
    x2, _ = get_cbgru_feature(file_paths[1])
    input_size, time_stamp = x1.shape[1], x1.shape[2]
    # _, y_noise = get_cbgru_feature(noise_path)

    # same_indices = np.where(y == y_noise)
    # x1, x2, y = x1[same_indices], x2[same_indices], y[same_indices]

    x1 = torch.tensor(x1.reshape(-1, 1, 100, 300), dtype=torch.float32)
    x2 = torch.tensor(x2, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    y = y.flatten().long()

    ds = TensorDataset(x1, x2, y)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    
    return dl, input_size, time_stamp


def gen_cbgru_valid_dl(vul, id=0, batch=16):
    embeddings = ['word2vec', 'FastText']
    file_paths = []
    for emb in embeddings:
        file_paths.append(f'./merge_sc_dataset/cbgru_dataset/{vul}/cbgru_valid_dataset_{emb}_fragment_vectors.pkl')
        # file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{id}/cbgru_non_noise_{0.05*100:03.0f}_{emb}_fragment_vectors.pkl')

    x1, y = get_cbgru_feature(file_paths[0])
    x2, _ = get_cbgru_feature(file_paths[1])
    
    x1 = torch.tensor(x1.reshape(-1, 1, 100, 300), dtype=torch.float32)
    x2 = torch.tensor(x2, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    y = y.flatten().long()

    ds = TensorDataset(x1, x2, y)
    dl = DataLoader(ds,batch_size=batch)
    
    return dl
    

def gen_cbgru_client_noise_dl(client_id, vul, noise_type, global_labels, noise_rate, batch):
    embeddings = ['word2vec', 'FastText']
    file_paths = []

    if noise_type == 'noise' or noise_type == 'non_noise':
        for emb in embeddings:
            file_paths.append(f'./merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_non_noise_{noise_rate*100:03.0f}_{emb}_fragment_vectors.pkl')
    elif noise_type == 'fn_noise':
        # Todo
        for emb in embeddings:
            file_paths.append(f'./merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_fn_noise_{noise_rate*100:03.0f}_{emb}_fragment_vectors.pkl')
    elif noise_type == 'pure':
        for emb in embeddings:
            file_paths.append(f'./merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_pure_{emb}_fragment_vectors.pkl')

    x1, y = get_cbgru_feature(file_paths[0])
    x2, _ = get_cbgru_feature(file_paths[1])
    input_size, time_stamp = x1.shape[1], x1.shape[2]

    x1 = torch.tensor(x1.reshape(-1, 1, 100, 300), dtype=torch.float32)
    x2 = torch.tensor(x2, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    global_labels = global_labels.flatten().long()
    y = y.flatten().long()

    ds = TensorDataset(x1, x2, y, global_labels)
    dl = DataLoader(ds, batch, shuffle=True)

    return dl, input_size, time_stamp


def gen_whole_dataset(client_num, vul, noise_type, noise_rate):
    embeddings = ['word2vec', 'FastText']
    x1_list = []
    x2_list = []
    y_list = []
    data_indices = []
    input_size, time_steps = 0, 0

    offset = 0
    for client_id in range(client_num):
        graph_path = f'./merge_sc_dataset/client_split/{vul}/client_{client_id}'

        if isinstance(noise_rate, list):
            nr = noise_rate[client_id]
        else:
            nr = noise_rate

        if isinstance(noise_type, list):
            nt = noise_type[client_id]
        else:
            nt = noise_type

        graph_train, _, labels_train, _, _ = get_graph_feature(vul, nt, graph_path, nr)
        pattern_train, _, _ , _ = get_pattern_feature(vul, graph_path)

        
        x1 = torch.tensor(graph_train, dtype=torch.float32)
        x2 = torch.tensor(pattern_train, dtype=torch.float32)
        y = torch.tensor(labels_train, dtype=torch.long)
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


def gen_cge_dataset(client_id, vul, noise_type, noise_rate=0.05):
    graph_path = f'./merge_sc_dataset/client_split/{vul}/client_{client_id}'
    graph_train, _, labels_train, _, _ = get_graph_feature(vul, noise_type, graph_path, noise_rate)
    pattern_train, _, _ , _ = get_pattern_feature(vul, graph_path)
    dataset = CustomDataset(graph_train, pattern_train, labels_train)
    return dataset
    


    