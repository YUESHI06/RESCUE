import numpy as np
import torch
import os
import pandas as pd
import json
import random
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import KNeighborsClassifier

def flip_values(names_path, labels_path, noise_rate, noise_type):
    names = []
    with open(names_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            names.append(line.strip())

    with open(labels_path, 'r') as file:
        lines = file.readlines()
        labels = []
        for line in lines:
            labels.append(int(line.strip('\n')))

    name_set = set()
    unique_names = []
    unique_labels = []
    for i, name in enumerate(names):
        if name not in name_set:
            unique_labels.append(labels[i])
            unique_names.append(name)
            name_set.add(name)

    # pure时什么也不做
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


def get_pattern_feature(vul, graph_path):
    # train_total_name_path = f"./data/graph_feature/{vul}/contract_name_train.txt"
    # valid_total_name_path = f"./data/graph_feature/{vul}/contract_name_valid.txt"
    # pattern_feature_path = f"./data/pattern_feature/original_pattern_feature/{vul}/"
    train_total_name_path = os.path.join(graph_path, f'contract_name_train.txt')
    # valid_total_name_path = os.path.join(graph_path, f'contract_name_valid.txt')
    # valid_total_name_path = f"./data/graph_feature/{vul}/contract_name_valid.txt"
    # pattern_feature_path = f"./data/pattern_feature/original_pattern_feature/{vul}/"
    valid_total_name_path = f'./merge_sc_dataset/graph_feature/{vul}/contract_name_valid.txt'
    pattern_feature_path = f"./merge_sc_dataset/pattern_feature/original_pattern_feature/{vul}/"

    final_pattern_feature_train = []  # pattern feature train
    pattern_feature_train_label_path = f"./merge_sc_dataset/pattern_feature/feature_by_zeropadding/{vul}/label_by_extractor_train.txt"
    
    final_pattern_feature_valid = []  # pattern feature valid
    pattern_feature_test_label_path = f"./merge_sc_dataset/pattern_feature/feature_by_zeropadding/{vul}/label_by_extractor_valid.txt"

    f_train = open(train_total_name_path, 'r')
    lines = f_train.readlines()
    for line in lines:
        line = line.strip('\n').split('.')[0]
        tmp_feature = np.loadtxt(pattern_feature_path + line + '.txt')
        final_pattern_feature_train.append(tmp_feature)

    f_test = open(valid_total_name_path, 'r')
    lines = f_test.readlines()
    for line in lines:
        line = line.strip('\n').split('.')[0]
        tmp_feature = np.loadtxt(pattern_feature_path + line + '.txt')
        final_pattern_feature_valid.append(tmp_feature)

    # labels of extractor definition
    label_by_extractor_train = []
    f_train_label_extractor = open(pattern_feature_train_label_path, 'r')
    labels = f_train_label_extractor.readlines()
    for label in labels:
        label_by_extractor_train.append(int(label.strip('\n')))

    label_by_extractor_valid = []
    f_test_label_extractor = open(pattern_feature_test_label_path, 'r')
    labels = f_test_label_extractor.readlines()
    for label in labels:
        label_by_extractor_valid.append(int(label.strip('\n')))

    for i in range(len(final_pattern_feature_train)):
        final_pattern_feature_train[i] = final_pattern_feature_train[i].tolist()

    for i in range(len(final_pattern_feature_valid)):
        final_pattern_feature_valid[i] = final_pattern_feature_valid[i].tolist()
    
    # tranfrom list or numpy array to tensor
    final_pattern_feature_train = torch.tensor(final_pattern_feature_train, dtype=torch.float32)
    final_pattern_feature_valid = torch.tensor(final_pattern_feature_valid, dtype=torch.float32)
    label_by_extractor_train = torch.tensor(label_by_extractor_train, dtype=torch.float32)
    label_by_extractor_train = label_by_extractor_train.reshape(-1,1)
    label_by_extractor_valid = torch.tensor(label_by_extractor_valid, dtype=torch.float32)
    label_by_extractor_valid = label_by_extractor_valid.reshape(-1, 1)

    return final_pattern_feature_train, final_pattern_feature_valid, label_by_extractor_train, label_by_extractor_valid


# def get_graph_feature(vul, noise_type, graph_path, noise_rate=0.05):
#     graph_feature_train_data_path = os.path.join(graph_path, 'train_feature.txt')
#     if noise_type == 'noise':
#         graph_feature_train_label_path = os.path.join(graph_path, f'non_noise_label_train_{noise_rate*100:03.0f}.txt')
#         # graph_feature_train_label_path = os.path.join(graph_path, f'noise_label_train.txt')
#     elif noise_type == 'fn_noise':
#         graph_feature_train_label_path = os.path.join(graph_path, f'FN_noise_label_train_{noise_rate*100:03.0f}.txt')
#     elif noise_type == 'pure':
#         graph_feature_train_label_path = os.path.join(graph_path, 'label_by_experts_train.txt')

#     graph_feature_test_data_path = f"./merge_sc_dataset/graph_feature/{vul}/valid_feature.txt"
#     graph_feature_test_label_path = f"./merge_sc_dataset/graph_feature/{vul}/label_by_experts_valid.txt"

#     #  labels of experts definition
#     print(f'Reading Labels: {graph_feature_train_label_path}...')
#     label_by_experts_train = []
#     f_train_label_expert = open(graph_feature_train_label_path, 'r')
#     labels = f_train_label_expert.readlines()
#     for label in labels:
#         label_by_experts_train.append(int(label.strip('\n')))

#     label_by_experts_valid = []
#     f_test_label_expert = open(graph_feature_test_label_path, 'r')
#     labels = f_test_label_expert.readlines()
#     for label in labels:
#         label_by_experts_valid.append(int(label.strip('\n')))

#     print(f'Reading Graph Features: {graph_feature_train_data_path}...')
#     graph_feature_train = np.loadtxt(graph_feature_train_data_path).tolist()  # graph feature train
#     graph_feature_test = np.loadtxt(graph_feature_test_data_path, delimiter=", ").tolist()  # graph feature test

#     for i in range(len(graph_feature_train)):
#         graph_feature_train[i] = [graph_feature_train[i]]

#     for i in range(len(graph_feature_test)):
#         graph_feature_test[i] = [graph_feature_test[i]]

#     # compute class_weight
#     label_by_experts_train = np.array(label_by_experts_train)
#     class_weights = compute_class_weight('balanced', classes=np.unique(label_by_experts_train), y=label_by_experts_train)
#     if len(class_weights == 1): 
#         pos_weight = 0
#     else:
#         pos_weight = class_weights[1]/class_weights[0]

#     # tranfrom list or numpy array to tensor
#     graph_feature_train = torch.tensor(graph_feature_train, dtype=torch.float32)
#     graph_feature_test = torch.tensor(graph_feature_test, dtype=torch.float32)
#     label_by_experts_train = torch.tensor(label_by_experts_train, dtype=torch.float32)
#     label_by_experts_train = label_by_experts_train.reshape(-1, 1)
#     label_by_experts_valid = torch.tensor(label_by_experts_valid, dtype=torch.float32)
#     label_by_experts_valid = label_by_experts_valid.reshape(-1, 1)

#     return graph_feature_train, graph_feature_test, label_by_experts_train, label_by_experts_valid, pos_weight

def get_noise_labels(client_id, vul, noise_type,  noise_rate):
    graph_path = f'./merge_sc_dataset/client_split/{vul}/client_{client_id}'
    train_total_name_path = os.path.join(graph_path, f'contract_name_train.txt')
    graph_feature_train_label_path = os.path.join(graph_path, 'label_by_experts_train.txt')
    label_by_experts_train = flip_values(train_total_name_path, graph_feature_train_label_path, noise_rate, noise_type)

    return label_by_experts_train

def get_graph_feature(vul, noise_type, graph_path, noise_rate=0.05):
    graph_feature_train_data_path = os.path.join(graph_path, 'train_feature.txt')
    # if noise_type == 'noise':
    #     graph_feature_train_label_path = os.path.join(graph_path, f'non_noise_label_train_{noise_rate*100:03.0f}.txt')
    #     # graph_feature_train_label_path = os.path.join(graph_path, f'noise_label_train.txt')
    # elif noise_type == 'fn_noise':
    #     graph_feature_train_label_path = os.path.join(graph_path, f'FN_noise_label_train_{noise_rate*100:03.0f}.txt')
    # elif noise_type == 'pure':
    #     graph_feature_train_label_path = os.path.join(graph_path, 'label_by_experts_train.txt')

    graph_feature_test_data_path = f"./merge_sc_dataset/graph_feature/{vul}/valid_feature.txt"
    graph_feature_test_label_path = f"./merge_sc_dataset/graph_feature/{vul}/label_by_experts_valid.txt"

    train_total_name_path = os.path.join(graph_path, f'contract_name_train.txt')
    graph_feature_train_label_path = os.path.join(graph_path, 'label_by_experts_train.txt')
    label_by_experts_train = flip_values(train_total_name_path, graph_feature_train_label_path, noise_rate, noise_type)

    #  labels of experts definition
    # print(f'Reading Labels: {graph_feature_train_label_path}...')
    # label_by_experts_train = []
    # f_train_label_expert = open(graph_feature_train_label_path, 'r')
    # labels = f_train_label_expert.readlines()
    # for label in labels:
    #     label_by_experts_train.append(int(label.strip('\n')))

    label_by_experts_valid = []
    f_test_label_expert = open(graph_feature_test_label_path, 'r')
    labels = f_test_label_expert.readlines()
    for label in labels:
        label_by_experts_valid.append(int(label.strip('\n')))

    print(f'Reading Graph Features: {graph_feature_train_data_path}...')
    graph_feature_train = np.loadtxt(graph_feature_train_data_path).tolist()  # graph feature train
    graph_feature_test = np.loadtxt(graph_feature_test_data_path, delimiter=", ").tolist()  # graph feature test

    for i in range(len(graph_feature_train)):
        graph_feature_train[i] = [graph_feature_train[i]]

    for i in range(len(graph_feature_test)):
        graph_feature_test[i] = [graph_feature_test[i]]

    # compute class_weight
    label_by_experts_train = np.array(label_by_experts_train)
    class_weights = compute_class_weight('balanced', classes=np.unique(label_by_experts_train), y=label_by_experts_train)
    if len(class_weights == 1): 
        pos_weight = 0
    else:
        pos_weight = class_weights[1]/class_weights[0]

    # tranfrom list or numpy array to tensor
    graph_feature_train = torch.tensor(graph_feature_train, dtype=torch.float32)
    graph_feature_test = torch.tensor(graph_feature_test, dtype=torch.float32)
    label_by_experts_train = torch.tensor(label_by_experts_train, dtype=torch.float32)
    label_by_experts_train = label_by_experts_train.reshape(-1, 1)
    label_by_experts_valid = torch.tensor(label_by_experts_valid, dtype=torch.float32)
    label_by_experts_valid = label_by_experts_valid.reshape(-1, 1)

    return graph_feature_train, graph_feature_test, label_by_experts_train, label_by_experts_valid, pos_weight


def vec2one(input):
    if input.is_cuda:
        input = input.cpu()
    input = input.numpy()
    one_hot_labels = np.zeros(input.shape[0], 2)
    one_hot_labels[np.arange(input.shape[0]), input.flattern()] = 1
    one_hot_labels = torch.from_numpy(one_hot_labels).to('cuda')
    return one_hot_labels


def get_cbgru_feature(file_path):
    try:
        df = pd.read_pickle(file_path)
        # print(f"{file_path}加载成功！")
    except FileNotFoundError:
        print(f"错误：{file_path}不存在！")

    x_train = np.stack(df.iloc[:, 0].values)
    labels = df.iloc[:,1].values

    # -----
    # positive_index = np.where(labels == 1)[0][:1988]
    # negative_index = np.where(labels == 0)[0][:]
    # undersampled_negative_idxs = np.random.choice(negative_index, len(positive_index), replace=False)
    # resampled_idxs = np.concatenate([positive_index, undersampled_negative_idxs])
    # x_train = vectors[resampled_idxs]
    # labels = labels[resampled_idxs]

    return x_train, labels


def relabel_with_pretrained_knn(labels, features, num_classses, weights='uniform', num_neighbors=10, noise_theshold=0.15):
    # Initialize
    _labels = np.array(labels, dtype=np.int64)
    knn = KNeighborsClassifier(n_neighbors=num_neighbors, weights=weights, n_jobs=1)
    knn.fit(features, _labels)
    knn.classes_ = np.arange(2)
    predictions = np.squeeze(knn.predict(features).astype(np.int64))
    # Estimate label noise
    est_noise_lvl = (predictions!=_labels).astype(np.int64).mean()
    return predictions.astype(np.float32) if est_noise_lvl>=noise_theshold else labels


def read_pretrain_feature(names, feature_dir):
    features = list()
    for name in names:
        name = name.split('.')[0]
        feature_path = os.path.join(feature_dir, f"{name}.json")
        with open(feature_path, 'r') as file:
            feature = np.array(json.load(file))
            feature = feature.reshape(-1)
        features.append(feature)
    
    features = np.array(features)
    return features


def reduced_name_labels(name_path, labels):
    reduced_names = list()
    reduced_labels = list()
    name_set = set()
    with open(name_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            name = line.strip()
            if name not in name_set:
                name_set.add(name)
                reduced_names.append(name)
                reduced_labels.append(labels[i])

    return reduced_names, reduced_labels


if __name__ == "__main__":
    vul = 'reentrancy'
    noise_valid = True
    graph_train, graph_test, graph_experts_train, graph_experts_test = get_graph_feature(vul)
    pattern_train, pattern_test, pattern_experts_train, pattern_experts_test = get_pattern_feature(vul, noise_valid)
    print()

