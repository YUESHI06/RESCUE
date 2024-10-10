import os
import json
import torch
import numpy as np
import torch
import pandas
from sklearn.neighbors import KNeighborsClassifier


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
        df = pandas.read_pickle(file_path)
        # print(f"{file_path}加载成功！")
    except FileNotFoundError:
        print(f"错误：{file_path}不存在！")

    x_train = np.stack(df.iloc[:, 0].values)
    labels = df.iloc[:,1].values

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
