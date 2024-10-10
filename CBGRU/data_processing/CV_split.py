import numpy as np
import os
from collections import Counter

if __name__ == "__main__":
    vul = 'reentrancy'
    client_num = 8
    noise_rate = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    for id in range(client_num):
        name_path = f'./merge_sc_dataset/client_split/{vul}/client_{id}/contract_name_train.txt'
        feature_path = f'../../smart_contract_data/data/client_split/{vul}/client_{id}/train_feature.txt'
        output_dir = f'../../smart_contract_data/data/client_split/{vul}/client_{id}/CV'
        
        name_set = set()
        origin_name = []
        with open(name_path,'r') as f:
            contracts = f.readlines()
            for index, con_name in enumerate(contracts):
                con_name = con_name.strip()
                origin_name.append(con_name)
                if(con_name not in name_set):
                    name_set.add(con_name)

        graph_feature_train = np.loadtxt(feature_path)
        cv_set = [set() for _ in range(2)]
        name_nums = len(name_set)
        
        cv_size = name_nums // 2
        print(name_nums, cv_size)
        for index, name in enumerate(name_set):
            cv_index = index // cv_size
            if cv_index < cv_size:
                cv_set[cv_index].add(name)
            else:
                cv_set[-1].add(name)


        labels = [[] for _ in range(6)]
        for i in range(6):
            # 改为non-iid标签
            noise_label_path = f'../../smart_contract_data/data/client_split/{vul}/client_{id}/non_noise_label_train_{noise_rate[i]*100:03.0f}.txt'
            with open(noise_label_path, 'r') as f:
                lines = f.readlines()
                for label in lines:
                    labels[i].append(int(label.strip('\n')))
        
        # 分别存储两个cv数据
        for cv_id in range(2):
            cv_dir = os.path.join(output_dir, f'cv_{cv_id}')
            os.makedirs(cv_dir, exist_ok=True)
            output_name_path = os.path.join(cv_dir, 'contract_name_train.txt')
            output_feature_path = os.path.join(cv_dir, 'train_feature.txt')

            output_names = []
            output_labels = [[] for _ in range(6)]
            output_features = []
            for idx, name in enumerate(origin_name):
                if name in cv_set[cv_id]:
                    output_names.append(name)
                    output_features.append(graph_feature_train[idx])
                    for i in range(6):
                        output_labels[i].append(labels[i][idx])

            np.savetxt(output_name_path, output_names, fmt='%s', delimiter=' ')
            np.savetxt(output_feature_path, output_features, fmt='%s', delimiter=' ')
            for i in range(6):
                output_label_path = os.path.join(cv_dir, f'non_noise_label_train_{noise_rate[i]*100:03.0f}.txt')
                np.savetxt(output_label_path, output_labels[i], fmt='%s', delimiter=' ')

