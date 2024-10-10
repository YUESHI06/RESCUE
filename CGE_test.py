import torch
import torch.nn as nn
import json
import os
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from models.CGE_Variants import CGEVariant
from data_processing.preprocessing import get_graph_feature, get_pattern_feature
from data_processing.CustomDataset import CustomDataset


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vul = 'timestamp'
    noise_valid = False

    graph_path =  f'./data/graph_feature/{vul}'
    graph_train, graph_test, graph_experts_train, graph_experts_test, pos_weight = get_graph_feature(vul, noise_valid, graph_path)
    pattern_train, pattern_test, extractor_train, extractor_test = get_pattern_feature(vul, graph_path)
    # print(graph_train.shape, pattern_train.shape,graph_experts_train.shape)
    train_dataset = CustomDataset(graph_train, pattern_train, graph_experts_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = CGEVariant()
    model.to(device)

    pos_tensor = torch.tensor([pos_weight], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model.train()
    for epoch in range(50):
        for graph_feature, pattern_feature, labels in train_loader:
            graph_feature, pattern_feature, labels = graph_feature.to(device), pattern_feature.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(graph_feature, pattern_feature)
            one_hot_labels = F.one_hot(labels.long().flatten(), num_classes=2)
            one_hot_labels = one_hot_labels.float() 
            # print(outputs.shape, one_hot_labels.shape)
            loss = criterion(outputs, one_hot_labels)
            loss.backward()
            optimizer.step()
        print(loss.item())

    test_dataset = CustomDataset(graph_test, pattern_test, graph_experts_test)
    test_loader = DataLoader(test_dataset, shuffle=False)
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():  # 在评估阶段，不需要计算梯度
        for graph, pattern, targets in test_loader:
            graph, pattern, targets = graph.to(device), pattern.to(device), targets.to(device)
            outputs = model(graph, pattern)
            total_loss = 0
            correct_predictions = 0
            one_hot_targets = F.one_hot(targets.long().flatten(), num_classes=2)
            one_hot_targets = one_hot_targets.float()
            loss = criterion(outputs, one_hot_targets)
            total_loss += loss.item()
            # predictions = torch.sigmoid(outputs).round()
            predictions = F.log_softmax(outputs, dim=-1)
            predictions = torch.argmax(predictions, dim=-1)
            correct_predictions += (predictions == targets).sum().item()
            all_predictions.extend(predictions.flatten().tolist())
            all_targets.extend(targets.flatten().tolist())
    
    # 计算总体损失和准确率
    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / len(test_loader.dataset)
    print("Loss:", avg_loss, "Accuracy:", accuracy)
    
    # 计算其他指标
    # print(all_targets, all_predictions)
    tn, fp, fn, tp = confusion_matrix(all_targets, all_predictions).ravel()
    print("Accuracy: ", (tp + tn) / (tp + tn + fp + fn))
    print('False positive rate(FPR): ', fp / (fp + tn))
    print('False negative rate(FNR): ', fn / (fn + tp))
    recall = tp / (tp + fn)
    print('Recall(TPR): ', recall)
    precision = tp / (tp + fp)
    print('Precision: ', precision)
    print('F1 score: ', (2 * precision * recall) / (precision + recall))


def CGE_test(model, test_loader, criterion, device, args, method='Fed_PLE', reduction='mean'):
    all_predictions = []
    all_targets = []
    with torch.no_grad():  # 在评估阶段，不需要计算梯度
        for graph, pattern, targets in test_loader:
            graph, pattern, targets = graph.to(device), pattern.to(device), targets.to(device)
            outputs = model(graph, pattern)
            total_loss = 0
            correct_predictions = 0
            one_hot_targets = F.one_hot(targets.long().flatten(), num_classes=2)
            one_hot_targets = one_hot_targets.float()
            loss = criterion(outputs, one_hot_targets)
            if reduction == 'none':
                loss = loss.mean()
            total_loss += loss.item()
            # predictions = torch.sigmoid(outputs).round()
            predictions = F.log_softmax(outputs, dim=-1)
            predictions = torch.argmax(predictions, dim=-1)
            correct_predictions += (predictions == targets).sum().item()
            all_predictions.extend(predictions.flatten().tolist())
            all_targets.extend(targets.flatten().tolist())
    
    # 计算总体损失和准确率
    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / len(test_loader.dataset)
    print("Loss:", avg_loss, "Accuracy:", accuracy)
    
    # 计算其他指标
    # print(all_targets, all_predictions)
    tn, fp, fn, tp = confusion_matrix(all_targets, all_predictions).ravel()
    result_dict = dict()
    result_dict['Accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    result_dict['False positive rate(FPR)'] = fp / (fp + tn)
    result_dict['False negative rate(FNR)'] = fn / (fn + tp)
    result_dict['Recall(TPR)'] = tp / (tp + fn)
    result_dict['Precision'] = tp / (tp + fp)
    result_dict['F1 score'] = (2 * result_dict['Precision'] * result_dict['Recall(TPR)']) / (result_dict['Precision'] + result_dict['Recall(TPR)'])

    # save results
    result_path = Path(os.path.realpath(__file__)).parents[0].joinpath(
        'merge_result',
        str(args.noise_rate),
        method,
    )
    Path.mkdir(result_path, parents=True, exist_ok=True)
    if args.noise_type == 'noise':
        result_file_path = result_path.joinpath(f'{args.vul}_result.json')
    elif args.noise_type == 'fn_noise':
        result_file_path = result_path.joinpath(f'fn_{args.vul}_result.json')
    elif args.noise_type == 'diff_noise':
        result_file_path = result_path.joinpath(f'diff_noise_{args.vul}_result.json')
    elif args.noise_type == 'pure':
        result_file_path = result_path.joinpath(f'pure_{args.vul}_result.json')

    if os.path.exists(result_file_path):
        with open(result_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if type(data) is dict:
                data = [data]
            data.append(result_dict)
        with open(result_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    else:
        data = [result_dict]
        file = open(str(result_file_path), "w")
        json.dump(data, file, ensure_ascii=False, indent=4)
        file.close()

    print("Accuracy: ", (tp + tn) / (tp + tn + fp + fn))
    print('False positive rate(FPR): ', fp / (fp + tn))
    print('False negative rate(FNR): ', fn / (fn + tp))
    recall = tp / (tp + fn)
    print('Recall(TPR): ', recall)
    precision = tp / (tp + fp)
    print('Precision: ', precision)
    print('F1 score: ', (2 * precision * recall) / (precision + recall))


def CBGRU_test(model, dataloader, criterion, args, method):
    all_predictions = []
    all_targets = []
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(args.device), x2.to(args.device), y.to(args.device)
            y = y.flatten().long()
            outputs = model(x1, x2)

            loss = criterion(outputs, y)
            total_loss += loss.item()
            softmax = nn.Softmax(dim=1)
            pred = torch.argmax(softmax(outputs), dim=-1)
            all_predictions.extend(pred.flatten().tolist())
            all_targets.extend(y.flatten().tolist())
            
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Averge Loss: {avg_loss}")
    tn, fp, fn, tp = confusion_matrix(all_targets, all_predictions).ravel()
    result_dict = dict()
    result_dict['Accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    result_dict['False positive rate(FPR)'] = fp / (fp + tn)
    result_dict['False negative rate(FNR)'] = fn / (fn + tp)
    result_dict['Recall(TPR)'] = tp / (tp + fn)
    result_dict['Precision'] = tp / (tp + fp)
    result_dict['F1 score'] = (2 * result_dict['Precision'] * result_dict['Recall(TPR)']) / (result_dict['Precision'] + result_dict['Recall(TPR)'])\
    
    if args.noise_type == 'pure':
        args.noise_rate =  0.

    result_path = Path(os.path.realpath(__file__)).parents[0].joinpath(
        'merge_result',
        str(args.noise_rate),
        f"{method}_{args.cbgru_net1}_{args.cbgru_net2}",
    )
    Path.mkdir(result_path, parents=True, exist_ok=True)
    if args.noise_type == 'noise':
        result_file_path = result_path.joinpath(f'{args.vul}_result.json')
    elif args.noise_type == 'fn_noise':
        result_file_path = result_path.joinpath(f'fn_{args.vul}_result.json')
    elif args.noise_type == 'pure':
         result_file_path = result_path.joinpath(f'pure_{args.vul}_result.json')
        
    if os.path.exists(result_file_path):
        with open(result_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if type(data) is dict:
                data = [data]
            data.append(result_dict)
        with open(result_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    else:
        data = [result_dict]
        file = open(str(result_file_path), "w")
        json.dump(data, file, ensure_ascii=False, indent=4)
        file.close()

    print("Accuracy: ", result_dict['Accuracy'])
    print('False positive rate(FPR): ', result_dict['False positive rate(FPR)'])
    print('False negative rate(FNR): ', result_dict['False negative rate(FNR)'])
    print('Recall(TPR): ', result_dict['Recall(TPR)'])
    print('Precision: ', result_dict['Precision'])
    print('F1 score: ', result_dict['F1 score'])
            
            





    
