import torch.nn as nn
import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def print_metadata(dataset, ks, dataset_name):
    print(f"About {dataset_name}")
    print(f"Data size: {len(dataset[5])}")

    for k in sorted(ks):
        print(f"------------{k}------------")
        print(f".   Size: {len(dataset[k])}")
        print(f".   x shape: {dataset[k][0].x.shape}")


def get_baseline_acc(dataset, bug_list, result_file):
    dataset_in_bug_list = [d for d in dataset[5] if d.bug_name in bug_list]
    num_total = len(dataset_in_bug_list)
    num_true = 0
    for data in dataset_in_bug_list:
        if data.y:
            num_true += 1
    baseline_acc = num_true / num_total
    print(f"Baseline accuracy: {baseline_acc}")

    with open(result_file, 'a+') as rf:
        rf.write(f'baseline accuracy: {baseline_acc:.4f}\n')


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p, num_layers):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2)
        ])
        self.conv_out = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout_p = dropout_p

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p = self.dropout_p, training=self.training)
        x = self.conv_out(x, edge_index, edge_weight)
        x = global_mean_pool(x, batch) 
        x = self.fc(x)
        return x
    



def train(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        if out.dim() == 2 and out.size(1) == 1:
            out = out.view(-1)
        loss = criterion(out, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calculate accuracy
        pred = (torch.sigmoid(out) >= 0.5).int()
        correct += int((pred == data.y).sum())
        total += len(data.y)

    accuracy = correct / total
    return total_loss / len(train_loader), accuracy

def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data).squeeze()
        pred = (torch.sigmoid(out) >= 0.5).int()
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)


def test_with_auc(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            if out.dim() == 2 and out.size(1) == 1:
                out = out.view(-1)

            preds = torch.sigmoid(out).cpu().numpy()
            
            labels = data.y.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    return fpr, tpr, auc

def evaluate_with_fixed_threshold_precision(model, loader, device):
    threshold=0.5
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            if out.dim() == 2 and out.size(1) == 1:
                out = out.view(-1)
                
            pred = (torch.sigmoid(out) >= threshold).int()

            if pred.ndim == 0:
                pred = torch.tensor(pred.item())
            labels = data.y.cpu().numpy()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels)

    precision = precision_score(all_labels, all_preds, zero_division=0)
    
    return precision

def evaluate_with_fixed_threshold_recall(model, loader, device):
    threshold = 0.5
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            if out.dim() == 2 and out.size(1) == 1:
                out = out.view(-1)
            pred = (torch.sigmoid(out) >= threshold).int()
            labels = data.y.cpu().numpy()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels)
    
    recall = recall_score(all_labels, all_preds, zero_division=0)
    
    return recall

def evaluate_with_npv(model, loader, device):
    threshold = 0.5
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            if out.dim() == 2 and out.size(1) == 1:
                out = out.view(-1)
            pred = (torch.sigmoid(out) >= threshold).int()
            labels = data.y.cpu().numpy()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels)
    
    unique_labels = np.unique(all_labels)
    unique_preds = np.unique(all_preds)
    all_classes = np.unique(np.concatenate([unique_labels, unique_preds]))

    if len(unique_labels) == 1:
        print(f"Warning: Only one class ({unique_labels[0]}) present in test labels")

        if unique_labels[0] == 0:
            tn = np.sum((all_labels == 0) & (all_preds == 0))
            fp = np.sum((all_labels == 0) & (all_preds == 1))
            fn = 0
            tp = 0
        else:
            tn = 0
            fp = 0
            fn = np.sum((all_labels == 1) & (all_preds == 0))
            tp = np.sum((all_labels == 1) & (all_preds == 1))
    
        npv = tn / (tn+fn) if (tn+fn) > 0 else 0.0
    
    else:
        # Confusion matrix에서 True Negative와 False Negative 값을 추출
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        
        # Negative Predictive Value (NPV) 계산
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # 나누는 값이 0이 아닐 경우에만 계산
    
    return npv

def evaluate_with_specificity(model, loader, device):
    threshold = 0.5
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            if out.dim() == 2 and out.size(1) == 1:
                out = out.view(-1)
            pred = (torch.sigmoid(out) >= threshold).int()
            labels = data.y.cpu().numpy()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels)
    
    unique_labels = np.unique(all_labels)
    unique_preds = np.unique(all_preds)
    
    if len(unique_labels) == 1:
        print(f"Warning: Only one class ({unique_labels[0]}) present in test labels")

        if unique_labels[0] == 0:
            tn = np.sum((all_labels == 0) & (all_preds == 0))
            fp = np.sum((all_labels == 0) & (all_preds == 1))
            fn = 0
            tp = 0
        else:
            tn = 0
            fp = 0
            fn = np.sum((all_labels == 1) & (all_preds == 0))
            tp = np.sum((all_labels == 1) & (all_preds == 1))
    
        specificity = tn / (tn+fp) if (tn+fp) > 0 else 0.0
    
    else:
        
        # Confusion matrix에서 True Negative와 False Positive 값을 추출
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        
        # Specificity (True Negative Rate) 계산
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # 나누는 값이 0이 아닐 경우에만 계산
    
    return specificity

def train_and_test_model(dataset, train_bug_names, test_bug_names, criterion, output_dim, lr, batch_size, hidden_dim, dropout_p, num_layer, num_epochs, ks, result_file, device, dataset_name, dir_dict):
    print(f"Training and testing with {dataset_name}")
    with open(result_file, "a+") as rf:
        rf.write(f"{dataset_name.split('_')[-1]}\n")
    
    get_baseline_acc(dataset, test_bug_names, result_file)

    for k in sorted(ks):
        with open(result_file, "a+") as rf:
            rf.write(f'k={k}\n')
        print(f"==================For {k}=======================")

        input_dim = dataset[k][0].x.shape[1]

        train_dataset = [d for d in dataset[k] if d.bug_name in train_bug_names]
        test_dataset = [d for d in dataset[k] if d.bug_name in test_bug_names]

        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

        # Print class distribution
        train_labels = [int(d.y.item()) for d in train_dataset]
        train_num_pos = sum(train_labels)
        train_num_neg = len(train_labels) - train_num_pos
        print(f"Class distribution in train dataset - Positive: {train_num_pos}, Negative: {train_num_neg}")

        test_labels = [int(d.y.item()) for d in test_dataset]
        test_num_pos = sum(test_labels)
        test_num_neg = len(test_labels) - test_num_pos
        print(f"Class distribution in test dataset - Positive: {test_num_pos}, Negative: {test_num_neg}")

        # Initialize model
        model = GCN(input_dim, hidden_dim, output_dim, dropout_p, num_layer).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)

        # Track metrics
        train_accs, test_accs = [], []
        precisions, recalls, npvs, specificities = [], [], [], []
        fprs, tprs, aucs_list = [], [], []

        # For saving best model
        best_test_acc = 0.0
        best_model_state = None
        best_epoch = 0

        # Model save path
        model_dir = dir_dict['trained_model_dir']
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, f"{k}k_best_model.pt")

        # Training_loop
        for epoch in range(num_epochs):
            loss, train_acc = train(model, optimizer, criterion, train_loader, device)
            test_acc = test(model, test_loader, device)
            fpr, tpr, auc_score = test_with_auc(model, test_loader, device)
            precision = evaluate_with_fixed_threshold_precision(model, test_loader, device)
            recall = evaluate_with_fixed_threshold_recall(model, test_loader, device)
            npv = evaluate_with_npv(model, test_loader, device)
            specificity = evaluate_with_specificity(model, test_loader, device)

            train_accs.append(train_acc)
            test_accs.append(test_acc)
            precisions.append(precision)
            recalls.append(recall)
            npvs.append(npv)
            specificities.append(specificity)
            fprs.append(fpr)
            tprs.append(tpr)
            aucs_list.append(auc_score)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
        
        # Save the best model
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'train_acc': train_accs[best_epoch],
            'test_acc': best_test_acc,
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'dropout_p': dropout_p,
            'num_layer': num_layer
        }, model_path)
        print(f"Best model saved to {model_path}")

        # Use metrics from best epoch
        best_train_acc = train_accs[best_epoch]
        best_test_acc = test_accs[best_epoch]
        best_fpr = fprs[best_epoch]
        best_tpr = tprs[best_epoch]
        best_auc = aucs_list[best_epoch]
        best_precision = precisions[best_epoch]
        best_recall = recalls[best_epoch]
        best_npv = npvs[best_epoch]
        best_specificity = specificities[best_epoch]

        # Plot accuracy graphs over epochs
        graph_dir = dir_dict['graph_dir']
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)

        acc_graph_path = os.path.join(graph_dir, f"{k}k_acc.png")
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs + 1), train_accs, label='Train Accuracy')
        plt.plot(range(1, num_epochs + 1), test_accs, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy per Epoch (k={k})')
        plt.legend()
        plt.grid(True)
        plt.savefig(acc_graph_path)
        plt.close()

        roc_auc_graph_path = os.path.join(graph_dir, f"{k}k_roc_auc.png")
        plt.figure(figsize=(4.5, 4))
        plt.plot(best_fpr, best_tpr, label=f'ROC Curve (AUC = {best_auc:.4f})', color='blue', linewidth=2)
        plt.plot([0, 1], [0, 1], linestyle='--', color='red', linewidth=1.5, label='Random Guess')
        plt.xlabel('False Positive Rate (FPR)', fontsize=14)
        plt.ylabel('True Positive Rate (TPR)', fontsize=14)
        plt.title(f'ROC Curve for GCN Model (F+A, k={k})', fontsize=16)
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.savefig(roc_auc_graph_path)
        plt.close()
        
        print(f"Epoch = {best_epoch+1}")
        print(f"Best train accuracy: {best_train_acc:.4f}")
        print(f"Best test accuracy: {best_test_acc:.4f}")
        print(f"Best AUC: {best_auc:.4f}")
        print(f"Best precision: {best_precision:.4f}")
        print(f"Best recall: {best_recall:.4f}")
        print(f"Best npv: {best_npv:.4f}")
        print(f"Best specificity: {best_specificity:.4f}")
        print('-------------------------------------------------------------------')
        with open(result_file, "a+") as rf:
            rf.write(f"Epoch = {best_epoch+1}\n")
            rf.write(f"Best train accuracy: {best_train_acc:.4f}\n")
            rf.write(f"Best test accuracy: {best_test_acc:.4f}\n")
            rf.write(f"Best AUC: {best_auc:.4f}\n")
            rf.write(f"Best precision: {best_precision:.4f}\n")
            rf.write(f"Best recall: {best_recall:.4f}\n")
            rf.write(f"Best npv: {best_npv:.4f}\n")
            rf.write(f"Best specificity: {best_specificity:.4f}\n")


def main(dir_dict, hidden_dim, num_layer, balanced):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_dir = dir_dict['data_dir']
    result_dir = dir_dict['result_dir']

    ks = [int(k) for k in os.listdir(dataset_dir)]
    # ks = [100]

    smallest_k = min(ks)
    reference_dataset = torch.load(os.path.join(dataset_dir, str(smallest_k), "gcn_dataset.pt"), weights_only = False)

    pos_data = [d for d in reference_dataset if int(d.y.item()) == 1]
    neg_data = [d for d in reference_dataset if int(d.y.item()) == 0]

    num_pos = len(pos_data)
    num_neg = len(neg_data)
    
    if balanced:
        if num_neg < num_pos:
            raise ValueError(f"The number of data with label 1 is bigger than the data with label 0")

        sampled_neg_data = random.sample(neg_data, num_pos)
        reference_dataset = pos_data + sampled_neg_data
        random.shuffle(reference_dataset)
        num_neg = num_pos

    all_bug_names = [data.bug_name for data in reference_dataset]

    train_bug_names, test_bug_names = train_test_split(all_bug_names, test_size=0.2, random_state=42, shuffle=True)

    print(f"Total bugs: {len(all_bug_names)}")
    print(f"Positive: {num_pos}")
    print(f"Negative: {num_neg}")

    if balanced:
        with open('../balanced_train_bugs.txt', 'w') as f:
            f.write('\n'.join(train_bug_names))
        with open('../balanced_test_bugs.txt', 'w') as f:
            f.write('\n'.join(test_bug_names))
    else:
        with open('../train_bugs.txt', 'w') as f:
            f.write('\n'.join(train_bug_names))
        with open('../test_bugs.txt', 'w') as f:
            f.write('\n'.join(test_bug_names))

    # if balanced:
    #     with open('../balanced_train_bugs.txt', 'r') as f:
    #         train_bug_names = f.read().splitlines()
    #     with open('../balanced_test_bugs.txt', 'r') as f:
    #         test_bug_names = f.read().splitlines()
    # else:
    #     with open('../train_bugs.txt', 'r') as f:
    #         train_bug_names = f.read().splitlines()
    #     with open('../test_bugs.txt', 'r') as f:
    #         test_bug_names = f.read().splitlines()

    dataset_FA = {}
    for k in ks:
        dataset_for_k = torch.load(os.path.join(dataset_dir, str(k), "gcn_dataset.pt"), weights_only = False)
        dataset_FA[k] = dataset_for_k

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_file = os.path.join(result_dir, f"gcn_result.txt")

    if os.path.exists(result_file):
        os.remove(result_file)
        print(f"{result_file} is removed")

    # print_metadata(dataset_FA, ks, "dataset_FA")

    criterion = nn.BCEWithLogitsLoss()
    output_dim = 1
    lr = 0.001
    batch_size = 32
    hidden_dim = hidden_dim
    dropout_p = 0.8
    num_layer = num_layer
    num_epochs = 100

    train_and_test_model(dataset_FA, train_bug_names, test_bug_names, criterion, output_dim, lr, batch_size, hidden_dim, dropout_p, num_layer, num_epochs, ks, result_file, device, "dataset_FA", dir_dict)

def get_dir_dict(dataset_dir, hidden_dim, num_layer, balanced):
    dir_dict = dict()

    parsed_dir = dataset_dir.split('/')
    if balanced:
        result_dir = os.path.join('../results', '/'.join(parsed_dir[2:]), f"{hidden_dim}h_{num_layer}l/no_cv/balanced")
        trained_model_dir = os.path.join('../trained_model', '/'.join(parsed_dir[2:]), f"{hidden_dim}h_{num_layer}l/no_cv/balanced")
        graph_dir = os.path.join('../graphs', '/'.join(parsed_dir[2:]), f"{hidden_dim}h_{num_layer}l/no_cv/balanced")
    else:
        result_dir = os.path.join('../results', '/'.join(parsed_dir[2:]), f"{hidden_dim}h_{num_layer}l/no_cv")
        trained_model_dir = os.path.join('../trained_model', '/'.join(parsed_dir[2:]), f"{hidden_dim}h_{num_layer}l/no_cv")
        graph_dir = os.path.join('../graphs', '/'.join(parsed_dir[2:]), f"{hidden_dim}h_{num_layer}l/no_cv")

    dir_dict = {
        'data_dir': dataset_dir,
        'result_dir': result_dir,
        'trained_model_dir': trained_model_dir,
        'graph_dir': graph_dir
    }

    return dir_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', default='../data/clustering/fasttext/word_vector/100/processed_response/0.99_0.99/plausible_patch/label_criteria_1')
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('-l', '--num_layer', default=3, type=int)
    parser.add_argument('-b', '--balanced', default=0, type=int)
    args = parser.parse_args()

    balanced = True if args.balanced == 1 else False

    dir_dict = get_dir_dict(args.dataset_dir, args.hidden_dim, args.num_layer, balanced)
    main(dir_dict, args.hidden_dim, args.num_layer, balanced)
    