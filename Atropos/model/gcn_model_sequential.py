import torch.nn as nn
import os
import json
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, average_precision_score, accuracy_score
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
    
def data_load(data_dir):
    dataset = dict()
    
    ks = [int(k) for k in os.listdir(data_dir)]
    for k in ks:
        data_for_k = torch.load(os.path.join(data_dir, str(k), 'gcn_dataset.pth'), weights_only=False)

        dataset[k] = data_for_k["dataset"]
    
    print("Datasets loaded successfully")

    return dataset


def print_metadata(dataset, ks, dataset_name):
    print(f"About {dataset_name}")
    print(f"Data size: {len(dataset[5])}")

    for k in sorted(ks):
        print(f"------------{k}------------")
        print(f".   Size: {len(dataset[k])}")
        print(f".   x shape: {dataset[k][0].x.shape}")

def compute_baseline_accuracy(test_dataset):
    """Calculate baseline accuracy on test dataset (majority class baseline)"""
    all_labels = [data.y.item() if hasattr(data.y, 'item') else data.y for data in test_dataset]
    num_total = len(all_labels)
    num_positive = sum(all_labels)

    # Baseline: always predict the majority class
    baseline_acc = max(num_positive, num_total - num_positive) / num_total
    return baseline_acc

def compute_confidence_metrics(test_dataset):
    """
    Calculate AutoFL confidence-based metrics on test dataset.
    Returns: (auc, best_accuracy, best_threshold, threshold_accuracies_dict)
    """


    d4j_combined_result_file = '../AutoFL/combined_fl_results/d4j_gpt4o_results_R10_full.json'
    bip_combined_result_file = '../AutoFL/combined_fl_results/bip_gpt4o_results_R10_full.json'

    with open(d4j_combined_result_file, 'r') as f:
        d4j_combined_result = json.load(f)
    with open(bip_combined_result_file, 'r') as f:
        bip_combined_result = json.load(f)

    all_labels = []
    reversed_all_confidences = []
    all_confidences = []

    for data in test_dataset:
        bug_name = data.bug_name

        if bug_name in d4j_combined_result.get("buggy_methods", {}):
            combined_result = d4j_combined_result
        elif bug_name in bip_combined_result.get("buggy_methods", {}):
            combined_result = bip_combined_result

        all_labels.append(int(data.y.item()))

        confidence = combined_result["confidence"][bug_name]
        reversed_all_confidences.append(1 - confidence)
        all_confidences.append(confidence)

    auc = roc_auc_score(all_labels, reversed_all_confidences)
    aupr = average_precision_score(all_labels, reversed_all_confidences)
    fpr, tpr, _ = roc_curve(all_labels, reversed_all_confidences)
    idx = np.where(tpr >= 0.95)[0]
    fpr_at_95 = fpr[idx[0]] if len(idx) > 0 else 1.0

    thresholds = [i / 10 for i in range(11)]
    threshold_accs = {}

    for th in thresholds:
        all_preds = [1 if cs < th else 0 for cs in all_confidences]
        accuracy = accuracy_score(all_labels, all_preds)
        threshold_accs[th] = accuracy
        

    return auc, aupr, fpr_at_95, threshold_accs

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
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
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
    return total_loss / len(train_loader)

def evaluate_model(model, loader, device, threshold=0.5):
    """
    Comprehensive model evaluation function that computes all metrics in a single pass.

    Returns a dictionary containing:
    - accuracy: Classification accuracy
    - auc: Area Under ROC Curve
    - aupr: Area Under Precision-Recall Curve
    - fpr_at_95: False Positive Rate at 95% TPR
    - precision: Precision at threshold
    - recall: Recall (Sensitivity) at threshold
    - npv: Negative Predictive Value at threshold
    - specificity: Specificity (True Negative Rate) at threshold
    - fpr: False Positive Rate array for ROC curve
    - tpr: True Positive Rate array for ROC curve
    """
    model.eval()
    all_probs = []
    all_labels = []

    # Single pass through the data to collect predictions
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            if out.dim() == 2 and out.size(1) == 1:
                out = out.view(-1)

            probs = torch.sigmoid(out).cpu().numpy()
            labels = data.y.cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Binary predictions at threshold
    all_preds = (all_probs >= threshold).astype(int)

    # Accuracy
    accuracy = np.mean(all_preds == all_labels)

    # ROC AUC and curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = roc_auc_score(all_labels, all_probs)

    # AUPR
    aupr = average_precision_score(all_labels, all_probs)

    # FPR at 95% TPR
    idx = np.where(tpr >= 0.95)[0]
    fpr_at_95 = fpr[idx[0]] if len(idx) > 0 else 1.0

    # Confusion matrix based metrics
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'accuracy': accuracy,
        'auc': roc_auc,
        'aupr': aupr,
        'fpr_at_95': fpr_at_95,
        'precision': precision,
        'recall': recall,
        'npv': npv,
        'specificity': specificity,
        'fpr': fpr,
        'tpr': tpr
    }

def train_and_test_model(dataset, criterion, output_dim, K, kf, lr, batch_size, hidden_dim, dropout_p, num_layer, num_epochs, ks, result_file, device, dataset_name, dir_dict):
    print(f"Training and testing with {dataset_name}")
    dataset_type = dataset_name.split('_')[-1]
    with open(result_file, "a+") as rf:
        rf.write(f"{dataset_type}\n")
    
    test_bug_names_dict = {}
    test_bug_names_file = os.path.join(dir_dict['result'], 'test_bug_names.json')
    # ks = [2]
    for k in sorted(ks):
        with open(result_file, "a+") as rf:
            rf.write(f'k={k}\n')
        
        ckpt_base = os.path.join(dir_dict['trained_model'], dataset_type, str(k))

        print(f"==================For {k}=======================")
        
        input_dim = dataset[k][0].x.shape[1]
        splits = kf.split(dataset[k])

        test_baseline_accs = []
        val_confidence_aucs = []
        test_confidence_aucs, test_confidence_auprs, test_confidence_fpr_at_95 = [], [], []
        val_all_confidence_threshold_accs = []
        test_all_confidence_threshold_accs = []

        train_accs, val_accs = [0] * num_epochs, [0] * num_epochs
        train_aucs, val_aucs = [0] * num_epochs, [0] * num_epochs

        test_accs, test_aucs, test_precisions, test_recalls, test_npvs, test_specificities = [], [], [], [], [], []
        test_auprs, test_fpr_at_95s = [], []

        best_val_auc = 0
        best_val_epoch = 0
        best_model_state = None


        for fold, (train_idx, val_test_idx) in tqdm(enumerate(splits)):
            val_idx = val_test_idx[:len(val_test_idx) // 2]
            test_idx = val_test_idx[len(val_test_idx) // 2:]

            train_dataset = [dataset[k][i] for i in train_idx]
            val_dataset = [dataset[k][i] for i in val_idx]
            test_dataset = [dataset[k][i] for i in test_idx]

            if not os.path.exists(test_bug_names_file):
                test_bug_name = [data.bug_name for data in test_dataset]
                test_bug_names_dict[fold] = test_bug_name

            # Calculate baseline accuracy for test set
            test_baseline_acc = compute_baseline_accuracy(test_dataset)
            test_baseline_accs.append(test_baseline_acc)

            # Calculate confidence metrics for validation set
            # val_conf_auc, val_conf_aupr, val_conf_fpr_at_95, val_conf_th_accs = compute_confidence_metrics(val_dataset)
            # val_confidence_aucs.append(val_conf_auc)
            # val_all_confidence_threshold_accs.append(val_conf_th_accs)

            # Calculate confidence metrics for test set
            # test_conf_auc, test_conf_aupr, test_conf_fpr_at_95, test_conf_th_accs = compute_confidence_metrics(test_dataset)
            # test_confidence_aucs.append(test_conf_auc)
            # test_confidence_auprs.append(test_conf_aupr)
            # test_confidence_fpr_at_95.append(test_conf_fpr_at_95)
            # test_all_confidence_threshold_accs.append(test_conf_th_accs)
            

            train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
            test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

            model = GCN(input_dim, hidden_dim, output_dim, dropout_p, num_layer).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr = lr)


            for epoch in range(num_epochs):
                loss = train(model, optimizer, criterion, train_loader, device)

                # Evaluate on train set
                train_metrics = evaluate_model(model, train_loader, device)
                train_accs[epoch] += train_metrics['accuracy']
                train_aucs[epoch] += train_metrics['auc']

                # Evaluate on validation set
                val_metrics = evaluate_model(model, val_loader, device)
                val_accs[epoch] += val_metrics['accuracy']
                val_aucs[epoch] += val_metrics['auc']

                # Save best model based on validation AUC
                if val_metrics['auc'] > best_val_auc:
                    best_val_auc = val_metrics['auc']
                    best_val_epoch = epoch
                    best_model_state = {
                        'state_dict': model.state_dict().copy(),
                        'epoch': epoch,
                        'val_auc': val_metrics['auc']
                    }
            
            best_model = GCN(input_dim, hidden_dim, output_dim, dropout_p, num_layer).to(device)
            best_model.load_state_dict(best_model_state['state_dict'])

            ckpt_file = os.path.join(ckpt_base, f'fold_{str(fold)}', 'best_auc.pt')
            save_checkpoint(
                best_model,
                ckpt_file,
                meta = {
                    "metric": 'val_auc',
                    "value": float(best_val_auc),
                    'fold': int(fold),
                    "k": int(k),
                    "hparams":{
                        "input_dim": int(input_dim),
                        "hidden_dim": int(hidden_dim),
                        "output_dim": int(output_dim),
                        "dropout_p": float(dropout_p),
                        "num_layer": int(num_layer),
                        "lr": float(lr),
                        "batch_size": int(batch_size)
                    }
                }
            )
            print(f"\nBest model saved! (Fold {fold}, Epoch {best_val_epoch}, Val Auc: {best_val_auc:.4f})")

            # Evaluate on test set with best model
            test_metrics = evaluate_model(best_model, test_loader, device)

            test_accs.append(test_metrics['accuracy'])
            test_aucs.append(test_metrics['auc'])
            test_precisions.append(test_metrics['precision'])
            test_recalls.append(test_metrics['recall'])
            test_npvs.append(test_metrics['npv'])
            test_specificities.append(test_metrics['specificity'])
            test_auprs.append(test_metrics['aupr'])
            test_fpr_at_95s.append(test_metrics['fpr_at_95'])

        # confidence_result_file = os.path.join(dir_dict['result'], 'autofl_confidence_result.json')
        # if not os.path.exists(confidence_result_file):
            # all_confidence_threshold_accs = {
                # 'val': val_all_confidence_threshold_accs,
                # 'test': test_all_confidence_threshold_accs
            # }
            # with open(confidence_result_file, 'w') as f:
                # json.dump(all_confidence_threshold_accs, f, indent=4)

        # Calculate mean baseline and confidence metrics
        mean_baseline_acc = np.mean(test_baseline_accs)
        # mean_confidence_auc = np.mean(test_confidence_aucs)
        # mean_confidence_aupr = np.mean(test_confidence_auprs)
        # mean_confidence_fpr_at_95 = np.mean(test_confidence_fpr_at_95)

        # Find best threshold across all folds (the one with highest average accuracy) from validation set
        # thresholds = sorted(val_all_confidence_threshold_accs[0].keys())
        # val_avg_threshold_accs = {}
        # for th in thresholds:
            # val_avg_threshold_accs[th] = np.mean([fold_th_accs[th] for fold_th_accs in val_all_confidence_threshold_accs])

        # best_conf_threshold = max(val_avg_threshold_accs, key=val_avg_threshold_accs.get)
        # val_best_mean_conf_acc = val_avg_threshold_accs[best_conf_threshold]
        # test_mean_conf_acc = np.mean([fold_th_accs[best_conf_threshold] for fold_th_accs in test_all_confidence_threshold_accs])

        mean_test_acc = np.mean(test_accs)
        mean_test_auc = np.mean(test_aucs)
        mean_test_precision = np.mean(test_precisions)
        mean_test_recall = np.mean(test_recalls)
        mean_test_npv = np.mean(test_npvs)
        mean_test_specificity = np.mean(test_specificities)
        mean_test_aupr = np.mean(test_auprs)
        mean_test_fpr_at_95 = np.mean(test_fpr_at_95s)

        mean_train_accs = [acc / K for acc in train_accs]
        mean_train_aucs = [auc / K for auc in train_aucs]
        mean_val_accs = [acc / K for acc in val_accs]
        mean_val_aucs = [auc / K for auc in val_aucs]

        graph_dir = os.path.join(dir_dict['train_graph'], dataset_type)
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
        acc_graph_path = os.path.join(graph_dir, f"k_{k}_acc.png")
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs+1), mean_train_accs, label = 'Train Accuracy')
        plt.plot(range(1, num_epochs+1), mean_val_accs, label = 'Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy per Epoch for {dataset_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(acc_graph_path)
        plt.close()

        auc_graph_path = os.path.join(graph_dir, f"k_{k}_auc.png")
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs+1), mean_train_aucs, label = 'Train AUC')
        plt.plot(range(1, num_epochs+1), mean_val_aucs, label = 'Val AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title(f'AUC per Epoch for {dataset_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(auc_graph_path)
        plt.close()

        print(f"Best Epoch = {best_val_epoch}")
        print(f"Baseline Accuracy: {mean_baseline_acc:.4f}")
        # print(f"Confidence AUC: {mean_confidence_auc:.4f}")
        # print(f"Confidence AUPR: {mean_confidence_aupr:.4f}")
        # print(f"Confidence FPR@95: {mean_confidence_fpr_at_95:.4f}")
        # print(f"Confidence Accuracy: {test_mean_conf_acc:.4f}")
        # print(f"Confidence Accuracy (Best Threshold={best_conf_threshold}): {test_mean_conf_acc:.4f}")
        print(f"Test Accuracy: {mean_test_acc:.4f}")
        print(f"Test ROC-AUC: {mean_test_auc:.4f}")
        print(f"Test AUPR: {mean_test_aupr:.4f}")
        print(f"Test FPR@95: {mean_test_fpr_at_95:.4f}")
        print(f"Test Precision: {mean_test_precision:.4f}")
        print(f"Test Recall: {mean_test_recall:.4f}")
        print(f"Test NPV: {mean_test_npv:.4f}")
        print(f"Test Specificity: {mean_test_specificity:.4f}")
        print('-------------------------------------------------------------------')

        with open(result_file, "a+") as rf:
            rf.write(f"Best Epoch = {best_val_epoch}\n")
            rf.write(f"Baseline Accuracy (Test): {mean_baseline_acc:.4f}\n")
            # rf.write(f"Confidence AUC (Test): {mean_confidence_auc:.4f}\n")
            # rf.write(f"Confidence AUPR (Test): {mean_confidence_aupr:.4f}\n")
            # rf.write(f"Confidence FPR@95 (Test): {mean_confidence_fpr_at_95:.4f}\n")
            # rf.write(f"Confidence Accuracy (Best Threshold={best_conf_threshold}): {test_mean_conf_acc:.4f}\n")
            rf.write(f"Test Accuracy: {mean_test_acc:.4f}\n")
            rf.write(f"Test ROC-AUC: {mean_test_auc:.4f}\n")
            rf.write(f"Test AUPR: {mean_test_aupr:.4f}\n")
            rf.write(f"Test FPR@95: {mean_test_fpr_at_95:.4f}\n")
            rf.write(f"Test Precision: {mean_test_precision:.4f}\n")
            rf.write(f"Test Recall: {mean_test_recall:.4f}\n")
            rf.write(f"Test NPV: {mean_test_npv:.4f}\n")
            rf.write(f"Test Specificity: {mean_test_specificity:.4f}\n")


        if not os.path.exists(test_bug_names_file):
            with open(test_bug_names_file, 'w') as f:
                json.dump(test_bug_names_dict, f, indent=4)
    
def save_checkpoint(model, path, meta=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if meta:
        payload.update(meta)
    torch.save(payload, path)



def main(dir_dict):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = dir_dict['data']
    dataset = data_load(data_dir)

    # --- [디버깅 코드 시작] ---
    print("Checking for malformed data labels...")
    for k, data_list in dataset.items():
        for i, data in enumerate(data_list):
            # y가 스칼라(1개)가 아닌 경우를 찾음
            if data.y.numel() != 1:
                print(f"!!! Found malformed data at k={k}, index={i}")
                print(f"    Shape of y: {data.y.shape}")
                print(f"    Value of y: {data.y}")
                # 필요하다면 여기서 강제로 수정 (예: 첫 번째 값만 사용)
                # data.y = data.y[0].unsqueeze(0) 
    # --- [디버깅 코드 끝] ---

    result_dir = dir_dict['result']
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file = os.path.join(result_dir, 'result.txt')

    if os.path.exists(result_file):
        os.remove(result_file)
        print(f"{result_file} is removed")
    confidence_result_file = os.path.join(dir_dict["result"], 'autofl_confidence_result.json')
    if os.path.exists(confidence_result_file):
        os.remove(confidence_result_file)
        print(f"{confidence_result_file} is removed")

    ks = [int(k) for k in os.listdir(data_dir)]
    print_metadata(dataset, ks, "dataset")

    criterion = nn.BCEWithLogitsLoss()
    output_dim = 1
    K = 5 
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    lr = 0.001
    batch_size = 32
    hidden_dim = 32
    dropout_p = 0.8
    num_layer = 2
    num_epochs = 150

    train_and_test_model(dataset, criterion, output_dim, K, kf, lr, batch_size, hidden_dim, dropout_p, num_layer, num_epochs, ks, result_file, device, "dataset_FA", dir_dict)

def get_dir_dict(label_criteria, threshold, merge_threshold):    
    dir_dict = {
        'data': f'../data/sequential/nhot_normal/center_vector/{threshold}_{merge_threshold}/label_criteria_{label_criteria}',
        'result': f'../results/sequential/nhot_normal/center_vector/{threshold}_{merge_threshold}/label_criteria_{label_criteria}',
        'trained_model': f'../trained_model/sequential/nhot_normal/center_vector/{threshold}_{merge_threshold}/label_criteria_{label_criteria}',
        'train_graph': f'../train_graph/sequential/nhot_normal/center_vector/{threshold}_{merge_threshold}/label_criteria_{label_criteria}'
    }

    return dir_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label_criteria', default=1, type=int)
    parser.add_argument('-t', '--threshold', default=0.9, type=float)
    parser.add_argument('-m', '--merge_threshold', default=0.9, type=float)
    args = parser.parse_args()
    
    dir_dict = get_dir_dict(args.label_criteria, args.threshold, args.merge_threshold)

    main(dir_dict)