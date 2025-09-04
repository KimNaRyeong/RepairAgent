# agnn_search_for_binary.py
import os
import copy
import math
import random
import argparse
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------
# Minimal Search Space (reduced for speed)
# ---------------------------
class SearchSpace:
    def __init__(self):
        # smaller candidate lists for speed (you can expand later)
        self.hidden_dims = [16, 32, 64, 128]
        self.attention_functions = ['GCN', 'GAT']
        self.attention_heads = [1, 4, 8]
        self.aggregate_functions = ['add', 'mean']  # maps to PyG aggr names
        self.combine_functions = ['identity', 'mlp']
        self.activation_functions = ['ReLU', 'LeakyReLU']
        self.dropout_p = [0.3, 0.5, 0.8]
        self.learning_rate = [0.001, 0.005, 0.01]
        self.batch_size = [32, 64]

        self.action_classes = [
            self.hidden_dims,
            self.attention_functions,
            self.attention_heads,
            self.aggregate_functions,
            self.combine_functions,
            self.activation_functions,
            self.dropout_p,
            self.learning_rate,
            self.batch_size
        ]

        self.class_names = [
            'hidden_dim',
            'attention_func',
            'attention_heads',
            'aggregate_func',
            'combine_func',
            'activation',
            'dropout_p',
            'learning_rate',
            'batch_size'
        ]

    def sample_random_arch(self, num_layers=3):
        # sample per-class a sequence length=num_layers (same action chosen for all layers)
        # (simpler: we choose same action across layers for non-layerwise params like lr, bs)
        arch = []
        # For layer-wise classes (hidden_dim, attention_func, attention_heads, aggregate, combine, activation, dropout):
        # create per-layer lists
        for cls_idx, cls_list in enumerate(self.action_classes):
            if cls_idx <= 6:  # layer-wise
                # sample a list of length num_layers
                seq = [random.choice(cls_list) for _ in range(num_layers)]
                arch.append(seq)
            else:
                # global hyperparameters (learning_rate, batch_size) -> single value
                arch.append(random.choice(cls_list))
        # arch is list: [hidden_dims_seq, attn_seq, heads_seq, aggr_seq, combine_seq, activation_seq, dropout, lr, batch_size]
        return arch

    def arch_to_string(self, arch, num_layers=3):
        lines = []
        for l in range(num_layers):
            items = []
            for i, name in enumerate(self.class_names):
                val = arch[i][l] if i <= 6 else arch[i]
                items.append(f"{name}={val}")
            lines.append("Layer %d: %s" % (l+1, ", ".join(items)))
        # add global lr, batchsize if present
        lines.append(f"global_lr={arch[7]}, batch_size={arch[8]}")
        return "\n".join(lines)

# ---------------------------
# Graph Layer that supports GCN/GAT + combine (identity/mlp)
# ---------------------------
class GraphConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, attention_type='GCN', heads=1, combine='identity', activation='ReLU', aggr='add', dropout=0.3):
        super().__init__()
        self.attention_type = attention_type
        self.combine = combine
        self.activation = getattr(nn, activation)() if activation in ['ReLU', 'LeakyReLU'] else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.aggr = aggr

        if attention_type == 'GCN':
            # GCNConv maps in_dim -> out_dim
            self.conv = GCNConv(in_dim, out_dim, cached=False)
            self.out_dim = out_dim
        elif attention_type == 'GAT':
            # GATConv outputs out_dim per head; we'll set out_dim as final out per head, then flatten heads
            # we want final node dim = out_dim * heads
            self.conv = GATConv(in_dim, out_dim, heads=heads, concat=True, dropout=dropout)
            self.out_dim = out_dim * heads
        else:
            # fallback to GCN
            self.conv = GCNConv(in_dim, out_dim)
            self.out_dim = out_dim

        if combine == 'mlp':
            # MLP combining (concat input and conv output)
            self.combine_mlp = nn.Sequential(
                nn.Linear(in_dim + self.out_dim, max(self.out_dim, 128)),
                nn.ReLU(),
                nn.Linear(max(self.out_dim, 128), self.out_dim)
            )

    def forward(self, x, edge_index, edge_attr=None):
        h = self.conv(x, edge_index)
        # if GCNConv returns shape (N, out_dim), if GAT returns (N, out_dim * heads)
        if self.combine == 'identity':
            out = h
        else:
            out = torch.cat([x, h], dim=-1)
            out = self.combine_mlp(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out

# ---------------------------
# Complete GNN built from GraphConvBlock
# ---------------------------
class GNNModel(nn.Module):
    def __init__(self, input_dim, arch, num_layers=3, output_dim=1):
        super().__init__()
        # arch structure:
        # arch[0] = hidden_dims_seq (len=num_layers)
        # arch[1] = attention_funcs_seq
        # arch[2] = attention_heads_seq
        # arch[3] = aggregate_seq (unused directly because conv implements)
        # arch[4] = combine_seq
        # arch[5] = activation_seq
        # arch[6] = dropout_p (can be list or scalar)
        # arch[7] = lr (global)
        # arch[8] = batch_size (global)
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        in_dim = input_dim
        # dropout can be per-layer or scalar
        dropout_seq = arch[6] if isinstance(arch[6], list) else [arch[6]]*num_layers

        for l in range(num_layers):
            hidden = arch[0][l]
            attn = arch[1][l]
            heads = arch[2][l]
            combine = arch[4][l]
            act = arch[5][l]
            drop = dropout_seq[l] if l < len(dropout_seq) else dropout_seq[-1]

            block = GraphConvBlock(in_dim, hidden, attention_type=attn, heads=heads,
                                   combine=combine, activation=act, dropout=drop)
            self.layers.append(block)
            in_dim = block.out_dim  # next input dim

        self.pool = global_mean_pool
        # final classifier to scalar logit for binary classification (BCEWithLogitsLoss)
        self.fc = nn.Linear(in_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.pool(x, batch)  # (batch_size, in_dim)
        out = self.fc(x).view(-1)  # flatten to (batch,)
        return out  # logits

# ---------------------------
# RNN Controller (per action-class)
# ---------------------------
class RNNController(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_cardinality, num_layers=3, device='cpu'):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.action_cardinality = action_cardinality
        self.embedding = nn.Embedding(action_cardinality + 1, input_dim)  # +1 pad
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, action_cardinality)

        # init
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, subarch_indices_tensor):
        # subarch_indices_tensor: (1, seq_len) or (batch, seq_len) but we use batch=1 here
        # embed and run lstm
        # print(subarch_indices_tensor)
        emb = self.embedding(subarch_indices_tensor)  # (1, seq, emb)
        # print(emb.shape)
        out, (h_n, c_n) = self.lstm(emb)
        # print(h_n.shape)
        # print(c_n.shape)
        # we'll decode step by step using hidden states
        hs = h_n.squeeze(0)  # (batch=1, hidden)
        actions = []
        probs = []
        log_probs = []
        h = h_n
        c = c_n
        # iterative decode
        for t in range(self.num_layers):
            logits = self.classifier(h[0])  # (batch, card)
            prob = F.softmax(logits, dim=-1)
            logp = F.log_softmax(logits, dim=-1)
            # sample
            action = torch.multinomial(prob, num_samples=1)  # (batch,1)
            actions.append(action.item())
            probs.append(prob.detach().cpu())
            log_probs.append(logp.gather(1, action).squeeze(1))
            # feed action back
            a_emb = self.embedding(action)  # (batch, emb)
            # print(a_emb.shape)
            a_emb = a_emb.unsqueeze(1)  # (batch,1,emb)
            # print("a_emb shape:", a_emb.shape)
            if a_emb.dim() == 4:
                a_emb = a_emb.squeeze(2)
            _, (h, c) = self.lstm(a_emb, (h, c))
        return actions, probs, log_probs

# ---------------------------
# Trainer/Evaluator: uses 3-fold CV and returns mean validation accuracy (and test if needed)
# ---------------------------
class Evaluator:
    def __init__(self, device='cpu'):
        self.device = device

    def evaluate_architecture(self, arch, dataset_list, input_dim, num_layers=3,
                              folds=3, train_epochs=40, verbose=False):
        """
        dataset_list: list of torch_geometric.data.Data graphs (for k=20)
        Returns mean validation accuracy across folds.
        """
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        val_accs = []

        # unpack hyperparams
        lr = arch[7]
        batch_size = arch[8]

        for train_idx, val_idx in kf.split(dataset_list):
            train_dataset = [dataset_list[i] for i in train_idx]
            val_dataset = [dataset_list[i] for i in val_idx]
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = GNNModel(input_dim, arch, num_layers=num_layers, output_dim=1).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            criterion = nn.BCEWithLogitsLoss()

            # training (small number of epochs for search speed)
            for epoch in range(train_epochs):
                model.train()
                total_loss = 0.0
                for batch in train_loader:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    out = model(batch)  # logits (batch_size,)
                    loss = criterion(out, batch.y.float())
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            # validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    out = model(batch)
                    preds = (torch.sigmoid(out) >= 0.5).long()
                    correct += (preds == batch.y).sum().item()
                    total += batch.y.size(0)
            val_acc = correct / total if total > 0 else 0.0
            val_accs.append(val_acc)

        mean_val = float(np.mean(val_accs))
        if verbose:
            print(f"Evaluated arch -> mean_val_acc: {mean_val:.4f}")
        return mean_val

# ---------------------------
# AGNN-style search controller
# ---------------------------
class AGNNSearch:
    def __init__(self, search_space: SearchSpace, dataset_list: List, input_dim: int, device='cpu',
                 num_layers=3, folds=3, controller_hidden=64, num_random_starts=2, num_iterations=30,
                 train_epochs=40):
        self.ss = search_space
        self.dataset_list = dataset_list
        self.device = device
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.folds = folds
        self.train_epochs = train_epochs

        # Create controllers: one per action class (we treat per-class cardinality)
        self.controllers = []
        self.controller_opts = []
        for cls in self.ss.action_classes:
            card = len(cls) if not isinstance(cls, list) else len(cls)
            ctrl = RNNController(input_dim=32, hidden_dim=controller_hidden, action_cardinality=card,
                                 num_layers=num_layers, device=device).to(self.device)
            opt = torch.optim.Adam(ctrl.parameters(), lr=3.5e-4)
            self.controllers.append(ctrl)
            self.controller_opts.append(opt)

        # evaluator
        self.evaluator = Evaluator(device=device)

        # bookkeeping
        self.best_arch = None
        self.best_score = -1.0
        self.arch_history = []
        self.score_history = []

        # search params
        self.num_random_starts = num_random_starts
        self.num_iterations = num_iterations

    def calc_entropy(self, probs_list):
        # probs_list = list of torch tensors (prob vectors) per step
        ent = 0.0
        for p in probs_list:
            p = p.squeeze(0)
            p = p + 1e-8
            ent += -torch.sum(p * torch.log(p)).item()
        return ent

    def select_classes_by_entropy(self, entropies, num_select=1):
        arr = np.array(entropies)
        if arr.sum() == 0:
            # fallback: choose first classes
            return list(range(min(num_select, len(arr))))
        probs = np.exp(arr / arr.max())  # crude softmax-like
        probs = probs / probs.sum()
        chosen = np.random.choice(len(arr), size=min(num_select, len(arr)), replace=False, p=probs)
        return chosen.tolist()

    def modify_arch(self, base_arch, selected_class_indices, new_actions_per_class):
        # base_arch formatting as in SearchSpace.sample_random_arch()
        arch = copy.deepcopy(base_arch)
        for cls_idx in selected_class_indices:
            # if cls is layer-wise (idx <=6), new_actions_per_class[cls_idx] should be length=num_layers list of action indices
            if cls_idx <= 6:
                indices = new_actions_per_class[cls_idx]  # list of indices len=num_layers
                for l in range(self.num_layers):
                    arch[cls_idx][l] = self.ss.action_classes[cls_idx][indices[l]]
            else:
                # global param single index
                idx = new_actions_per_class[cls_idx][0]
                arch[cls_idx] = self.ss.action_classes[cls_idx][idx]
        return arch

    def run_search(self):
        # 1) random starts
        print("== AGNN SEARCH START ==")
        for s in range(self.num_random_starts):
            arch = self.ss.sample_random_arch(self.num_layers)
            # ensure lr and batch_size exist as scalars
            if not isinstance(arch[7], float):
                arch[7] = float(arch[7])
            if not isinstance(arch[8], int):
                arch[8] = int(arch[8])
            score = self.evaluator.evaluate_architecture(arch, self.dataset_list, self.input_dim,
                                                         num_layers=self.num_layers, folds=self.folds,
                                                         train_epochs=self.train_epochs)
            print(f"Random start {s+1}: val_acc={score:.4f}")
            if score > self.best_score:
                self.best_score = score
                self.best_arch = arch
                print("  -> new best (init)")

        print("Initial best score:", self.best_score)
        # 2) main iterations
        for it in range(self.num_iterations):
            # for each controller/class, build subarchitecture (remove actions of that class) representation
            entropies = []
            sampled_actions = {}
            sampled_log_probs = {}

            for cls_idx, ctrl in enumerate(self.controllers):
                # Build subarch indices: flatten other classes into a sequence as input
                # For simplicity, we'll form a fixed-length vector by mapping each other class's current values to indices
                sub_indices = []
                for j in range(len(self.ss.action_classes)):
                    if j == cls_idx:
                        continue
                    # if layer-wise, append indices for each layer; else append single index
                    if j <= 6:
                        for l in range(self.num_layers):
                            # print(self.best_arch[j])
                            val = self.best_arch[j][l]
                            try:
                                idx = self.ss.action_classes[j].index(val)
                            except ValueError:
                                print("Value Error here")
                                idx = 0
                            sub_indices.append(idx)
                    else:
                        val = self.best_arch[j]
                        try:
                            idx = self.ss.action_classes[j].index(val)
                        except ValueError:
                            idx = 0
                        sub_indices.append(idx)
                if len(sub_indices) == 0:
                    sub_indices = [0]  # fallback
                sub_tensor = torch.tensor([sub_indices], dtype=torch.long, device=self.device)
                

                actions_idx_list, probs_list, log_probs_list = ctrl(sub_tensor)
                # actions_idx_list: list len=num_layers of ints
                # store
                sampled_actions[cls_idx] = actions_idx_list
                sampled_log_probs[cls_idx] = log_probs_list
                ent = self.calc_entropy(probs_list)
                entropies.append(ent)

            # choose which classes to modify
            selected = self.select_classes_by_entropy(entropies, num_select=1)
            if len(selected) == 0:
                selected = [0]
            # prepare new_actions dict mapped to actual indices (for layer-wise and for globals)
            new_actions = {}
            for cls_idx in selected:
                # sampled_actions[cls_idx] has length num_layers (each is index into that class card)
                new_actions[cls_idx] = sampled_actions[cls_idx]

            # turn selected actions into arch values and evaluate offspring
            offspring = self.modify_arch(self.best_arch, selected, new_actions)
            val_acc = self.evaluator.evaluate_architecture(offspring, self.dataset_list, self.input_dim,
                                                          num_layers=self.num_layers, folds=self.folds,
                                                          train_epochs=self.train_epochs)
            reward = val_acc - self.best_score
            print(f"[Iter {it+1}/{self.num_iterations}] selected_classes={[self.ss.class_names[i] for i in selected]} val_acc={val_acc:.4f} reward={reward:.4f}")

            # REINFORCE update for controllers of selected classes
            for cls_idx in selected:
                logps = sampled_log_probs[cls_idx]  # list of tensors per step
                # sum log probs across steps
                total_logp = sum([lp.sum() for lp in logps])
                # loss = -advantage * logp
                advantage = reward
                loss = - (advantage * total_logp)
                opt = self.controller_opts[cls_idx]
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.controllers[cls_idx].parameters(), 1.0)
                opt.step()

            # if improved, replace best
            if reward > 0:
                print("  >>> Offspring better: updating best architecture")
                self.best_arch = offspring
                self.best_score = val_acc

            self.arch_history.append(copy.deepcopy(offspring))
            self.score_history.append(val_acc)

        print("Search completed. Best val_acc:", self.best_score)
        return self.best_arch, self.best_score

# ---------------------------
# Entrypoint that loads user's dataset directory and runs search on k=20
# ---------------------------
def main(dataset_dir, device_str='cpu', num_iterations=20, random_starts=2, train_epochs=40):
    set_seed(42)
    device = torch.device(device_str)
    print("Device:", device)

    # load dataset structure same as original script: ks folders each containing gcn_dataset.pt
    ks = [int(k) for k in os.listdir(dataset_dir) if k.isdigit()]
    if 20 not in ks:
        raise ValueError("k=20 dataset not found in dataset_dir. Available ks: %s" % sorted(ks))

    # load dataset k=20 (expected file name: gcn_dataset.pt in folder dataset_dir/20)
    path20 = os.path.join(dataset_dir, '20', 'gcn_dataset.pt')
    if not os.path.exists(path20):
        raise FileNotFoundError(f"Expected {path20} but not found.")
    dataset_k20 = torch.load(path20, weights_only=False)

    graphs = dataset_k20

    print(f"Loaded k=20 graphs: {len(graphs)} graphs")

    # search space and searcher
    ss = SearchSpace()
    # sample an initial arch to shape correctly (all layer-wise parts should be lists)
    init_arch = ss.sample_random_arch(num_layers=3)
    # normalize global hyperparams to correct types
    if not isinstance(init_arch[7], float): # Learning rate
        init_arch[7] = float(init_arch[7])
    if not isinstance(init_arch[8], int): # Batch size
        init_arch[8] = int(init_arch[8])

    agnn = AGNNSearch(search_space=ss, dataset_list=graphs, input_dim=graphs[0].x.shape[1],
                      device=device, num_layers=3, folds=3,
                      controller_hidden=64, num_random_starts=random_starts, num_iterations=num_iterations,
                      train_epochs=train_epochs)
    agnn.best_arch = init_arch
    agnn.best_score = agnn.evaluator.evaluate_architecture(init_arch, graphs, graphs[0].x.shape[1],
                                                          num_layers=3, folds=3, train_epochs=train_epochs)
    print("Starting search with initial val_acc:", agnn.best_score)
    best_arch, best_score = agnn.run_search()

    print("=== FINAL BEST ARCH ===")
    print(ss.arch_to_string(best_arch, num_layers=3))
    print("Best validation acc:", best_score)

    # Optionally: train final model on whole dataset using best hyperparams (or use k-fold to get test)
    return best_arch, best_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str, default='/home/kimnal0/RepairAgent/Atropos/data/clustering/fasttext/word_vector/100/processed_response/0.96_0.97/plausible_patch/label_criteria_1', help='dataset directory containing numeric k folders (e.g., .../100/... ). Expect folder 20/gcn_dataset.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_iterations', type=int, default=20)
    parser.add_argument('--random_starts', type=int, default=2)
    parser.add_argument('--train_epochs', type=int, default=40, help='epochs per fold during search (reduce for speed)')
    args = parser.parse_args()

    main(args.dataset_dir, device_str=args.device, num_iterations=args.num_iterations,
         random_starts=args.random_starts, train_epochs=args.train_epochs)
