import os
import copy
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
        self.hidden_dims = [16, 32, 64, 128]
        self.attention_functions = ['GCN', 'GAT']
        self.attention_heads = [1, 4, 8]
        self.aggregate_functions = ['add', 'mean']
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

        self._build_token_map()

    def _build_token_map(self):
        self.token_offset = {}
        offset = 0
        for j, cls in enumerate(self.action_classes):
            self.token_offset[j] = offset
            offset += len(cls)
        self.vocab_size = offset
        self.value_to_index = []
        for cls in self.action_classes:
            mapping = {val: idx for idx, val in enumerate(cls)}
            self.value_to_index.append(mapping)

    def value_to_token(self, class_idx, option_value):
        idx_map = self.value_to_index[class_idx]
        if option_value not in idx_map:
            try:
                opt_idx = int(option_value)
            except Exception:
                opt_idx = 0
            if opt_idx < 0 or opt_idx >= len(self.action_classes[class_idx]):
                opt_idx = 0
        else:
            opt_idx = idx_map[option_value]
        return self.token_offset[class_idx] + opt_idx

    def sample_random_arch(self, num_layers=3):
        arch = []
        for cls_idx, cls_list in enumerate(self.action_classes):
            if cls_idx <= 6:
                seq = [random.choice(cls_list) for _ in range(num_layers)]
                arch.append(seq)
            else:
                arch.append(random.choice(cls_list))
        return arch

    def arch_to_string(self, arch, num_layers=3):
        lines = []
        for l in range(num_layers):
            items = []
            for i, name in enumerate(self.class_names):
                val = arch[i][l] if i <= 6 else arch[i]
                items.append(f"{name}={val}")
            lines.append("Layer %d: %s" % (l+1, ", ".join(items)))
        lines.append(f"global_lr={arch[7]}, batch_size={arch[8]}")
        return "\n".join(lines)

# ---------------------------
# GraphConv Block
# ---------------------------
class GraphConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, attention_type='GCN', heads=1,
                 combine='identity', activation='ReLU', aggr='add', dropout=0.3):
        super().__init__()
        self.combine = combine
        self.activation = getattr(nn, activation)() if activation in ['ReLU','LeakyReLU'] else nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if attention_type == 'GCN':
            self.conv = GCNConv(in_dim, out_dim, cached=False)
            self.out_dim = out_dim
        elif attention_type == 'GAT':
            self.conv = GATConv(in_dim, out_dim, heads=heads, concat=True, dropout=dropout)
            self.out_dim = out_dim * heads
        else:
            self.conv = GCNConv(in_dim, out_dim)
            self.out_dim = out_dim

        if combine == 'mlp':
            self.combine_mlp = nn.Sequential(
                nn.Linear(in_dim + self.out_dim, max(self.out_dim, 128)),
                nn.ReLU(),
                nn.Linear(max(self.out_dim, 128), self.out_dim)
            )

    def forward(self, x, edge_index, edge_attr=None):
        h = self.conv(x, edge_index)
        out = h if self.combine == 'identity' else self.combine_mlp(torch.cat([x,h],dim=-1))
        out = self.activation(out)
        out = self.dropout(out)
        return out

# ---------------------------
# GNN Model
# ---------------------------
class GNNModel(nn.Module):
    def __init__(self, input_dim, arch, num_layers=3, output_dim=1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        in_dim = input_dim
        dropout_seq = arch[6] if isinstance(arch[6], list) else [arch[6]]*num_layers

        for l in range(num_layers):
            block = GraphConvBlock(in_dim,
                                   arch[0][l],
                                   attention_type=arch[1][l],
                                   heads=arch[2][l],
                                   combine=arch[4][l],
                                   activation=arch[5][l],
                                   dropout=dropout_seq[l])
            self.layers.append(block)
            in_dim = block.out_dim

        self.pool = global_mean_pool
        self.fc = nn.Linear(in_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.pool(x, batch)
        return self.fc(x).view(-1)

# ---------------------------
# RNN Controller
# ---------------------------
class RNNController(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_cardinality,
                 emb_vocab_size, num_layers=3, device='cpu'):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(emb_vocab_size, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, action_cardinality)

        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, subarch_indices_tensor):
        emb = self.embedding(subarch_indices_tensor.to(self.embedding.weight.device))
        _, (h, c) = self.lstm(emb)
        actions, probs, log_probs = [], [], []

        for t in range(self.num_layers):
            logits = self.classifier(h[0])
            prob = F.softmax(logits, dim=-1)
            logp = F.log_softmax(logits, dim=-1)
            action = torch.multinomial(prob, 1)
            actions.append(int(action.item()))
            probs.append(prob.detach().cpu())
            log_probs.append(logp.gather(1, action).squeeze(1))
            a_emb = self.embedding(action.view(-1)).unsqueeze(1)
            _, (h, c) = self.lstm(a_emb, (h, c))
        return actions, probs, log_probs

# ---------------------------
# Evaluator
# ---------------------------
class Evaluator:
    def __init__(self, device='cpu'):
        self.device = device

    def evaluate_architecture(self, arch, dataset_list, input_dim, num_layers=3,
                              folds=3, train_epochs=40, verbose=False):
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        val_accs = []
        lr, batch_size = arch[7], arch[8]

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_list)):
            train_loader = DataLoader([dataset_list[i] for i in train_idx], batch_size=batch_size, shuffle=True)
            val_loader = DataLoader([dataset_list[i] for i in val_idx], batch_size=batch_size, shuffle=False)

            model = GNNModel(input_dim, arch, num_layers=num_layers, output_dim=1).to(self.device)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            crit = nn.BCEWithLogitsLoss()

            best_val_acc = 0.0

            for epoch in range(train_epochs):
                # ---- train ----
                model.train()
                for batch in train_loader:
                    batch = batch.to(self.device)
                    opt.zero_grad()
                    loss = crit(model(batch), batch.y.float())
                    loss.backward()
                    opt.step()

                # ---- validate ----
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        preds = (torch.sigmoid(model(batch)) >= 0.5).long()
                        correct += (preds == batch.y).sum().item()
                        total += batch.y.size(0)
                val_acc = correct / total if total > 0 else 0.0
                best_val_acc = max(best_val_acc, val_acc)

                if verbose and (epoch+1) % max(1, train_epochs//5) == 0:
                    print(f"[Fold {fold+1}] Epoch {epoch+1}/{train_epochs} val_acc={val_acc:.4f} (best={best_val_acc:.4f})")

            val_accs.append(best_val_acc)

        mean_val = float(np.mean(val_accs))
        if verbose:
            print(f"Evaluated arch -> mean_best_val_acc: {mean_val:.4f}")
        return mean_val


# ---------------------------
# Build subarch token sequence
# ---------------------------
def build_subarch_token_sequence(ss: SearchSpace, arch, ctrl_idx:int, num_layers:int):
    num_layerwise = sum(1 for j in range(len(ss.action_classes)) if j<=6)
    fixed_len = (num_layerwise-1)*num_layers
    seq = []
    for j in range(len(ss.action_classes)):
        if j>6 or j==ctrl_idx: continue
        for l in range(num_layers):
            seq.append(ss.value_to_token(j, arch[j][l]))
    if len(seq)<fixed_len: seq.extend([0]*(fixed_len-len(seq)))
    elif len(seq)>fixed_len: seq=seq[:fixed_len]
    return torch.tensor([seq],dtype=torch.long)

# ---------------------------
# AGNN Search
# ---------------------------
class AGNNSearch:
    def __init__(self, ss:SearchSpace, dataset_list:List, input_dim:int, device='cpu',
                 num_layers=3, folds=3, controller_hidden=64,
                 num_random_starts=2, num_iterations=30, train_epochs=40):
        self.ss=ss; self.dataset_list=dataset_list; self.device=device
        self.input_dim=input_dim; self.num_layers=num_layers; self.folds=folds
        self.train_epochs=train_epochs; self.best_arch=None; self.best_score=-1
        self.arch_history=[]; self.score_history=[]
        self.num_random_starts=num_random_starts; self.num_iterations=num_iterations
        self.controllers=[]; self.controller_opts=[]
        for cls in ss.action_classes:
            ctrl=RNNController(32, controller_hidden, len(cls), ss.vocab_size, num_layers, device).to(device)
            opt=torch.optim.Adam(ctrl.parameters(),lr=3.5e-4)
            self.controllers.append(ctrl); self.controller_opts.append(opt)
        self.evaluator=Evaluator(device)

    def calc_entropy(self, probs_list):
        return sum([-(p+1e-8).mul((p+1e-8).log()).sum().item() for p in probs_list])

    def select_classes_by_entropy(self, ents, k=1):
        arr=np.array(ents)
        if arr.sum()==0: return [0]
        probs=np.exp(arr/arr.max()); probs/=probs.sum()
        return np.random.choice(len(arr),size=k,replace=False,p=probs).tolist()

    def _would_change(self, arch, cls_idx, actions):
        """선택된 actions가 arch를 실제로 바꾸는지 검사(no-op 회피용)."""
        if cls_idx <= 6:  # layer-wise
            for l in range(self.num_layers):
                cur_val = arch[cls_idx][l]
                cur_idx = self.ss.action_classes[cls_idx].index(cur_val)
                if int(actions[l]) != int(cur_idx):
                    return True
            return False
        else:  # global
            cur_val = arch[cls_idx]
            cur_idx = self.ss.action_classes[cls_idx].index(cur_val)
            return int(actions[0]) != int(cur_idx)

    def _resample_non_noop(self, cls_idx, ctrl, sub_tensor, max_tries=10, verbose=True):
        """
        동일 액션(no-op)이 나올 경우 최대 max_tries까지 재샘플.
        성공 시 (acts, logps, True), 실패 시 마지막 샘플과 함께 (acts, logps, False) 반환.
        """
        last_acts, last_logps = None, None
        for t in range(max_tries):
            acts, probs, logps = ctrl(sub_tensor)
            last_acts, last_logps = acts, logps
            if self._would_change(self.best_arch, cls_idx, acts):
                return acts, logps, True
            if verbose and t == 0:
                print(f"  [resample] {self.ss.class_names[cls_idx]} produced no-op. Resampling...")
        return last_acts, last_logps, False

    def modify_arch(self, base_arch, selected_class_indices, new_actions_per_class):
        arch = copy.deepcopy(base_arch)
        for cls_idx in selected_class_indices:
            if cls_idx <= 6:
                indices = new_actions_per_class[cls_idx]
                for l in range(self.num_layers):
                    arch[cls_idx][l] = self.ss.action_classes[cls_idx][indices[l]]
            else:
                idx = new_actions_per_class[cls_idx][0]
                arch[cls_idx] = self.ss.action_classes[cls_idx][idx]
        return arch

    def run_search(self, verbose=True, temperature=1.0, num_select=1, max_resamples=10):
        print("== AGNN SEARCH START ==")

        # 1) random starts
        for s in range(self.num_random_starts):
            arch = self.ss.sample_random_arch(self.num_layers)
            if not isinstance(arch[7], float):
                arch[7] = float(arch[7])
            if not isinstance(arch[8], int):
                arch[8] = int(arch[8])
            score = self.evaluator.evaluate_architecture(
                arch, self.dataset_list, self.input_dim,
                num_layers=self.num_layers, folds=self.folds,
                train_epochs=self.train_epochs, verbose=True
            )
            print(f"Random start {s+1}: val_acc={score:.4f}")
            if score > self.best_score:
                self.best_score = score
                self.best_arch = arch
                print("  -> new best (init)")

        print("Initial best score:", self.best_score)

        # 2) main search loop
        for it in range(self.num_iterations):
            entropies, sub_tensors = [], {}
            sampled_actions, sampled_logps = {}, {}

            # (a) sample once for entropy
            for ci, ctrl in enumerate(self.controllers):
                sub_tensor = build_subarch_token_sequence(
                    self.ss, self.best_arch, ci, self.num_layers
                ).to(self.device)
                sub_tensors[ci] = sub_tensor
                acts, probs, logps = ctrl(sub_tensor)
                sampled_actions[ci] = acts
                sampled_logps[ci] = logps
                ent = self.calc_entropy(probs)
                entropies.append(ent)
                if verbose:
                    print(f"  Controller {self.ss.class_names[ci]}: acts={acts}, entropy={ent:.4f}")

            # (b) entropy -> class probs
            ents = np.array(entropies, dtype=np.float64)
            if np.allclose(ents, 0):
                class_probs = np.ones_like(ents) / len(ents)
            else:
                class_probs = np.exp(ents / max(1e-8, float(temperature)))
                class_probs = class_probs / class_probs.sum()

            # (c) sample classes with no-op avoidance
            selectable = np.arange(len(self.controllers)).tolist()
            selected, new_actions = [], {}

            for _ in range(num_select):
                if not selectable:
                    break
                local_probs = class_probs[selectable]
                local_probs = local_probs / local_probs.sum()
                idx = int(np.random.choice(selectable, p=local_probs))
                ctrl = self.controllers[idx]

                acts, logps, ok = self._resample_non_noop(
                    idx, ctrl, sub_tensors[idx], max_tries=max_resamples, verbose=verbose
                )
                if ok:
                    selected.append(idx)
                    sampled_actions[idx] = acts
                    sampled_logps[idx] = logps
                    new_actions[idx] = acts
                    selectable.remove(idx)
                else:
                    if verbose:
                        print(f"  [skip] {self.ss.class_names[idx]} identical after {max_resamples} resamples.")
                    selectable.remove(idx)

            # (d) if none selected, force mutate one
            if not selected:
                if len(entropies) > 0:
                    try:
                        idx = int(np.random.choice(len(entropies), p=class_probs))
                    except Exception:
                        idx = int(np.argmax(entropies))
                else:
                    idx = 0
                card = len(self.ss.action_classes[idx])
                acts = sampled_actions[idx]
                if idx <= 6:
                    l = np.random.randint(self.num_layers)
                    cur_val = self.best_arch[idx][l]
                    cur_idx = self.ss.action_classes[idx].index(cur_val)
                    alt = (cur_idx + np.random.randint(1, card)) % card
                    acts[l] = int(alt)
                else:
                    cur_val = self.best_arch[idx]
                    cur_idx = self.ss.action_classes[idx].index(cur_val)
                    alt = (cur_idx + np.random.randint(1, card)) % card
                    acts = [int(alt)]
                selected = [idx]
                new_actions[idx] = acts
                if verbose:
                    print(f"  [force-change] Mutated one action in {self.ss.class_names[idx]} to avoid no-op. acts={acts}")

            # (e) apply & eval
            if verbose:
                print("  Selected classes:", [self.ss.class_names[i] for i in selected])
                print("  Original arch:")
                print(self.ss.arch_to_string(self.best_arch, self.num_layers))

            offspring = self.modify_arch(self.best_arch, selected, new_actions)

            if verbose:
                print("  Offspring arch:")
                print(self.ss.arch_to_string(offspring, self.num_layers))

            val_acc = self.evaluator.evaluate_architecture(
                offspring, self.dataset_list, self.input_dim,
                num_layers=self.num_layers, folds=self.folds,
                train_epochs=self.train_epochs, verbose=False
            )
            reward = val_acc - self.best_score
            print(f"[Iter {it+1}/{self.num_iterations}] val_acc={val_acc:.4f} reward={reward:.4f}")

            # (f) REINFORCE update
            for ci in selected:
                loss = -reward * sum([lp.sum() for lp in sampled_logps[ci]])
                opt = self.controller_opts[ci]
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.controllers[ci].parameters(), 1.0)
                opt.step()

            # (g) best update
            if reward > 0:
                print("  >>> Offspring better: updating best architecture")
                self.best_arch, self.best_score = offspring, val_acc

            self.arch_history.append(copy.deepcopy(offspring))
            self.score_history.append(val_acc)

        print("Search completed. Best val_acc:", self.best_score)
        return self.best_arch, self.best_score

# ---------------------------
# Main
# ---------------------------
def main(dataset_dir, device_str='cpu', num_iterations=20, random_starts=2, train_epochs=40, num_layers=3):
    set_seed(42); device=torch.device(device_str)
    print("Device:", device)
    print("num_layers:", num_layers)

    
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
    
    ss=SearchSpace()
    init_arch=ss.sample_random_arch(num_layers)
    init_arch[7]=float(init_arch[7]); init_arch[8]=int(init_arch[8])

    agnn=AGNNSearch(ss, graphs, graphs[0].x.shape[1],
                    device=device, num_layers=num_layers, folds=3,
                    controller_hidden=64, num_random_starts=random_starts,
                    num_iterations=num_iterations, train_epochs=train_epochs)

    agnn.best_arch=init_arch
    agnn.best_score=agnn.evaluator.evaluate_architecture(
        init_arch, graphs, graphs[0].x.shape[1],
        num_layers=num_layers, folds=3, train_epochs=train_epochs, verbose=True
    )
    print("Starting search with initial val_acc:", agnn.best_score)

    best_arch,best_score=agnn.run_search()
    print("=== FINAL BEST ARCH ===")
    print(ss.arch_to_string(best_arch, num_layers))
    print("Best val_acc:",best_score)
    return best_arch,best_score

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-d','--dataset_dir',type=str,default='/home/kimnal0/RepairAgent/Atropos/data/clustering/fasttext/word_vector/100/processed_response/0.96_0.97/plausible_patch/label_criteria_1')
    parser.add_argument('--device',type=str,default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_iterations',type=int,default=20)
    parser.add_argument('--random_starts',type=int,default=2)
    parser.add_argument('--train_epochs',type=int,default=40)
    parser.add_argument('--num_layers',type=int,default=3, help='GNN layer count for search')
    args=parser.parse_args()
    main(args.dataset_dir, args.device, args.num_iterations, args.random_starts, args.train_epochs, args.num_layers)
