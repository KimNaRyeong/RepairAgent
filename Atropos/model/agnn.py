import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.loader import DataLoader
import numpy as np
import random
from collections import defaultdict
import copy
import argparse
import os
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Complete Search Space Definition with Attention
SEARCH_SPACE = {
    'num_layers': [2, 3, 4, 5],
    'hidden_dim': [16, 32, 64, 128, 256],
    'activation': ['relu', 'tanh', 'sigmoid', 'elu', 'leaky_relu'],
    'attention_type': ['gcn', 'gat', 'sage_mean', 'sage_max'],
    'attention_heads': [1, 2, 4, 8],
    'aggregate_func': ['mean', 'max', 'add'],
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'dropout_p': [0.0, 0.2, 0.5, 0.8],
    'batch_size': [16, 32, 64, 128]
}

class ActionEncoder:
    """Encode/decode actions to/from integers for RNN processing"""
    def __init__(self, search_space):
        self.search_space = search_space
        self.action_to_id = {}
        self.id_to_action = {}
        self.class_vocab_sizes = {}
        
        # Create vocabulary for each action class
        for action_class, values in search_space.items():
            vocab = {}
            for i, value in enumerate(values):
                vocab[value] = i
            self.action_to_id[action_class] = vocab
            self.id_to_action[action_class] = {i: v for v, i in vocab.items()}
            self.class_vocab_sizes[action_class] = len(values)
    
    def encode_architecture(self, architecture):
        """Convert architecture dict to integer sequences"""
        encoded = {}
        for action_class, value in architecture.items():
            encoded[action_class] = self.action_to_id[action_class][value]
        return encoded
    
    def decode_architecture(self, encoded_arch):
        """Convert integer sequences back to architecture dict"""
        decoded = {}
        for action_class, action_id in encoded_arch.items():
            decoded[action_class] = self.id_to_action[action_class][action_id]
        return decoded
    
    def create_subarchitecture(self, architecture, remove_class):
        """Create subarchitecture by removing one action class (Paper Section 4.2.1)"""
        subarch = []
        for action_class, value in architecture.items():
            if action_class != remove_class:
                subarch.append(self.action_to_id[action_class][value])
        return torch.tensor(subarch, dtype=torch.long).unsqueeze(0)  # Add batch dimension

class RNNEncoder(nn.Module):
    """RNN encoder for each action class as described in the paper (Section 4.2.1)"""
    def __init__(self, vocab_size, hidden_size=128, embedding_dim=50):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        
    def forward(self, input_seq):
        """Forward pass through RNN encoder"""
        if input_seq.size(1) == 0:  # Empty sequence
            batch_size = input_seq.size(0)
            hidden = torch.zeros(1, batch_size, self.hidden_size, device=input_seq.device)
            cell = torch.zeros(1, batch_size, self.hidden_size, device=input_seq.device)
            return [], (hidden, cell)
        
        embedded = self.embedding(input_seq)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Get probabilities for each step
        outputs = []
        for i in range(lstm_out.size(1)):
            logits = self.output_layer(lstm_out[:, i, :])
            probs = F.softmax(logits, dim=-1)
            outputs.append(probs)
        
        return outputs, (hidden, cell)
    
    def sample_action(self, input_seq, temperature=1.0):
        """Sample action from the RNN encoder with temperature scaling"""
        with torch.no_grad():
            if input_seq.size(1) == 0:  # Empty sequence, random sample
                action = torch.randint(0, self.vocab_size, (1,)).item()
                probs = torch.ones(self.vocab_size) / self.vocab_size
                return action, probs
            
            # Use final hidden state to predict next action
            embedded = self.embedding(input_seq)
            lstm_out, _ = self.lstm(embedded)
            final_hidden = lstm_out[:, -1, :]  # Use last hidden state
            
            logits = self.output_layer(final_hidden) / temperature
            probs = F.softmax(logits, dim=-1).squeeze()
            
            action = torch.multinomial(probs, 1).item()
            return action, probs

class GNNArchitecture(nn.Module):
    """Dynamic GNN architecture with attention mechanisms"""
    def __init__(self, input_dim, output_dim, config):
        super(GNNArchitecture, self).__init__()
        self.config = config
        self.num_layers = config['num_layers']
        self.hidden_dim = config['hidden_dim']
        self.activation_name = config['activation']
        self.dropout_p = config['dropout_p']
        self.attention_type = config['attention_type']
        self.attention_heads = config['attention_heads']
        self.aggregate_func = config['aggregate_func']
        
        # Build layers based on attention type
        self.convs = nn.ModuleList()
        
        # Determine actual hidden dimension for multi-head attention
        if self.attention_type == 'gat':
            head_dim = max(1, self.hidden_dim // self.attention_heads)
            actual_hidden_dim = head_dim * self.attention_heads
        else:
            actual_hidden_dim = self.hidden_dim
        
        # Build layer architecture
        dims = [input_dim] + [actual_hidden_dim] * (self.num_layers - 1) + [actual_hidden_dim]
        
        for i in range(self.num_layers):
            layer = self._create_conv_layer(dims[i], dims[i+1])
            self.convs.append(layer)
        
        # Final classifier
        self.classifier = nn.Linear(actual_hidden_dim, output_dim)
        
        # Activation function
        self.activation = self._get_activation(self.activation_name)
    
    def _create_conv_layer(self, in_dim, out_dim):
        """Create appropriate conv layer based on attention type"""
        if self.attention_type == 'gat':
            return GATConv(
                in_channels=in_dim,
                out_channels=out_dim // self.attention_heads,
                heads=self.attention_heads,
                concat=True,
                dropout=self.dropout_p,
                add_self_loops=True
            )
        elif self.attention_type == 'sage_mean':
            return SAGEConv(
                in_channels=in_dim,
                out_channels=out_dim,
                aggr='mean'
            )
        elif self.attention_type == 'sage_max':
            return SAGEConv(
                in_channels=in_dim,
                out_channels=out_dim,
                aggr='max'
            )
        else:  # gcn
            return GCNConv(
                in_channels=in_dim,
                out_channels=out_dim,
                add_self_loops=True,
                normalize=True
            )
    
    def _get_activation(self, name):
        activations = {
            'relu': F.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'elu': F.elu,
            'leaky_relu': F.leaky_relu
        }
        return activations.get(name, F.relu)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = getattr(data, 'edge_attr', None)
        
        # Forward through conv layers
        for i, conv in enumerate(self.convs):
            if self.attention_type in ['sage_mean', 'sage_max']:
                x = conv(x, edge_index)
            elif self.attention_type == 'gat':
                x = conv(x, edge_index)
            else:  # gcn
                x = conv(x, edge_index, edge_weight)
            
            if i < len(self.convs) - 1:  # Don't apply activation to last conv layer
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        # Global pooling based on aggregate function
        if self.aggregate_func == 'mean':
            x = global_mean_pool(x, batch)
        elif self.aggregate_func == 'max':
            x = global_max_pool(x, batch)
        else:  # add
            x = global_add_pool(x, batch)
        
        # Final classification
        x = self.classifier(x)
        return x

class RCNASController:
    """Reinforced Conservative Neural Architecture Search Controller with RNN Encoders"""
    def __init__(self, search_space, device='cpu', hidden_size=128):
        self.search_space = search_space
        self.device = device
        self.best_architecture = None
        self.best_performance = -float('inf')
        self.search_history = []
        
        # Initialize action encoder
        self.action_encoder = ActionEncoder(search_space)
        
        # Initialize RNN encoders for each action class
        self.rnn_encoders = nn.ModuleDict()
        self.optimizers = {}
        
        for action_class in search_space.keys():
            vocab_size = self.action_encoder.class_vocab_sizes[action_class]
            encoder = RNNEncoder(vocab_size, hidden_size).to(device)
            self.rnn_encoders[action_class] = encoder
            
            # Initialize optimizer for each encoder (Paper: lr=3.5e-4)
            self.optimizers[action_class] = torch.optim.Adam(
                encoder.parameters(), 
                lr=3.5e-4
            )
        
        # Initialize baselines for variance reduction
        self.baselines = {cls: 0.0 for cls in search_space.keys()}
        self.baseline_counts = {cls: 0 for cls in search_space.keys()}
        
        # Initialize with random starting architecture
        self.best_architecture = self._sample_random_architecture()
        
    def _sample_random_architecture(self):
        """Sample a random architecture from search space"""
        architecture = {}
        for key, values in self.search_space.items():
            architecture[key] = random.choice(values)
        return architecture
    
    def _calculate_entropy(self, probabilities):
        """Calculate entropy for action selection (Paper Section 4.2.2)"""
        probabilities = torch.clamp(probabilities, min=1e-8)  # Avoid log(0)
        entropy = -torch.sum(probabilities * torch.log(probabilities))
        return entropy.item()
    
    def _get_action_class_entropies(self):
        """Calculate decision entropy for each action class using RNN encoders"""
        if self.best_architecture is None:
            return {cls: 1.0 for cls in self.search_space.keys()}
        
        entropies = {}
        
        for action_class in self.search_space.keys():
            # Create subarchitecture by removing the current action class
            subarch_seq = self.action_encoder.create_subarchitecture(
                self.best_architecture, action_class
            ).to(self.device)
            
            # Get probability distribution from RNN encoder
            encoder = self.rnn_encoders[action_class]
            encoder.eval()
            
            with torch.no_grad():
                # Sample action to get probability distribution
                _, probs = encoder.sample_action(subarch_seq)
                entropy = self._calculate_entropy(probs)
                entropies[action_class] = entropy
        
        return entropies
    
    def _select_action_class_to_modify(self):
        """Select which action class to modify based on RNN encoder uncertainty"""
        entropies = self._get_action_class_entropies()
        
        # Select action class with highest entropy (Paper Section 4.2.2)
        selected_class = max(entropies.keys(), key=lambda k: entropies[k])
        return selected_class
    
    def guided_architecture_modification(self):
        """Generate offspring architecture using RNN encoders (Paper Section 4.2)"""
        if self.best_architecture is None:
            return self._sample_random_architecture()
        
        # Conservative approach: start with best architecture
        offspring = copy.deepcopy(self.best_architecture)
        
        # Step 1: Select action class to modify based on entropy
        action_class = self._select_action_class_to_modify()
        
        # Step 2: Use RNN encoder to decide new action
        subarch_seq = self.action_encoder.create_subarchitecture(
            self.best_architecture, action_class
        ).to(self.device)
        
        encoder = self.rnn_encoders[action_class]
        encoder.eval()
        
        # Sample new action from RNN encoder (Paper uses temperature=5.0)
        new_action_id, _ = encoder.sample_action(subarch_seq, temperature=5.0)
        new_action_value = self.action_encoder.id_to_action[action_class][new_action_id]
        
        # Step 3: Architecture modification - replace only the selected action class
        offspring[action_class] = new_action_value
        
        return offspring
    
    def update_rnn_encoders(self, architecture, reward):
        """Update RNN encoders using REINFORCE (Paper Section 4.3)"""
        if self.best_architecture is None:
            return
        
        # Find which action class was modified
        modified_class = None
        for action_class in self.search_space.keys():
            if architecture[action_class] != self.best_architecture[action_class]:
                modified_class = action_class
                break
        
        if modified_class is None:
            return
        
        # Update only the RNN encoder of the modified class
        encoder = self.rnn_encoders[modified_class]
        optimizer = self.optimizers[modified_class]
        
        # Create subarchitecture sequence
        subarch_seq = self.action_encoder.create_subarchitecture(
            self.best_architecture, modified_class
        ).to(self.device)
        
        # Forward pass
        encoder.train()
        
        # Get the action that was taken
        action_taken = self.action_encoder.action_to_id[modified_class][architecture[modified_class]]
        
        if subarch_seq.size(1) > 0:
            # Get log probability of the action taken
            embedded = encoder.embedding(subarch_seq)
            lstm_out, _ = encoder.lstm(embedded)
            final_hidden = lstm_out[:, -1, :]
            logits = encoder.output_layer(final_hidden)
            log_probs = F.log_softmax(logits, dim=-1)
            log_prob = log_probs[0, action_taken]
        else:
            # Handle empty sequence case
            log_prob = torch.log(torch.tensor(1.0 / encoder.vocab_size, device=self.device))
        
        # Calculate policy gradient loss (REINFORCE with baseline)
        baseline = self.baselines[modified_class]
        advantage = reward - baseline
        loss = -log_prob * advantage
        
        # Update encoder
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update baseline using moving average
        alpha = 0.1
        self.baselines[modified_class] = baseline + alpha * (reward - baseline)
        self.baseline_counts[modified_class] += 1
    
    def update_best_architecture(self, architecture, performance):
        """Update best architecture if performance improved"""
        # Calculate reward as performance difference (Paper: Rc = Mo - Mb)
        reward = performance - self.best_performance
        
        # Update RNN encoders with the reward
        self.update_rnn_encoders(architecture, reward)
        
        # Update best architecture if improved
        improved = False
        if performance > self.best_performance:
            self.best_architecture = copy.deepcopy(architecture)
            self.best_performance = performance
            improved = True
        
        return improved

class AutoGNNTrainer:
    """Main trainer for Auto-GNN with complete RCNAS implementation"""
    def __init__(self, dataset, input_dim, output_dim, device='cpu', 
                 max_search_iterations=50, max_epochs_per_arch=30):
        self.dataset = dataset
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.max_search_iterations = max_search_iterations
        self.max_epochs_per_arch = max_epochs_per_arch
        
        # Initialize controller with RNN encoders
        self.controller = RCNASController(SEARCH_SPACE, device)
        
        # Search results
        self.search_results = []
        
    def evaluate_architecture(self, architecture, train_loader, val_loader):
        """Evaluate a single architecture"""
        try:
            # Create model
            model = GNNArchitecture(self.input_dim, self.output_dim, architecture).to(self.device)
            
            # Setup optimizer with architecture-specific learning rate
            optimizer = torch.optim.Adam(model.parameters(), lr=architecture['learning_rate'])
            criterion = nn.BCEWithLogitsLoss()
            
            # Training loop
            model.train()
            for epoch in range(self.max_epochs_per_arch):
                total_loss = 0
                for data in train_loader:
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    out = model(data)
                    if out.dim() == 2 and out.size(1) == 1:
                        out = out.view(-1)
                    loss = criterion(out, data.y.float())
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            
            # Validation evaluation
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(self.device)
                    out = model(data)
                    if out.dim() == 2 and out.size(1) == 1:
                        out = out.view(-1)
                    preds = torch.sigmoid(out).cpu().numpy()
                    labels = data.y.cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels)
            
            # Calculate AUC as performance metric
            if len(set(all_labels)) > 1:
                performance = roc_auc_score(all_labels, all_preds)
            else:
                performance = 0.5
                
            return performance, model
            
        except Exception as e:
            print(f"Error evaluating architecture {architecture}: {str(e)}")
            return 0.0, None
    
    def search(self, k_fold=3):
        """Main search loop implementing RCNAS algorithm"""
        print("Starting Auto-GNN Architecture Search with RCNAS...")
        print(f"Search space size: {np.prod([len(v) for v in SEARCH_SPACE.values()]):,} possible architectures")
        
        # Prepare data for k-fold cross validation
        kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
        
        for iteration in tqdm(range(self.max_search_iterations), desc="RCNAS Search"):
            # Generate offspring architecture
            if iteration == 0:
                # Start with random architecture
                architecture = self.controller._sample_random_architecture()
            else:
                # Use guided modification with RNN encoders
                architecture = self.controller.guided_architecture_modification()
            
            # Evaluate architecture using k-fold cross validation
            fold_performances = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):
                train_dataset = [self.dataset[i] for i in train_idx]
                val_dataset = [self.dataset[i] for i in val_idx]
                
                train_loader = DataLoader(train_dataset, 
                                        batch_size=architecture['batch_size'], 
                                        shuffle=True)
                val_loader = DataLoader(val_dataset, 
                                      batch_size=architecture['batch_size'], 
                                      shuffle=False)
                
                performance, _ = self.evaluate_architecture(architecture, train_loader, val_loader)
                fold_performances.append(performance)
            
            # Average performance across folds
            avg_performance = np.mean(fold_performances)
            
            # Update controller (includes RNN encoder updates via REINFORCE)
            improved = self.controller.update_best_architecture(architecture, avg_performance)
            
            # Record results
            result = {
                'iteration': iteration,
                'architecture': copy.deepcopy(architecture),
                'performance': avg_performance,
                'improved': improved
            }
            self.search_results.append(result)
            
            if iteration % 10 == 0 or improved:
                print(f"\nIteration {iteration}: Performance = {avg_performance:.4f} "
                      f"{'(NEW BEST!)' if improved else ''}")
                print(f"Architecture: {architecture}")
                if improved:
                    entropies = self.controller._get_action_class_entropies()
                    print(f"Action class entropies: {entropies}")
                print("-" * 80)
        
        return self.controller.best_architecture, self.controller.best_performance
    
    def get_search_summary(self):
        """Get summary of search results"""
        if not self.search_results:
            return "No search results available"
        
        best_result = max(self.search_results, key=lambda x: x['performance'])
        
        summary = f"""
=== Auto-GNN RCNAS Search Summary ===
Total iterations: {len(self.search_results)}
Best performance: {best_result['performance']:.4f}
Best architecture found at iteration: {best_result['iteration']}

Best Architecture Configuration:
"""
        for key, value in best_result['architecture'].items():
            summary += f"  {key}: {value}\n"
        
        # Performance progression
        performances = [r['performance'] for r in self.search_results]
        improvements = sum(1 for r in self.search_results if r['improved'])
        
        summary += f"\nSearch Statistics:\n"
        summary += f"  Total improvements: {improvements}\n"
        summary += f"  Final performance: {performances[-1]:.4f}\n"
        summary += f"  Performance std: {np.std(performances):.4f}\n"
        
        return summary

def main(dataset_dir):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    ks = [int(k) for k in os.listdir(dataset_dir) if k.isdigit()]
    k = ks[0]  # Use first available k
    
    dataset = torch.load(os.path.join(dataset_dir, str(k), "gcn_dataset.pt"), weights_only=False)
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Get input dimension from first sample
    input_dim = dataset[0].x.shape[1]
    output_dim = 1  # Binary classification
    
    # Initialize and run Auto-GNN search with RCNAS
    trainer = AutoGNNTrainer(
        dataset=dataset,
        input_dim=input_dim,
        output_dim=output_dim,
        device=device,
        max_search_iterations=100,
        max_epochs_per_arch=30
    )
    
    # Run RCNAS architecture search
    best_arch, best_perf = trainer.search(k_fold=3)
    
    # Print results
    print("\n" + "="*80)
    print(trainer.get_search_summary())
    
    # Save results
    results_dir = '../results/auto_gnn_search'
    os.makedirs(results_dir, exist_ok=True)
    
    torch.save({
        'best_architecture': best_arch,
        'best_performance': best_perf,
        'search_results': trainer.search_results,
        'search_space': SEARCH_SPACE
    }, os.path.join(results_dir, f'auto_gnn_results_k{k}.pt'))
    
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', 
                       default='../data/clustering/fasttext/sentence_vector/100/raw_response/0.9_0.9/label_criteria_5',
                       help='Dataset directory path')
    args = parser.parse_args()
    
    main(args.dataset_dir)