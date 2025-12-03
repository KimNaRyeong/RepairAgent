import torch
import os
import argparse
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader


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
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv_out(x, edge_index, edge_weight)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


def load_model(model_path, device):
    """Load the trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract model parameters
    input_dim = checkpoint['input_dim']
    hidden_dim = checkpoint['hidden_dim']
    output_dim = checkpoint['output_dim']
    dropout_p = checkpoint['dropout_p']
    num_layer = checkpoint['num_layer']

    # Initialize model
    model = GCN(input_dim, hidden_dim, output_dim, dropout_p, num_layer).to(device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded from {model_path}")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Train accuracy: {checkpoint['train_acc']:.4f}")
    print(f"Test accuracy: {checkpoint['test_acc']:.4f}")

    return model


def predict(model, loader, device):
    """Run inference and return predictions"""
    model.eval()
    predictions = []
    bug_names = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            if out.dim() == 2 and out.size(1) == 1:
                out = out.view(-1)

            # Apply sigmoid and threshold at 0.5 for binary prediction
            pred = (torch.sigmoid(out) >= 0.5).int()

            predictions.extend(pred.cpu().numpy())
            bug_names.extend([data.bug_name for _ in range(len(pred))])

    return bug_names, predictions


def main(model_path, test_bug_list_file, dataset_dir, output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the trained model
    model = load_model(model_path, device)

    # 2. Load test bug list
    print(f"\nLoading test bug list from {test_bug_list_file}")
    with open(test_bug_list_file, 'r') as f:
        test_bug_names = f.read().splitlines()
    print(f"Number of test bugs: {len(test_bug_names)}")

    # 3. Load dataset
    # Extract k value from model path (e.g., "20k_best_model.pt" -> 20)
    model_filename = os.path.basename(model_path)
    k = int(model_filename.split('k_')[0])

    dataset_path = os.path.join(dataset_dir, str(k), "gcn_dataset.pt")
    print(f"\nLoading dataset from {dataset_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    full_dataset = torch.load(dataset_path, weights_only=False)

    # Filter dataset for test bugs only
    test_dataset = [d for d in full_dataset if d.bug_name in test_bug_names]
    print(f"Number of test data points: {len(test_dataset)}")

    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 4. Run predictions
    print("\nRunning predictions...")
    bug_names, predictions = predict(model, test_loader, device)

    # 5. Save results
    print(f"\nSaving results to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("bug_name,prediction\n")
        for bug_name, pred in zip(bug_names, predictions):
            f.write(f"{bug_name},{pred}\n")

    print(f"Results saved successfully!")

    # Print summary statistics
    num_positive = sum(predictions)
    num_negative = len(predictions) - num_positive
    print(f"\nPrediction Summary:")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Predicted as 1 (positive): {num_positive}")
    print(f"  Predicted as 0 (negative): {num_negative}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test GCN model on test dataset')
    parser.add_argument(
        '-m', '--model_path',
        default='/home/kimnal0/RepairAgent/Atropos/trained_model/clustering/fasttext/word_vector/100/processed_response/0.99_0.99/plausible_patch/label_criteria_1/64h_3l/no_cv/balanced/20k_best_model.pt',
        help='Path to the trained model checkpoint'
    )
    parser.add_argument(
        '-b', '--bug_list',
        default='/home/kimnal0/RepairAgent/Atropos/bug_list/balanced_test_bugs_label_criteria_1.txt',
        help='Path to test bug list file'
    )
    parser.add_argument(
        '-d', '--dataset_dir',
        default='/home/kimnal0/RepairAgent/Atropos/data/clustering/fasttext/word_vector/100/processed_response/0.99_0.99/plausible_patch/label_criteria_1',
        help='Directory containing the dataset (should have subdirectories for different k values)'
    )
    parser.add_argument(
        '-o', '--output',
        default='/home/kimnal0/RepairAgent/Atropos/test_results/label_criteria_1/balanced/20k_predictions.txt',
        help='Output file path for predictions'
    )

    args = parser.parse_args()

    main(args.model_path, args.bug_list, args.dataset_dir, args.output)