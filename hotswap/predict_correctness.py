"""
Script to predict correctness for k=8 test set tasks using trained GCN models.
This script loads the trained models for each fold and predicts whether tasks are correct or incorrect.
"""

import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath('..'))
from atropos.model.gcn_model_fasttext_sentence import GCN, set_seed

def load_model_for_fold(fold, k, device, model_dir):
    """Load the trained model for a specific fold and k value."""
    model_path = os.path.join(model_dir, str(k), f'fold_{fold}', 'best_auc.pt')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract hyperparameters
    hparams = checkpoint['hparams']
    input_dim = hparams['input_dim']
    hidden_dim = hparams['hidden_dim']
    output_dim = hparams['output_dim']
    dropout_p = hparams['dropout_p']
    num_layer = hparams['num_layer']

    # Create model and load weights
    model = GCN(input_dim, hidden_dim, output_dim, dropout_p, num_layer).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model

def predict_task(model, data, device, threshold=0.5):
    """Predict correctness for a single task."""
    with torch.no_grad():
        data = data.to(device)
        out = model(data)
        if out.dim() == 2 and out.size(1) == 1:
            out = out.view(-1)

        prob = torch.sigmoid(out).cpu().item()
        pred = 1 if prob >= threshold else 0

    return pred, prob

def main():
    set_seed(42)

    # Configuration
    k = 8
    label_criteria = 1
    embedding_size = 300
    threshold = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    base_path = '/home/kimnal0/auto-code-rover/atropos'
    RepairAgent/Atropos/trained_model/parallel/nhot_normal/center_vector/0.99_0.99/label_criteria_1/FA/20
    model_dir = f'{base_path}/trained_model/parallel/embedding/fasttext/nhot_normal/sentence_vector/{embedding_size}d/not_add/label_criteria_{label_criteria}'
    data_dir = f'{base_path}/data/parallel/embedding/fasttext/nhot_normal/sentence_vector/{embedding_size}d/not_add/label_criteria_{label_criteria}'
    result_dir = f'{base_path}/results/parallel/embedding/fasttext/nhot_normal/sentence_vector/{embedding_size}d/not_add/label_criteria_{label_criteria}'

    test_tasks_file = os.path.join(result_dir, 'test_bug_names2.json')
    dataset_file = os.path.join(data_dir, str(k), 'gcn_dataset.pth')

    # Load test task names
    print(f"Loading test tasks from {test_tasks_file}")
    with open(test_tasks_file, 'r') as f:
        test_tasks_dict = json.load(f)

    # Convert keys to integers
    test_tasks_dict = {int(k): v for k, v in test_tasks_dict.items()}

    # Load dataset
    print(f"Loading dataset from {dataset_file}")
    dataset_dict = torch.load(dataset_file, weights_only=False)
    dataset = dataset_dict['dataset']

    # Create task to data mapping
    task_to_data = {data.task: data for data in dataset}

    # Predictions storage
    predictions = {}

    # Process each fold
    print(f"\nPredicting correctness for k={k}")
    for fold in tqdm(range(5)):
        fold_str = str(fold)
        test_tasks = test_tasks_dict[fold]

        # Load model for this fold
        model = load_model_for_fold(fold, k, device, model_dir)

        predictions[fold_str] = {}

        # Predict for each test task in this fold
        for task_name in test_tasks:
            if task_name not in task_to_data:
                print(f"Warning: Task {task_name} not found in dataset")
                continue

            data = task_to_data[task_name]
            pred, prob = predict_task(model, data, device, threshold)

            # Get actual label
            actual_label = int(data.y.item())

            predictions[fold_str][task_name] = {
                'predicted_label': pred,
                'probability': float(prob),
                'actual_label': actual_label,
                'is_correct_prediction': (pred == actual_label)
            }

    # Save predictions
    output_dir = '/home/kimnal0/auto-code-rover/hotswap/predictions'
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'k{k}_predictions.json')
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=4)

    print(f"\nPredictions saved to {output_file}")

    # Print summary statistics
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)

    total_tasks = 0
    predicted_incorrect = 0
    actually_incorrect = 0
    correct_predictions = 0

    for fold, fold_preds in predictions.items():
        fold_predicted_incorrect = sum(1 for p in fold_preds.values() if p['predicted_label'] == 1)
        fold_actually_incorrect = sum(1 for p in fold_preds.values() if p['actual_label'] == 1)
        fold_correct = sum(1 for p in fold_preds.values() if p['is_correct_prediction'])

        total_tasks += len(fold_preds)
        predicted_incorrect += fold_predicted_incorrect
        actually_incorrect += fold_actually_incorrect
        correct_predictions += fold_correct

        print(f"\nFold {fold}:")
        print(f"  Total tasks: {len(fold_preds)}")
        print(f"  Predicted as incorrect (label=1): {fold_predicted_incorrect}")
        print(f"  Actually incorrect (label=1): {fold_actually_incorrect}")
        print(f"  Correct predictions: {fold_correct}/{len(fold_preds)} ({100*fold_correct/len(fold_preds):.2f}%)")

    print(f"\n{'='*60}")
    print(f"OVERALL (k={k}):")
    print(f"  Total tasks: {total_tasks}")
    print(f"  Predicted as incorrect: {predicted_incorrect}")
    print(f"  Actually incorrect: {actually_incorrect}")
    print(f"  Accuracy: {100*correct_predictions/total_tasks:.2f}%")
    print("="*60)

    # Save tasks that need re-execution (predicted as incorrect)
    tasks_to_rerun = {}
    for fold, fold_preds in predictions.items():
        tasks_to_rerun[fold] = [
            task for task, pred_info in fold_preds.items()
            if pred_info['predicted_label'] == 1
        ]

    rerun_file = os.path.join(output_dir, f'k{k}_tasks_to_rerun.json')
    with open(rerun_file, 'w') as f:
        json.dump(tasks_to_rerun, f, indent=4)

    print(f"\nTasks to re-run saved to {rerun_file}")
    print(f"Total tasks to re-run: {sum(len(tasks) for tasks in tasks_to_rerun.values())}")

if __name__ == "__main__":
    main()
