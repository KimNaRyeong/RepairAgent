"""
Convert k8_tasks_to_rerun.json to a text file for ACR.
"""

import json
import os

def main():
    predictions_dir = '/home/kimnal0/auto-code-rover/hotswap/predictions'
    tasks_file = os.path.join(predictions_dir, 'k8_tasks_to_rerun.json')
    output_file = '/home/kimnal0/auto-code-rover/hotswap/incorrect_tasks.txt'

    # Load tasks to rerun
    with open(tasks_file, 'r') as f:
        tasks_dict = json.load(f)

    # Flatten all tasks across folds
    all_tasks = set()
    for fold, tasks in tasks_dict.items():
        all_tasks.update(tasks)

    all_tasks = sorted(list(all_tasks))

    # Write to text file
    with open(output_file, 'w') as f:
        for task in all_tasks:
            f.write(f"{task}\n")

    print(f"Total tasks to run: {len(all_tasks)}")
    print(f"Task list saved to: {output_file}")

    # Also save fold information
    fold_info_file = os.path.join(predictions_dir, 'fold_info.json')
    fold_info = {}
    for fold, tasks in tasks_dict.items():
        fold_info[fold] = {
            'count': len(tasks),
            'tasks': sorted(tasks)
        }

    with open(fold_info_file, 'w') as f:
        json.dump(fold_info, f, indent=4)

    print(f"Fold information saved to: {fold_info_file}")

    # Print summary
    print("\nSummary by fold:")
    for fold in sorted(tasks_dict.keys(), key=int):
        print(f"  Fold {fold}: {len(tasks_dict[fold])} tasks")


if __name__ == "__main__":
    main()
