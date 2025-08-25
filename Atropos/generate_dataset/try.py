import csv, os
from collections import defaultdict

def load_labels(criteria_num, use_plausible_patch):
    bugs_list_file = '../bugs_list.txt'
    with open(bugs_list_file, 'r') as f:
        bugs_file_content = f.read().splitlines()
    bugs_list = []
    for bug_line in bugs_file_content:
        bug_name, start_idx, end_idx = bug_line.split()
        start_idx, end_idx = int(start_idx), int(end_idx)
        for i in range(start_idx, end_idx+1):
            bugs_list.append(f'{bug_name}_{i}')

    resolved_num_dict = defaultdict(int)
    labels_dict = {}

    for i in range(1, 11):
        bug_correctly_fixed_dict = {}
        result_file = f'../../repair_agent/experimental_setups/experiment_{i}_results.csv'
        if use_plausible_patch:
            plausible_patch_dir = f'../../repair_agent/experimental_setups/experiment_{i}/plausible_patches'

        with open(result_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                bug_correctly_fixed_dict[row['Log File']] = row['Correctly Fixed']
        
        for bug_name in bugs_list:
                if use_plausible_patch:
                    pid, vid = bug_name.split('_')
                    plausible_patches_file = os.path.join(plausible_patch_dir, f'plausible_patches_{pid}_{vid}.json')

                    if bug_name in bug_correctly_fixed_dict.keys():
                        if bug_correctly_fixed_dict[bug_name] == 'Yes':
                            resolved_num_dict[bug_name] += 1
                        elif os.path.exists(plausible_patches_file):
                            resolved_num_dict[bug_name] += 1
                    elif os.path.exists(plausible_patches_file):
                        resolved_num_dict[bug_name] += 1
                        
                else:
                    if bug_name in bug_correctly_fixed_dict.keys():
                        if bug_correctly_fixed_dict[bug_name] == 'Yes':
                            resolved_num_dict[bug_name] += 1
    
    for bug_name in bugs_list:
        if resolved_num_dict[bug_name] >= criteria_num:
            labels_dict[bug_name] = 1
        else:
            labels_dict[bug_name] = 0

    return labels_dict

print(sum(list(load_labels(2, True).values())))
print(sum(list(load_labels(2, False).values())))