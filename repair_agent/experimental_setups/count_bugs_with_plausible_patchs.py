import csv
import os
from collections import defaultdict

def main(repetition, model):
    if model == 'gpt-4o':
        bug_list_dict = {
            'Chart': (1, 26),
            'Cli': (1, 39),
            'Closure': (1, 50),
            'Closure': (151, 174),
            'Codec': (1, 18),
            'Math': (1, 106)
        }
    elif model == 'gpt-3.5':
        bug_list_dict = {
            'Chart': (1, 26),
            'Cli': (1, 39),
            'Closure': (1, 174),
            'Codec': (1, 18),
            'Math': (1, 106),
            'Compress': (1, 47),
            'Csv': (1, 16),
            'JacksonCore': (1, 26),
            'Lang': (1, 63),
            'Jsoup': (1, 93)
        }
    bug_list = []
    for pid, vid_range in bug_list_dict.items():
        for i in range(vid_range[0], vid_range[1]+1):
            bug_list.append(f'{pid}_{i}')
    
    bug_patch_dict = {bug_name: 0 for bug_name in bug_list}
    for i in range(1, repetition+1):
        bug_correctly_fixed_dict = {}
        if model == 'gpt-4o':
            results_csv_file = f'./{model}_experiment_{i}_results.csv'
        elif model == 'gpt-3.5':
            results_csv_file = f'./experiment_{i}_results.csv'
        with open(results_csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                bug_correctly_fixed_dict[row['Log File']] = row['Correctly Fixed']
    
        for bug_name in bug_list:
            patched = False
            if model == 'gpt-4o':
                plausible_patches_file = f'./{model}_experiment_{i}/plausible_patches/plausible_patches_{bug_name}.json'
            elif model == 'gpt-3.5':
                plausible_patches_file = f'./experiment_{i}/plausible_patches/plausible_patches_{bug_name}.json'
            if os.path.exists(plausible_patches_file):
                patched = True
            if bug_name in bug_correctly_fixed_dict.keys():
                if bug_correctly_fixed_dict[bug_name] == 'Yes':
                    patched = True
            if patched:
                bug_patch_dict[bug_name] += 1
    
    print(f"Total bug num: {len(bug_patch_dict)}")
    print(f"Number of patches bugs: {sum([num_patched_bug > 0 for num_patched_bug in bug_patch_dict.values()])}")



if __name__ == '__main__':
    main(10, 'gpt-3.5')