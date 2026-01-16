import os
import shutil

bugs_list_hotswap_fold_file = '/workspaces/RepairAgent/repair_agent/experimental_setups/bugs_list_hotswap_fold3'
bugs_list_hotswap_fold_reurn_file = '/workspaces/RepairAgent/repair_agent/experimental_setups/bugs_list_hotswap_fold3_rerun'

with open(bugs_list_hotswap_fold_reurn_file, 'r') as f:
    bugs_list_hotswap_fold_rerurn = f.read().split('\n\n')

source_base_dir = '/workspaces/RepairAgent/repair_agent/experimental_setups/hotswap_gpt-4o_experiment_10'
dest_base_dir = '/workspaces/RepairAgent/repair_agent/experimental_setups/hotswap_gpt-4o_experiment_7'
print(bugs_list_hotswap_fold_rerurn)
for bug_name in bugs_list_hotswap_fold_rerurn:
    if not bug_name:
        continue
    pid, vid = bug_name.strip().split()
    bug_name = f'{pid}_{vid}'

    source_prompt_log_path = os.path.join(source_base_dir, f'logs/prompt_history_{bug_name}')
    dest_prompt_log_path = os.path.join(dest_base_dir, f'logs/prompt_history_{bug_name}')

    source_mutants_path = os.path.join(source_base_dir, f'mutations_history/mutants_{bug_name}.json')
    dest_mutants_path = os.path.join(dest_base_dir, f'mutations_history/mutants_{bug_name}.json')

    source_mutants_raw_path = os.path.join(source_base_dir, f'mutations_history/mutants_raw_{bug_name}.json')
    dest_mutants_raw_path = os.path.join(dest_base_dir, f'mutations_history/mutants_raw_{bug_name}.json')

    source_mutations_prompt_path = os.path.join(source_base_dir, f'mutations_history/mutations_prompt_{bug_name}')
    dest_mutations_prompt_path = os.path.join(dest_base_dir, f'mutations_history/mutations_prompt_{bug_name}')

    source_plausible_path = os.path.join(source_base_dir, f'plausible_patches/plausible_patches_{bug_name}.json')
    dest_plausible_path = os.path.join(dest_base_dir, f'plausible_patches/plausible_patches_{bug_name}.json')

    source_response_path = os.path.join(source_base_dir, f'responses/model_responses_{bug_name}')
    dest_response_path = os.path.join(dest_base_dir, f'responses/model_responses_{bug_name}')

    source_processed_command_path = os.path.join(source_base_dir, f'responses/processed_command_{bug_name}.json')
    dest_processed_command_path = os.path.join(dest_base_dir, f'responses/processed_command_{bug_name}.json')

    if os.path.exists(source_prompt_log_path):
        shutil.copy2(source_prompt_log_path, dest_prompt_log_path)
    if os.path.exists(source_mutants_path):
        shutil.copy2(source_mutants_path, dest_mutants_path)
    if os.path.exists(source_mutants_raw_path):
        shutil.copy2(source_mutants_raw_path, dest_mutants_raw_path)
    if os.path.exists(source_mutations_prompt_path):
        shutil.copy2(source_mutations_prompt_path, dest_mutations_prompt_path)
    if os.path.exists(source_plausible_path):
        shutil.copy2(source_plausible_path, dest_plausible_path)
    if os.path.exists(source_response_path):
        shutil.copy2(source_response_path, dest_response_path)
    if os.path.exists(source_processed_command_path):
        shutil.copy2(source_processed_command_path, dest_processed_command_path)


with open(bugs_list_hotswap_fold_file, 'r') as f:
    bugs_list_hotswap_fold = f.read().split('\n\n')

for i in range(11, 14):
    source_base_dir = f'/workspaces/RepairAgent/repair_agent/experimental_setups/hotswap_gpt-4o_experiment_{i}'
    dest_base_dir = f'/workspaces/RepairAgent/repair_agent/experimental_setups/hotswap_gpt-4o_experiment_{i-3}'
    for bug_name in bugs_list_hotswap_fold:
        if not bug_name:
            continue
        pid, vid = bug_name.strip().split()
        bug_name = f'{pid}_{vid}'
        print(bug_name)

        source_prompt_log_path = os.path.join(source_base_dir, f'logs/prompt_history_{bug_name}')
        dest_prompt_log_path = os.path.join(dest_base_dir, f'logs/prompt_history_{bug_name}')

        source_mutants_path = os.path.join(source_base_dir, f'mutations_history/mutants_{bug_name}.json')
        dest_mutants_path = os.path.join(dest_base_dir, f'mutations_history/mutants_{bug_name}.json')

        source_mutants_raw_path = os.path.join(source_base_dir, f'mutations_history/mutants_raw_{bug_name}.json')
        dest_mutants_raw_path = os.path.join(dest_base_dir, f'mutations_history/mutants_raw_{bug_name}.json')

        source_mutations_prompt_path = os.path.join(source_base_dir, f'mutations_history/mutations_prompt_{bug_name}')
        dest_mutations_prompt_path = os.path.join(dest_base_dir, f'mutations_history/mutations_prompt_{bug_name}')

        source_plausible_path = os.path.join(source_base_dir, f'plausible_patches/plausible_patches_{bug_name}.json')
        dest_plausible_path = os.path.join(dest_base_dir, f'plausible_patches/plausible_patches_{bug_name}.json')

        source_response_path = os.path.join(source_base_dir, f'responses/model_responses_{bug_name}')
        dest_response_path = os.path.join(dest_base_dir, f'responses/model_responses_{bug_name}')

        source_processed_command_path = os.path.join(source_base_dir, f'responses/processed_command_{bug_name}.json')
        dest_processed_command_path = os.path.join(dest_base_dir, f'responses/processed_command_{bug_name}.json')

        if os.path.exists(source_prompt_log_path):
            shutil.copy2(source_prompt_log_path, dest_prompt_log_path)
        if os.path.exists(source_mutants_path):
            shutil.copy2(source_mutants_path, dest_mutants_path)
        if os.path.exists(source_mutants_raw_path):
            shutil.copy2(source_mutants_raw_path, dest_mutants_raw_path)
        if os.path.exists(source_mutations_prompt_path):
            shutil.copy2(source_mutations_prompt_path, dest_mutations_prompt_path)
        if os.path.exists(source_plausible_path):
            shutil.copy2(source_plausible_path, dest_plausible_path)
        if os.path.exists(source_response_path):
            shutil.copy2(source_response_path, dest_response_path)
        if os.path.exists(source_processed_command_path):
            shutil.copy2(source_processed_command_path, dest_processed_command_path)