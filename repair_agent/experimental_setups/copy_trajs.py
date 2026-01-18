import shutil
import os

for i in range(1, 11):
    source_base_dir = f'/home/kimnal0/RepairAgent/repair_agent/experimental_setups/experiment_{i}'
    dest_base_dir = f'/home/kimnal0/atropos/trajectories/repair_agent/experiment_{i}'
    if not os.path.exists(dest_base_dir):
        os.makedirs(dest_base_dir)
    
    source_dir = os.path.join(source_base_dir, 'external_fixes')
    dest_dir = os.path.join(dest_base_dir, 'external_fixes')
    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)

    source_dir = os.path.join(source_base_dir, 'logs')
    dest_dir = os.path.join(dest_base_dir, 'logs')
    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)

    source_dir = os.path.join(source_base_dir, 'mutations_history')
    dest_dir = os.path.join(dest_base_dir, 'mutations_history')
    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)

    source_dir = os.path.join(source_base_dir, 'plausible_patches')
    dest_dir = os.path.join(dest_base_dir, 'plausible_patches')
    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)

    source_dir = os.path.join(source_base_dir, 'processed_response')
    dest_dir = os.path.join(dest_base_dir, 'processed_response')
    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)

    source_dir = os.path.join(source_base_dir, 'saved_contexts')
    dest_dir = os.path.join(dest_base_dir, 'saved_contexts')
    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)

    response_files = os.listdir(os.path.join(source_base_dir, 'responses'))
    for response in response_files:
        if not response.endswith('.json'):
            source_path = os.path.join(source_base_dir, 'responses', response)
            dest_dir = os.path.join(dest_base_dir, 'responses')
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            dest_path = os.path.join(dest_dir, response)
            shutil.copy2(source_path, dest_path)