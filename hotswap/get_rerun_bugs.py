import json
import os

hotswap_list_file = './predictions/k20_tasks_to_rerun.json'
with open(hotswap_list_file, 'r') as f:
    hotswap_dict = json.load(f)

hotswap_list = []
for i in range(3):
    bugs = hotswap_dict[str(i)]
    hotswap_list.extend(bugs)

for i in range(1, 11):
    print(f"====================== {i} =======================")
    for bug in hotswap_list:
        processed_command_file = f'../repair_agent/experimental_setups/experiment_{i}/processed_response/processed_command_{bug}.json'
        if not os.path.exists(processed_command_file):
            continue
    
        with open(processed_command_file, 'r') as f:
            processed_command_content = f.read()
            num_commands = processed_command_content.count('\"command_name\"')
            if num_commands <= 19:
                continue
        
        hotswap_response_file = f'../repair_agent/experimental_setups/hotswap_gpt-4o_experiment_{i}/responses/processed_command_{bug}.json'
        if not os.path.exists(hotswap_response_file):
            print(f"Not exists: {bug}")
            continue

        with open(hotswap_response_file, 'r') as f:
            content = f.read()
            num_commands = content.count('\"command_name\"')
            if num_commands <= 19:
                print(f"less command: {bug}")
        

        