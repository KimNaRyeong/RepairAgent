import json, os

def parse_json_block(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        blocks = []
        buffer = ""
        brace_count = 0

        for line in f:
            brace_count += line.count('{')
            brace_count -= line.count('}')
            buffer += line

            if brace_count == 0 and buffer.strip():
                try:
                    obj = json.loads(buffer)
                    blocks.append(obj)
                except json.JSONDecodeError as e:
                    # print("Failed to parse:", buffer)
                    print("Error:", e)
                    print(file_path)
                buffer = ""
    
    return blocks

def parse_and_save_responses_files():
    for i in range(1, 11):
        file_path = f'../repair_agent/experimental_setups/experiment_{i}/responses'
        files = os.listdir(file_path)
        print(files)
        for file in files:
            if file.startswith('processed'):
                splitted_file = file.split('_')
                bug_name = splitted_file[-2]
                bug_num = splitted_file[-1][:-5]

                parsed_blocks = parse_json_block(os.path.join(file_path, file))
                save_path = f'../repair_agent/experimental_setups/experiment_{i}/processed_response/processed_command_{bug_name}_{bug_num}.json'
                with open(save_path, 'w') as f:
                    json.dump(parsed_blocks, f, indent=4)
    

    

def main():
    for i in range(1, 11):
        files = os.listdir(f'../repair_agent/experimental_setups/experiment_{i}/responses')
        for file in files:
            if file.startswith('processed'):
                # parsed_blocks = 
                pass

if __name__ == "__main__":
    # chart_1 = "/workspaces/RepairAgent/repair_agent/experimental_setups/experiment_1/responses/processed_command_Chart_1.json"
    # parsed_blocks = parse_json_block(chart_1)
    # save_path = '../repair_agent/experimental_setups/experiment_1/parsed_chart1'
    # with open(save_path, 'w') as f:
    #     json.dump(parsed_blocks, f, indent=4)
    parse_and_save_responses_files()