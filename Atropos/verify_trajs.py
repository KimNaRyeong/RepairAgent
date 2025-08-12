
import json, re, os

# def convert_content_to_list(input_text):
#     objects = []
#     brace_count = 0
#     current_object = ""
#     in_string = False
#     escape_next = False

#     for char in input_text:
#         if escape_next:
#             escape_next = False
#             current_object += char
#             continue
        
#         if char == '\\':
#             escape_next = True
#             current_object += char
#             continue

#         if char == '"':
#             in_string = not in_string

        
#         if not in_string:
#             if char == '{':
#                 brace_count += 1
#             elif char == '}':
#                 brace_count -= 1
        
#         current_object += char

#         if brace_count == 0 and current_object.strip():
#             parsed_obj = json.loads(current_object.strip())
#             objects.append(parsed_obj)
#             current_object = ""
    
#     return objects

def convert_response_to_string_list(input_text):
    command_str_list = []
    pattern = r"(\{\s*\n\"thoughts\".*?\n\})(?:\s*\{|$)"
    
    search_start = 0

    while search_start < len(input_text):
        remaining = input_text[search_start:]
        # print(remaining)
        # print('----------')
        match = re.search(pattern, remaining, re.DOTALL)

        if not match:
            break
        # print(match.start(), remaining[match.start():match.start()+10])
        # print(match.end(), remaining[match.end():match.end()+10])

        abs_start = search_start + match.start()
        abs_end = search_start + match.end() -1

        command_str_list.append(match.group(1))

        search_start = abs_end
    
    return command_str_list

    

def extract_last_command_count(input_text):
    pattern = r'executed, (\d+) commands'
    matches = re.findall(pattern, input_text, re.IGNORECASE)
    if matches:
        return int(matches[-1])
    else:
        return None

def search_zero_commands(input_text):
    pattern = "You have, so far, executed, 0 commands"
    matches = re.findall(pattern, input_text, re.IGNORECASE)
    return len(matches)

def extract_last_execution_log(input_text):
    sequences = input_text.split("============== ChatSequence ==============")

    if sequences and not sequences[0].strip():
        sequences = sequences[1:]
    
    zero_command_indices = []

    for i, sequence in enumerate(sequences):
        if "You have, so far, executed, 0 commands" in sequence:
            zero_command_indices.append(i)
    
    if not zero_command_indices:
        return input_text

    last_zero_index = zero_command_indices[-1]
    last_execution_sequences = sequences[last_zero_index:]
    result = "============== ChatSequence ==============".join(["\n"]+last_execution_sequences)

    return result

def process_log_files():
    for i in range(1, 11):
        log_file_dir = f'../repair_agent/experimental_setups/experiment_{i}/logs'
        log_files = os.listdir(log_file_dir)
        for log_file in log_files:
            splitted_log_file = log_file.split('_')
            pid, vid = splitted_log_file[-2], splitted_log_file[-1]
            if pid and vid:
                with open(os.path.join(log_file_dir, log_file), 'r') as f:
                    log_content = f.read()

                if search_zero_commands(log_content) > 1:
                    processed_log_content = extract_last_execution_log(log_content)
                    with open(os.path.join(log_file_dir, f'processed_prompt_history_{pid}_{vid}'), 'w') as f:
                        f.write(processed_log_content)

def find_command_strings_from_log(input_text):
    assistant_pattern = r"--------------- ASSISTANT ----------------\s*\n(\{.*?\n\})\s*\n(?=---|$)"
    matches = re.findall(assistant_pattern, input_text, re.DOTALL)

    command_strings = []

    for match in matches:
        match = match.strip()

        command_strings.append(match)
    return command_strings

def process_response_files():
    for i in range(1, 11):
        log_file_dir = f'../repair_agent/experimental_setups/experiment_{i}/logs'
        log_files = os.listdir(log_file_dir)
        for log_file in log_files:
            splitted_log_file = log_file.split('_')
            pid, vid = splitted_log_file[-2], splitted_log_file[-1]
            if pid and vid:
                with open(os.path.join(log_file_dir, log_file), 'r') as f:
                    log_content = f.read()

                if search_zero_commands(log_content) > 1:
                    # print(i, pid, vid)
                    last_execution_log = extract_last_execution_log(log_content)
                    command_strings_from_log = find_command_strings_from_log(last_execution_log)

                    response_file = f'../repair_agent/experimental_setups/experiment_{i}/responses/model_responses_{pid}_{vid}'

                    # if not os.path.exists(processed_response_file):
                    #     continue
                    with open(response_file, 'r') as f:
                        response_content = f.read()

                    command_str_list = convert_response_to_string_list(response_content)

                    first_command_in_last_execution = command_strings_from_log[0]
                    second_command_in_last_execution = command_strings_from_log[1]

                    idx_of_first_command = -1
                    for j, c in enumerate(command_str_list):
                        if c == first_command_in_last_execution and command_str_list[j+1] == second_command_in_last_execution:
                            idx_of_first_command = j

                    if idx_of_first_command != -1:
                        last_execution_response = command_str_list[idx_of_first_command:]
                        with open(f'../repair_agent/experimental_setups/experiment_{i}/responses/processed_model_responses_{pid}_{vid}', 'w') as f:
                            f.write("".join(last_execution_response))

def generate_bug_list():
    bug_list_file = './bugs_list.txt'

    log_file_dir = f'../repair_agent/experimental_setups/experiment_10/responses'
    log_files = os.listdir(log_file_dir)
    bug_set = set()
    
    for log_file in sorted(log_files):
        splitted_log_file = log_file.split('_')
        pid, vid = splitted_log_file[-2], splitted_log_file[-1]
        if vid.isdigit():
            bug_set.add(f'{pid}_{vid}')

    with open (bug_list_file, 'w') as f:
        f.write("\n".join(sorted(list(bug_set))))

def generate_response_json_files():
    bug_list_file = './bugs_list.txt'
    with open(bug_list_file, 'r') as f:
        bug_list = f.read().splitlines()
    
    for i in range(1, 11):
        response_file_dir = f'../repair_agent/experimental_setups/experiment_{i}/responses'
        response_files = os.listdir(response_file_dir)
        for bug_name in bug_list:
            if f'processed_model_responses_{bug_name}' in response_files:
                response_file = f'processed_model_responses_{bug_name}'
            else:
                response_file = f'model_responses_{bug_name}'

            response_file_path = os.path.join(response_file_dir, response_file)
            if os.path.exists(response_file_path):
                with open(os.path.join(response_file_dir, response_file), 'r') as f:
                    response_content = f.read()
                response_str_list = convert_response_to_string_list(response_content)
                response_json_file = f'model_responses_{bug_name}.json'
                with open(os.path.join(response_file_dir, response_json_file), 'w') as f:
                    json.dump(response_str_list, f, indent=2)

if __name__ == '__main__':
    generate_response_json_files()

    
                    


