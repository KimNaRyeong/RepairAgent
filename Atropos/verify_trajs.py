
import json, re

def convert_content_to_list(input_text):
    objects = []
    brace_count = 0
    current_object = ""
    in_string = False
    escape_next = False

    for char in input_text:
        if escape_next:
            escape_next = False
            current_object += char
            continue
        
        if char == '\\':
            escape_next = True
            current_object += char
            continue

        if char == '"':
            in_string = not in_string

        
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
        
        current_object += char

        if brace_count == 0 and current_object.strip():
            parsed_obj = json.loads(current_object.strip())
            objects.append(parsed_obj)
            current_object = ""
    
    return objects

def extract_last_command_count(input_text):
    pattern = r'executed, (\d+) commands'
    matches = re.findall(pattern, input_text, re.IGNORECASE)
    if matches:
        return int(matches[-1])
    else:
        return None

if __name__ == '__main__':
    processed_response_file_path = '/home/kimnal0/RepairAgent/repair_agent/experimental_setups/experiment_1/responses/processed_command_Csv_1.json'
    log_file_path = '/home/kimnal0/RepairAgent/repair_agent/experimental_setups/experiment_1/logs/prompt_history_Csv_1'

    with open(processed_response_file_path, 'r') as f:
        processed_response_content = f.read()
    with open(log_file_path, 'r') as f:
        log_content = f.read()
    
    json_list = convert_content_to_list(processed_response_content)
    print(len(json_list))

    print(extract_last_command_count(log_content))