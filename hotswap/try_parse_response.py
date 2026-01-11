import re, json, os, ast

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
    
    response_dict_list = []

    for response_str in command_str_list:
        start_triple_quote = response_str.find("```")
        if start_triple_quote != -1:
            response_str = response_str[start_triple_quote:]
            end_triple_quote = response_str[3:].find("```")
            if end_triple_quote != -1:
                response_str = response_str[:end_triple_quote+3]
                response_str = "\n".join(response_str.split('\n')[1:])
        
        try:
            response_dict = ast.literal_eval(response_str)
        except:
            response_dict = {}
        
        response_dict_list.append(response_dict)
            
    
    return response_dict_list

    # pattern = r"(\{\s*\n\"thoughts\".*?\n\})(?:\s*\{|$)"
    
    # search_start = 0

    # while search_start < len(input_text):
    #     remaining = input_text[search_start:]
    #     # print(remaining)
    #     # print('----------')
    #     match = re.search(pattern, remaining, re.DOTALL)

    #     if not match:
    #         break
    #     # print(match.start(), remaining[match.start():match.start()+10])
    #     # print(match.end(), remaining[match.end():match.end()+10])

    #     abs_start = search_start + match.start()
    #     abs_end = search_start + match.end() -1

    #     command_str_list.append(match.group(1))

    #     search_start = abs_end
    
    return command_str_list

for i in range(1, 11):
    responses_dir = f'../repair_agent/experimental_setups/experiment_{i}/responses'
    response_files = os.listdir(responses_dir)

    for file in response_files:
        if not file.endswith('.json'):
            parsed_file = file.split('_')

            if parsed_file[0] and parsed_file[1] and parsed_file[2] and parsed_file[3]:
                response_file = os.path.join(responses_dir, file)
                with open(response_file, 'r') as f:
                    content = f.read()

                    responses_dict_list = convert_response_to_string_list(content)

for res in responses_dict_list:
    print(res)
    print('----------------')


                    

                    
