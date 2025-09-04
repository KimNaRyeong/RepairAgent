import os, json, csv
import fasttext
import fasttext.util
import torch
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import from_networkx

def parse_json_block(file_path):
    blocks = []
    with open(file_path, 'r') as f:
        buffer = ""
        for line in f:
            buffer += line
            try:
                object = json.loads(buffer)
                blocks.append(object)
                buffer = ""
            except:
                continue
    
    if buffer:
        print(file_path)

    return blocks

def process_response_file(bug_name, exp_idx):
    experiment_dir = f'../../repair_agent/experimental_setups/experiment_{exp_idx}'
    raw_processed_response_file = os.path.join(experiment_dir, f'responses/processed_command_{bug_name}.json')
    
    if os.path.exists(raw_processed_response_file):
        parsed_blocks = parse_json_block(raw_processed_response_file)
        save_path = os.path.join(experiment_dir, f'processed_response/processed_command_{bug_name}.json')
        with open(save_path, 'w') as f:
            json.dump(parsed_blocks, f, indent=4)
        print(f'{save_path} is saved.')

def get_reasoning_paths_for_all_bugs(raw_response=True):
    bugs_list_file = '../bugs_list.txt'

    with open(bugs_list_file, 'r') as f:
        bugs_file_content = f.read().splitlines()
    bugs_list = []
    for bug_line in bugs_file_content:
        bug_name, start_idx, end_idx = bug_line.split()
        start_idx, end_idx = int(start_idx), int(end_idx)
        for i in range(start_idx, end_idx+1):
            bugs_list.append(f'{bug_name}_{i}')
    # bugs_list = ['Chart_1']
    # bugs_list = ['Lang_48']
    # bugs_list = ['Chart_1', 'Lang_48']
    # print(bugs_list)


    reasoning_paths_dict = defaultdict(list)

    for bug_name in tqdm(bugs_list):
        for i in range(1, 11):
            if raw_response: # Should be modified!!! to have only command
                traj_dir = f'../../repair_agent/experimental_setups/experiment_{i}/responses'
                traj_file = os.path.join(traj_dir, f'model_responses_{bug_name}.json')
            else:
                traj_dir = f'../../repair_agent/experimental_setups/experiment_{i}/processed_response'
                traj_file = os.path.join(traj_dir, f'processed_command_{bug_name}.json')
            if not os.path.exists(traj_file):
                process_response_file(bug_name, i)
            if os.path.exists(traj_file):
                with open(traj_file, 'r') as f:
                    trajectories = json.load(f)
                    reasoning_paths_dict[bug_name].append(trajectories)

            
    return reasoning_paths_dict



    
def embed_paths(model, reasoning_paths_dict, word_vector):
    embeddings_dict = defaultdict(list)

    print("Embedding trajectoreis...")
    for bug_name, trajectories in tqdm(reasoning_paths_dict.items()):
        for traj in trajectories:
            if word_vector:
                embedding_traj = [model.get_word_vector(str(f)) for f in traj]
            else:
                embedding_traj = [model.get_sentence_vector(str(f).replace('\n', ' ')) for f in traj]
            embeddings_dict[bug_name].append(embedding_traj)
    
    return embeddings_dict

# def visualize_graph(G, file_name, save_dir=None):
#     """Visualize the trajectory graph for a specific bug"""
#     save_path = os.path.join(save_dir, f'{file_name}.png')
#     plt.figure(figsize=(12, 8))
    
#     # Use spring layout for better visualization
#     pos = nx.spring_layout(G, k=1, iterations=50)
    
#     # Draw nodes with size proportional to cluster size
#     node_sizes = [G.nodes[node].get('size', 1) * 100 for node in G.nodes()]
#     nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
#                           node_color='lightblue', alpha=0.7)
    
#     # Draw edges with thickness proportional to weight
#     edges = G.edges()
#     if edges:
#         weights = [G[u][v]['weight'] for u, v in edges]
#         max_weight = max(weights) if weights else 1
#         edge_widths = [w / max_weight * 3 for w in weights]
        
#         nx.draw_networkx_edges(G, pos, width=edge_widths, 
#                               alpha=0.6, edge_color='gray', arrows=True)
    
#     # Draw labels
#     nx.draw_networkx_labels(G, pos, font_size=10)
    
#     plt.title(f"Trajectory Flow Graph - {file_name}")
#     plt.axis('off')
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')

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

    




def limit_embeddings_by_k(embeddings_dict, k):
    limited_embeddings_dict = defaultdict(list)

    for bugs_name, trajs in embeddings_dict.items():
        for traj_embeddings in trajs:
            limited_traj = traj_embeddings[:k]
            limited_embeddings_dict[bugs_name].append(limited_traj)
    
    return limited_embeddings_dict

def calculate_avg_node_num(graphs_dict):
    node_num = []
    for graph in graphs_dict.values():
        node_num.append(graph.number_of_nodes())
    return sum(node_num) / len(node_num)

def generate_command_to_label_mapping_dict():
    command_category_dict = {
        'Read and extract code': {
            'label': 0,
            'commands': {'read_range', 'get_classes_and_methods', 'extract_method', 'extract_tests'}
        },
        'Search and generate code': {
            'label': 1,
            'commands': {'search_code_base', 'find_similar_api_calls', 'generate_method_body'}
        },
        'Testing and patching': {
            'label': 2,
            'commands': {'run_tests', 'run_fault_localization', 'write_fix'}
        },
        'Control': {
            'label': 3,
            'commands': {'express_hypothesis', 'collect_more_information', 'discard_hypothesis', 'goal_accomplished'}
        }
    }

    command_to_label = {}
    for category_info in command_category_dict.values():
        label = category_info['label']
        for command in category_info['commands']:
            command_to_label[command] = label
    
    return command_to_label

def embedding_command_to_one_hot_vector_for_all_bugs(reasoning_paths_dict):
    command_to_label_dict = generate_command_to_label_mapping_dict()
    one_hot_vectors_dict = defaultdict(list)
    
    for bug_name, trajs in reasoning_paths_dict.items():
        for traj in trajs:
            one_hot_vector_for_traj = []
            for reasoning_step in traj:
                one_hot_vector = [0] * (max(list(command_to_label_dict.values())) + 1)
                command = reasoning_step["command_name"]
                if command in command_to_label_dict.keys():
                    one_hot_vector[command_to_label_dict[command]] = 1
                else:
                    one_hot_vector[-1] = 1
                one_hot_vector_for_traj.append(one_hot_vector)
            one_hot_vectors_dict[bug_name].append(one_hot_vector_for_traj)

    return one_hot_vectors_dict

def create_graph_for_bug(trajs):
    G = nx.DiGraph()

    for traj in trajs:
        prev_node = None
        for action_vector in traj:
            node = tuple(action_vector)
            G.add_node(node)
            current_node = node
            if prev_node is not None:
                if G.has_edge(prev_node, current_node):
                    G[prev_node][current_node]['weight'] += 1
                else:
                    G.add_edge(prev_node, current_node, weight=1)
            
            prev_node = current_node
    
    return G


def create_graphs_for_all_bugs(one_hot_vectors_dict):
    graphs_dict = {}

    for bug_name, one_hot_vectors in one_hot_vectors_dict.items():
        G = create_graph_for_bug(one_hot_vectors)
        graphs_dict[bug_name] = G
    return graphs_dict

def create_gcn_dataset_for_all_bugs(graphs_dict, labels_dict):
    dataset = []

    for bug_name, G in tqdm(graphs_dict.items()):
        data = from_networkx(G)
        node_features = []

        for node in G.nodes():
            node_features.append(list(node))
        data.bug_name = bug_name
        data.x = torch.tensor(node_features, dtype=torch.float)
        data.y = torch.tensor(labels_dict[bug_name], dtype=torch.long)

        dataset.append(data)

    return dataset

def visualize_graph(G, graph_title, save_path=None):
    plt.figure(figsize=(12, 8))
    
    
    # 레이아웃 설정
    try:
        pos = nx.spring_layout(G, k=1, iterations=50)
    except:
        pos = nx.random_layout(G)
    
    # 엣지 가중치 정보 가져오기
    edges = G.edges(data=True)
    weights = [edge[2].get('weight', 1) for edge in edges]
    total_weight = sum(weights)
    
    # 노드 차수에 따라 크기 결정
    node_sizes = [G.degree(node) * 100 + 100 for node in G.nodes()]
    
    # 그래프 그리기
    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes,
                          node_color='lightblue',
                          alpha=0.7)
    
    # 엣지 그리기 (가중치에 따라 두께 조절)
    if weights:
        max_weight = max(weights)
        edge_widths = [w / max_weight * 3 + 0.5 for w in weights]
        nx.draw_networkx_edges(G, pos, 
                              width=edge_widths,
                              alpha=0.6,
                              edge_color='gray')
    
    # 레이블은 노드가 적을 때만 표시
    if G.number_of_nodes() <= 20:
        # 노드 레이블을 간단하게 표시 (처음 몇 개 명령어만)
        labels = {}
        for i, node in enumerate(G.nodes()):
            if isinstance(node, (list, tuple)) and len(node) > 0:
                # n-hot vector에서 1인 위치들을 찾아서 표시
                active_indices = [j for j, val in enumerate(node) if val == 1]
                if active_indices:
                    labels[node] = f"Node_{i}\n{active_indices[:3]}"
                else:
                    labels[node] = f"Node_{i}"
            else:
                labels[node] = f"Node_{i}"
        
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title(f'Graph for {graph_title}\n'
              f'Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}, Total weight: {total_weight}')
    
    # 범례 추가
    plt.text(0.02, 0.98, 
             f'Original graph:\nNodes: {G.number_of_nodes()}\nEdges: {G.number_of_edges()}, Total weight: {total_weight}',
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.axis('off')
    plt.tight_layout()
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(os.path.join(save_path, f'{graph_title}.png'), dpi=300, bbox_inches='tight')
    print(f"Graph saved to {save_path}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label_criteria', default = 5, type=int)
    parser.add_argument('-r', '--raw_response', action="store_true")
    parser.add_argument('-p', '--plausible_patch', action="store_true")
    args = parser.parse_args()

    response_type = 'raw_response' if args.raw_response else 'processed_response'

    reasoning_paths_dict = get_reasoning_paths_for_all_bugs(args.raw_response)

    one_hot_vectors_dict = embedding_command_to_one_hot_vector_for_all_bugs(reasoning_paths_dict)
    labels_dict = load_labels(args.label_criteria, args.plausible_patch)

    k_values = [5, 10, 15, 20, 25, 30, 35, 40]

    for k in k_values:
        limited_one_hot_vectors_dict = limit_embeddings_by_k(one_hot_vectors_dict, k)

        graphs_dict = create_graphs_for_all_bugs(limited_one_hot_vectors_dict)
        gcn_dataset = create_gcn_dataset_for_all_bugs(graphs_dict, labels_dict)

        if args.plausible_patch:
            dataset_dir = f'../data/only_action/categorize/{response_type}/plausible_patch/label_criteria_{str(args.label_criteria)}/{k}'    
        else:
            dataset_dir = f'../data/only_action/categorize/{response_type}/label_criteria_{str(args.label_criteria)}/{k}'
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        
        torch.save(gcn_dataset, os.path.join(dataset_dir, 'gcn_dataset.pt'))

        print(f'Dataset for {k} is successfully genertaed!')
        


    # # For visualization

    # graphs_dict = create_graphs_for_all_bugs(one_hot_vectors_dict)
    # trajs_dir = f'../trajs_graphs/only_action/categorize'
    # if not os.path.exists(trajs_dir):
    #     os.makedirs(trajs_dir)

    # visualize_graph(graphs_dict['Chart_1'], f'Chart_1', trajs_dir)
    # visualize_graph(graphs_dict['Lang_48'], f'Lang_48', trajs_dir)

    # gcn_dataset = create_gcn_dataset_for_all_bugs(graphs_dict, clusterers_dict, labels_dict)

    # torch.save(gcn_dataset, 'gcn_dataset.pt')

    # # To calculate the average number of nodes for k
    # ks = list(range(1, 12))
    # # ks = [1]
    # avg_node_nums = []
    # for k in sorted(ks):
    #     limited_embeddings_dict = limit_embeddings_by_k(embedding_paths_dict, k)
    #     graphs_dict, clusterers_dict = create_trajectory_graphs_for_all_bugs(limited_embeddings_dict, threshold=0.98, merge_threshold=0.99)
        
    #     avg_node_nums.append(calculate_avg_node_num(graphs_dict))

    # for k in sorted(ks):
    #     print(f'{k} {avg_node_nums[k-1]}')






    