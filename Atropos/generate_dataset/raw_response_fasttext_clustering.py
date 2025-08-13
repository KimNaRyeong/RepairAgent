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

class Clusterer:
    def __init__(self, threshold=0.7, merge_threshold=0.8):
        self.threshold = threshold
        self.merge_threshold = merge_threshold
        self.clusters = []
        self.cluster_centers = []
        self.step_to_cluster = {}
    
    def _calculate_center(self, vectors):
        if not vectors:
            return None
        return np.mean(vectors, axis=0)
    
    def _cosine_sim(self, vec1, vec2):
        return cosine_similarity([vec1], [vec2])[0][0]
    
    def add_step(self, step_vector, step_id):
        if not self.clusters:
            self.clusters.append([step_vector])
            self.cluster_centers.append(step_vector.copy())
            self.step_to_cluster[step_id] = 0
            return 0
        
        similarities = []
        for center in self.cluster_centers:
            sim = self._cosine_sim(step_vector, center)
            similarities.append(sim)
        
        above_threshold = [(i, sim) for i, sim in enumerate(similarities) if sim > self.threshold]

        if len(above_threshold) == 0:
            cluster_idx = len(self.clusters)
            self.clusters.append([step_vector])
            self.cluster_centers.append(step_vector.copy())
            self.step_to_cluster[step_id] = cluster_idx
            return cluster_idx
        elif len(above_threshold) == 1:
            cluster_idx = above_threshold[0][0]
            self.clusters[cluster_idx].append(step_vector)
            self.cluster_centers[cluster_idx] = self._calculate_center(self.clusters[cluster_idx])
            self.step_to_cluster[step_id] = cluster_idx
            return cluster_idx
        else:
            best_cluster_idx = max(above_threshold, key=lambda x: x[1])[0]
            self.clusters[best_cluster_idx].append(step_vector)
            self.cluster_centers[best_cluster_idx] = self._calculate_center(self.clusters[best_cluster_idx])
            self.step_to_cluster[step_id] = best_cluster_idx
            return best_cluster_idx
    
    def merge_similar_clusters(self):
        """Merge clusters whose centers have similarity > threshold2"""
        if len(self.cluster_centers) <= 1:
            return
        
        merged = True
        while merged:
            merged = False
            to_remove = []
            for i in range(len(self.cluster_centers)):
                if i in to_remove:
                    continue
                for j in range(i + 1, len(self.cluster_centers)):
                    if j in to_remove:
                        continue
                    
                    sim = self._cosine_sim(self.cluster_centers[i], self.cluster_centers[j])
                    if sim > self.merge_threshold:
                        self.clusters[i].extend(self.clusters[j])
                        self.cluster_centers[i] = self._calculate_center(self.clusters[i])
                        
                        for step_id, cluster_idx in self.step_to_cluster.items():
                            if cluster_idx == j:
                                self.step_to_cluster[step_id] = i
                            elif cluster_idx > j:
                                self.step_to_cluster[step_id] = cluster_idx - 1
                        
                        to_remove.append(j)
                        merged = True
                        break
                
                if merged:
                    break
            
            for idx in sorted(to_remove, reverse=True):
                del self.clusters[idx]
                del self.cluster_centers[idx]
    
    def get_cluster_for_step(self, step_id):
        return self.step_to_cluster.get(step_id, -1)
    
    def get_most_central_vector_for_cluster(self, cluster_idx):
        cluster_vectors = self.clusters[cluster_idx]
        cluster_center = self.cluster_centers[cluster_idx]

        best_vector = None
        best_similarity = -1

        for vector in cluster_vectors:
            sim = self._cosine_sim(vector, cluster_center)
            if sim > best_similarity:
                best_similarity = sim
                best_vector = vector
        
        return best_vector

    


def load_fasttext_model(embedding_length):
    embedding_size = embedding_length
    fasttext.util.download_model('en', if_exists='ignore')
    model = fasttext.load_model('cc.en.300.bin')
    fasttext.util.reduce_model(model, embedding_size)
    return model

def get_reasoning_paths_for_all_bugs(raw_response=True):
    bugs_list_file = '../bugs_list.txt'

    with open(bugs_list_file, 'r') as f:
        bugs_list = f.read().splitlines()
    # bugs_list = ['Chart_1']


    reasoning_paths_dict = defaultdict(list)

    for bug_name in tqdm(bugs_list):
        for i in range(1, 11):
            traj_dir = f'../../repair_agent/experimental_setups/experiment_{i}/responses'
            if raw_response:
                traj_file = os.path.join(traj_dir, f'model_responses_{bug_name}.json')
            else:
                traj_file = os.path.join(traj_dir, f'processed_model_responses_{bug_name}.json') # Should be modified
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
                embedding_traj = [model.get_word_vector(f) for f in traj]
            else:
                embedding_traj = [model.get_sentence_vector(f.replace('\n', ' ')) for f in traj]
            embeddings_dict[bug_name].append(embedding_traj)
    
    return embeddings_dict

def create_trajectory_graph_for_bug(trajs, bug_name, threshold=0.1, merge_threshold=0.1):
    clusterer = Clusterer(threshold, merge_threshold)
    step_counter = 0


    for i, traj in enumerate(trajs):
        for j, reasoning_step in enumerate(traj):
            step_id = f"{bug_name}_{i}_{j}"
            clusterer.add_step(reasoning_step, step_id)
            step_counter += 1
    
    # print("done..")
    
    clusterer.merge_similar_clusters()


    G = nx.DiGraph()

    for i in range(len(clusterer.clusters)):
        G.add_node(i, size=len(clusterer.clusters[i]))
    
    for i, traj in enumerate(trajs):
        prev_cluster = None
        for j, reasoning_step in enumerate(traj):
            step_id = f"{bug_name}_{i}_{j}"
            current_cluster = clusterer.get_cluster_for_step(step_id)

            if prev_cluster is not None:
                if G.has_edge(prev_cluster, current_cluster):
                    G[prev_cluster][current_cluster]['weight'] += 1
                else:
                    G.add_edge(prev_cluster, current_cluster, weight=1)
            
            prev_cluster = current_cluster
    
    return G, clusterer




def create_trajectory_graphs_for_all_bugs(embeddings_dict, threshold=0.7, merge_threshold=0.8):
    graphs_dict = {}
    clusterers_dict = {}

    print("Creating graphs of trajectories...")
    for bug_name, trajs in tqdm(embeddings_dict.items()):
        G, clusterer = create_trajectory_graph_for_bug(trajs, bug_name, threshold, merge_threshold)
        graphs_dict[bug_name] = G
        clusterers_dict[bug_name] = clusterer
    
    return graphs_dict, clusterers_dict

def visualize_graph(G, file_name, save_dir=None):
    """Visualize the trajectory graph for a specific bug"""
    save_path = os.path.join(save_dir, f'{file_name}.png')
    plt.figure(figsize=(12, 8))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes with size proportional to cluster size
    node_sizes = [G.nodes[node].get('size', 1) * 100 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.7)
    
    # Draw edges with thickness proportional to weight
    edges = G.edges()
    if edges:
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        edge_widths = [w / max_weight * 3 for w in weights]
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, 
                              alpha=0.6, edge_color='gray', arrows=True)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title(f"Trajectory Flow Graph - {file_name}")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def load_labels(criteria_num):
    bugs_list_file = '../bugs_list.txt'
    with open(bugs_list_file, 'r') as f:
        bugs_list = f.read().splitlines()

    resolved_num_dict = defaultdict(int)
    labels_dict = {}

    for i in range(1, 11):
        result_file = f'../../repair_agent/experimental_setups/experiment_{i}_results.csv'

        with open(result_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Correctly Fixed'] == "Yes":
                    resolved_num_dict[row['Log File']] += 1
    
    for bug_name in bugs_list:
        if resolved_num_dict[bug_name] >= criteria_num:
            labels_dict[bug_name] = 1
        else:
            labels_dict[bug_name] = 0

    return labels_dict
    

def create_gcn_data_from_graph(G, clusterer, bug_name, label):
    node_embeddings = []

    for node_idx in range(G.number_of_nodes()):
        central_vector = clusterer.get_most_central_vector_for_cluster(node_idx)
        node_embeddings.append(central_vector)
    
    data = from_networkx(G)
    data.x = torch.tensor(np.array(node_embeddings), dtype=torch.float)

    # if G.number_of_edges() > 0:
    #     edge_weights = [G[u][v] for u, v in G.edges()]
    
    data.y = torch.tensor([label], dtype=torch.long)

    return data
    


def create_gcn_dataset_for_all_bugs(graphs_dict, clusterers_dict, labels_dict):
    dataset = []

    print("Creating GCN dataset...")
    for bug_name, G in tqdm(graphs_dict.items()):
        clusterer = clusterers_dict[bug_name]

        data = create_gcn_data_from_graph(G, clusterer, bug_name, labels_dict[bug_name])

        data.bug_name = bug_name
        dataset.append(data)

    return dataset

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label_criteria', default = 5, type=int)
    parser.add_argument('-r', '--raw_response', action="store_true")
    parser.add_argument('-w', '--word_vector', action="store_true")
    parser.add_argument('-e', '--embedding_length', default=100, type=int)
    parser.add_argument('-t', '--threshold', default=0.9, type=float)
    parser.add_argument('-m', '--merge_threshold', default=0.9, type=float)
    args = parser.parse_args()

    reasoning_paths_dict = get_reasoning_paths_for_all_bugs(args.raw_response)


    model = load_fasttext_model(args.embedding_length)
    embedding_paths_dict = embed_paths(model, reasoning_paths_dict, args.word_vector)

    # bugs_list = list(reasoning_paths_dict.keys())
    labels_dict = load_labels(args.label_criteria)

    # print(labels_dict)
    # k_values = [5, 10, 15, 20, 25, 30, 35, 40]
    k_values = [5, 10, 15, 20, 25]
    threshold = args.threshold
    merge_threshold = args.merge_threshold
    response_type = 'raw_response' if args.raw_response else 'processed_response'
    embedding_type = 'word_vector' if args.word_vector else 'sentence_vector'

    datasets_dict = {}
    for k in k_values:
        limited_embeddings_dict = limit_embeddings_by_k(embedding_paths_dict, k)

        graphs_dict, clusterers_dict = create_trajectory_graphs_for_all_bugs(limited_embeddings_dict, threshold, merge_threshold)

        gcn_dataset = create_gcn_dataset_for_all_bugs(graphs_dict, clusterers_dict, labels_dict)

        dataset_dir = f'../data/clustering/fasttext/{embedding_type}/{args.embedding_length}/{response_type}/{threshold}_{merge_threshold}/label_criteria_{str(args.label_criteria)}/{k}'
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        
        torch.save(gcn_dataset, os.path.join(dataset_dir, 'gcn_dataset.pt'))

        print(f'Dataset for {k} is successfully genertaed!')
        


    # # For visualization

    # graphs_dict, clusterers_dict = create_trajectory_graphs_for_all_bugs(embedding_paths_dict, threshold, merge_threshold)
    # trajs_dir = f'../trajs_graphs/clustering/fasttext/{embedding_type}/{args.embedding_length}/{response_type}/{threshold}_{merge_threshold}/label_criteria_{str(args.label_criteria)}'
    # if not os.path.exists(trajs_dir):
    #     os.makedirs(trajs_dir)

    # visualize_graph(graphs_dict['Chart_1'], f'Chart_1_{threshold}_{merge_threshold}', trajs_dir)

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






    