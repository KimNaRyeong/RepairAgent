import os
import json
import torch
import argparse
import fasttext
import fasttext.util
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import from_networkx

fasttext.util.download_model('en', if_exists='ignore')
fasttext_model = fasttext.load_model('cc.en.300.bin')
embedding_size = 300

def embed_with_fasttext(text):
    text_str = str(text)
    embedding = fasttext_model.get_sentence_vector(text_str)
    return embedding

class Clusterer:
    def __init__(self, threshold, merge_threshold):
        self.threshold = threshold
        self.merge_threshold = merge_threshold
        self.clusters = []
        self.cluster_centers = []
        self.step_to_cluster = {}

    def _cosine_sim(self, vec1, vec2):
        return cosine_similarity([vec1], [vec2])[0][0]
    
    def _calculate_center(self, vectors):
        if not vectors:
            return None
        return np.mean(vectors, axis=0)
    
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
        return self.step_to_cluster[step_id]


class Data_generator():
    def __init__(self, label_criteria):
        self.repetition = 10
        self.label_criteria = label_criteria
        self.ks = [5, 10, 15, 20, 25, 30, 35, 40]
        self.bug_list = self.get_bug_list()
        # self.bug_list = ['Chart_1', 'Chart_2']
        self.command_list = [
            "write_fix",
            "read_range",
            "go_back_to_collect_more_info",
            "discard_hypothesis",
            "goals_accomplished",
            "search_code_base",
            "get_classes_and_methods",
            "extract_similar_functions_calls",
            "extract_method_code",
            "AI_generates_method_code",
            "extract_test_code",
            "express_hypothesis",
        ]
        self.trajs_dict = self.extract_trajs_from_logs()
        self.labels_dict = self.get_labels_dict()
    
    def get_labels_dict(self):
        labels_dict = defaultdict(int)
        resolved_num_dict = defaultdict(int)

        print("Labeling data...")
        for bug_name in self.bug_list:
            for i in range(1, self.repetition+1):
                log_file = f'../../repair_agent/experimental_setups/experiment_{i}/logs/prompt_history_{bug_name}'

                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                    
                    if ' 0 failing test' in log_content:
                        resolved_num_dict[bug_name] += 1
                        continue
                
                plausible_patch_file = f'../../repair_agent/experimental_setups/experiment_{i}/plausible_patches/plausible_patches_{bug_name}.json'

                if os.path.exists(plausible_patch_file):
                    resolved_num_dict[bug_name] += 1
        
        for bug_name in self.bug_list:
            if resolved_num_dict[bug_name] < self.label_criteria: # label = 1 in incorrect case
                labels_dict[bug_name] = 1
            else:
                labels_dict[bug_name] = 0

        return labels_dict

    
    def extract_trajs_from_logs(self):
        trajs_dict = defaultdict(list)

        print("Extracting trajectories...")
        for bug_name in tqdm(self.bug_list):
            for i in range(1, self.repetition+1):
                traj_file = f'../../repair_agent/experimental_setups/experiment_{i}/processed_response/processed_command_{bug_name}.json'

                if os.path.exists(traj_file):
                    with open(traj_file, 'r') as f:
                        trajectory = json.load(f)
                        trajs_dict[bug_name].append(trajectory)
        
        return trajs_dict

    def get_bug_list(self):
        bugs_list_file = '../bugs_list.txt'
        bug_list = []
        with open(bugs_list_file, 'r') as f:
            bugs_file_content = f.read().splitlines()
        for bug_line in bugs_file_content:
            bug_name, start_idx, end_idx = bug_line.split()
            start_idx, end_idx = int(start_idx), int(end_idx)
            for i in range(start_idx, end_idx+1):
                bug_list.append(f'{bug_name}_{i}')
        return bug_list
    
    def embed_paths(self):
        embedding_path_dict = defaultdict(list)

        print("Embedding trajectories...")
        for bug_name in tqdm(self.bug_list):
            trajectories = self.trajs_dict[bug_name]
            for traj in trajectories:
                embed_traj = []
                for reasoning_step in traj:
                    command_name = reasoning_step["command_name"]
                    arguments = reasoning_step["arguments"]

                    command_vector = np.zeros(len(self.command_list)+1, dtype=np.float32)
                    if command_name in self.command_list:
                        command_vector[self.command_list.index(command_name)] = 1
                    else:
                        command_vector[-1] = 1
                    
                    arg_vector = embed_with_fasttext(str(arguments))

                    embedding = np.concatenate([command_vector, arg_vector])
                    embed_traj.append(embedding)
                embedding_path_dict[bug_name].append(embed_traj)
        
        return embedding_path_dict
    
    def cut_embeddings_by_k(self, embedding_path_dict, k):
        embeddings_until_k_dict = defaultdict(list)
        for bug_name in self.bug_list:
            trajectories = embedding_path_dict[bug_name]
            for traj in trajectories:
                cut_traj = traj[:k]
                embeddings_until_k_dict[bug_name].append(cut_traj)
        return embeddings_until_k_dict
    
    def create_graph_and_clusterer(self, trajectories, bug_name, threshold, merge_threshold):
        clusterer = Clusterer(threshold, merge_threshold)
        step_counter = 0

        for i, traj in enumerate(trajectories):
            for j, reasoning_step in enumerate(traj):
                step_id = f"{bug_name}_{i}_{j}"
                clusterer.add_step(reasoning_step, step_id)
                step_counter += 1
        
        clusterer.merge_similar_clusters()

        graph = nx.DiGraph()

        for i in range(len(clusterer.clusters)):
            graph.add_node(i, size=len(clusterer.clusters[i]))
        
        for i, traj in enumerate(trajectories):
            prev_cluster = None
            for j, reasoning_step in enumerate(traj):
                step_id = f"{bug_name}_{i}_{j}"
                current_cluster = clusterer.get_cluster_for_step(step_id)

                if prev_cluster is not None:
                    if graph.has_edge(prev_cluster, current_cluster):
                        graph[prev_cluster][current_cluster]['weight'] += 1
                    else:
                        graph.add_edge(prev_cluster, current_cluster, weight=1)
                
                prev_cluster = current_cluster
        
        return graph, clusterer

    
    def create_graphs_for_all_bugs(self, embeddings_dict, threshold, merge_threshold):
        graphs_dict = {}
        clusterers_dict = {}

        print("Creating graphs...")
        for bug_name in tqdm(self.bug_list):
            trajectories = embeddings_dict[bug_name]
            graph, clusterer = self.create_graph_and_clusterer(trajectories, bug_name, threshold, merge_threshold)
            graphs_dict[bug_name] = graph
            clusterers_dict[bug_name] = clusterer
        
        return graphs_dict, clusterers_dict
    
    def create_gcn_data_from_graph(self, graph, clusterer, bug_name):
        node_embeddings = []

        if len(graph.nodes()) == 0:
            command_vector = np.ones(len(self.command_list)+1, dtype=np.float32)
            none_embedding = embed_with_fasttext('None')
            node_embeddings.append(np.concatenate([command_vector, none_embedding]))
            graph.add_node(0, size=1)
        else:
            for cluster_idx in graph.nodes():
                cluster_center = clusterer.cluster_centers[cluster_idx]
                node_embeddings.append(cluster_center)

        data = from_networkx(graph)
        data.x = torch.tensor(np.array(node_embeddings), dtype = torch.float)
        data.y = torch.tensor([self.labels_dict[bug_name]], dtype=float)
        data.bug_name = bug_name
        data.edge_weight = torch.tensor([graph[u][v]['weight'] for u, v in graph.edges()], dtype=torch.float)

        if hasattr(data, 'weight'):
            delattr(data, 'weight')

        if hasattr(data, 'size'):
            delattr(data, 'size')

        return data
    
    def create_gcn_dataset_for_all_bugs(self, graphs_dict, clusterers_dict):
        dataset = []

        print("Creating GCN dataset")
        for bug_name in self.bug_list:
            graph = graphs_dict[bug_name]
            clusterer = clusterers_dict[bug_name]
            data = self.create_gcn_data_from_graph(graph, clusterer, bug_name)
            dataset.append(data)
        
        return dataset
        


    def generate_lig_for_all_k(self, threshold, merge_threshold, save_data, save_dir):
        embedding_path_dict = self.embed_paths()

        for k in self.ks:
            embeddings_until_k_dict = self.cut_embeddings_by_k(embedding_path_dict, k)
            graphs_dict, clusterers_dict = self.create_graphs_for_all_bugs(embeddings_until_k_dict, threshold, merge_threshold)

            gcn_dataset = self.create_gcn_dataset_for_all_bugs(graphs_dict, clusterers_dict)

            if save_data:
                save_dir_k = os.path.join(save_dir, str(k))
                os.makedirs(save_dir_k, exist_ok=True)
                torch.save({
                    "dataset": gcn_dataset
                }, os.path.join(save_dir_k, 'gcn_dataset.pth'))
                print(f"{k}th GCN dataset saved in {save_dir_k}")
        





def get_save_dir(label_criteria, threshold, merge_threshold):
    save_dir = f'../data/parallel/nhot_normal/center_vector/{threshold}_{merge_threshold}/label_criteria_{label_criteria}'
    return save_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label_criterion', default=1, type=int)
    parser.add_argument('-t', '--threshold', default=0.9, type=float)
    parser.add_argument('-m', '--merge_threshold', default=0.9, type=float)
    args = parser.parse_args()

    save_dir = get_save_dir(args.label_criterion, args.threshold, args.merge_threshold)

    data_generator = Data_generator(args.label_criterion)
    data_generator.generate_lig_for_all_k(args.threshold, args.merge_threshold, True, save_dir)
