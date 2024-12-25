import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from stages.stage_b_rule_application import main_rule_application as ra

def load_json_data(file_path):
    try:
        print(f"Loading data from {file_path}")
        with open(file_path, 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        data = None
    return data

def save_json_data(data, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Data has been converted to JSON and saved to {file_path}")
    except Exception as e:
        print(f"Error saving JSON data to {file_path}")

def write_to_file(content, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)

def load_learn_data(data, type):
    data_map = {
        'all': np.array(data.train_data_idx.tolist() + data.valid_data_idx.tolist() + data.test_data_idx.tolist()),
        'train': np.array(data.train_data_idx.tolist()),
        'valid': np.array(data.valid_data_idx.tolist()),
        'test': np.array(data.test_data_idx.tolist()),
        'train_valid': np.array(data.train_data_idx.tolist() + data.valid_data_idx.tolist())
    }
    return data_map[type]

def calculate_relation_similarity(llm_instance, all_rels, output_dir):
    embedding_A = llm_instance.run_embedding(all_rels)
    embedding_B = llm_instance.run_embedding(all_rels)
    similarity = cosine_similarity(embedding_A, embedding_B)
    np.fill_diagonal(similarity, 0)
    np.save(os.path.join(output_dir, 'relation_similarity.npy'), similarity)

def get_win_subgraph(test_data, data, learn_edges, window, win_start=0):
    unique_timestamp_id = np.unique(test_data[:, 3])
    win_subgraph = {}
    for timestamp_id in unique_timestamp_id:
        subgraph = ra.get_window_edges(data.all_idx, timestamp_id - win_start, learn_edges, window)
        win_subgraph[timestamp_id] = subgraph
    return win_subgraph

