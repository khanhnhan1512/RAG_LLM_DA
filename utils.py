import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from stages.stage_4_rule_application import main_rule_application as ra

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

def filter_candidates(test_query, candidates, test_data):
    """
    Filter out those candidates that are also answers to the test query
    but not the correct answer.

    Parameters:
        test_query (np.ndarray): test_query
        candidates (dict): answer candidates with corresponding confidence scores
        test_data (np.ndarray): test dataset

    Returns:
        candidates (dict): filtered candidates
    """

    other_answers = test_data[
        (test_data[:, 0] == test_query[0])
        * (test_data[:, 1] == test_query[1])
        * (test_data[:, 2] != test_query[2])
        * (test_data[:, 3] == test_query[3])
    ]

    if len(other_answers):
        objects = other_answers[:, 2]
        for obj in objects:
            candidates.pop(obj, None)

    return candidates

def calculate_rank(test_query_answer, candidates, num_entities, setting="best"):
    """
    Calculate the rank of the correct answer for a test query.
    Depending on the setting, the average/best/worst rank is taken if there
    are several candidates with the same confidence score.

    Parameters:
        test_query_answer (int): test query answer
        candidates (dict): answer candidates with corresponding confidence scores
        num_entities (int): number of entities in the dataset
        setting (str): "average", "best", or "worst"

    Returns:
        rank (int): rank of the correct answer
    """

    rank = num_entities
    if test_query_answer in candidates:
        conf = candidates[test_query_answer]
        all_confs = list(candidates.values())
        all_confs = sorted(all_confs, reverse=True)
        ranks = [idx for idx, x in enumerate(all_confs) if x == conf]

        try:

            if setting == "average":
                rank = (ranks[0] + ranks[-1]) // 2 + 1
            elif setting == "best":
                rank = ranks[0] + 1
            elif setting == "worst":
                rank = ranks[-1] + 1
        except Exception as e:
            ranks

    return rank

