import json
import os
import numpy as np
import argparse

from stages.stage_1_learn_rules_from_data.data_loader import DataLoader
from utils import filter_candidates, calculate_rank

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets', help='path to the dataset')
    parser.add_argument('--dataset', "-d", type=str, default='icews14', help='dataset name')
    parser.add_argument("--test_data", default="test", type=str)
    parser.add_argument('--graph_reasoning_type', type=str,
                        choices=['transformer', 'timestamp', 'based_source_with_timestamp', 'origin', 'fusion',
                                 'fusion_with_weight', 'fusion_with_source', 'fusion_with_relation', 'TADistmult',
                                 'TADistmult_with_recent', 'frequcy_only', 'new_origin_frequency', 'TiRGN', 'REGCN'],
                        default='timestamp')
    parser.add_argument("--rule_weight", default=0.5, type=float)
    parser.add_argument("--llm_weight", default=0.5, type=float)
    parser = vars(parser.parse_args())
    return parser

def stage_5_main():
    args = parse_args()
    data_dir = args['data_path']
    dataset = args['dataset']
    rule_candidates_file = "reasoning_result_1_1000.json"
    llm_candidates_file = "candidates_score_1000.json"

    dataset_path = os.path.join(data_dir, dataset)
    rule_reasoning_result_dir = os.path.join("./result", dataset, "stage_3")
    llm_reasoning_result_dir = os.path.join("./result", dataset, "stage_4")
    result_dir_path = os.path.join("./result", dataset, "stage_5")

    data = DataLoader(dataset_path)
    num_entities = len(data.id2entity)
    test_data = data.test_data_idx[:1000] if (args['test_data'] == "test") else data.valid_idx

    all_rule_candidates = load_candidates(rule_reasoning_result_dir, rule_candidates_file)
    all_llm_candidates = load_candidates(llm_reasoning_result_dir, llm_candidates_file)

    if args['graph_reasoning_type'] in ['TiRGN', 'REGCN']:
        test_numpy, score_numpy = load_test_and_score_data(dataset, dataset_path, args['graph_reasoning_type'])
    else:
        test_numpy, score_numpy = None, None

    results = evaluate(args, test_data, all_llm_candidates, all_rule_candidates, num_entities, test_numpy, score_numpy)
    hits_1, hits_3, hits_10, mrr = results

    hits_1 /= len(test_data)
    hits_3 /= len(test_data)
    hits_10 /= len(test_data)
    mrr /= len(test_data)

    print_results(hits_1, hits_3, hits_10, mrr)

    save_evaluation_results(result_dir_path, hits_1, hits_3, hits_10, mrr)

def load_candidates(ranked_rules_dir, candidates_file):
    with open(os.path.join(ranked_rules_dir, candidates_file), 'r') as f:
        candidates = json.load(f)
    return {int(k): {int(cand): v for cand, v in v.items()} for k, v in candidates.items()}

def calculate_test_interval(data):
    recent_time = max(data.valid_idx[:, 3])
    test_timestamp = set(data.test_idx[:, 3])
    return {timestamp: timestamp - recent_time for timestamp in test_timestamp}

def load_test_and_score_data(dataset, dataset_dir, graph_reasoning_type):
    test_numpy = np.load(os.path.join(dataset_dir, graph_reasoning_type, 'test.npy'))
    if dataset == 'icews18':
        test_numpy[:, 3] = (test_numpy[:, 3] / 24).astype(int)
    score_numpy = np.load(os.path.join(dataset_dir, graph_reasoning_type, 'score.npy'))
    return test_numpy, score_numpy

def evaluate(args, test_data, all_llm_candidates, all_rule_candidates, num_entities, test_numpy, score_numpy):
    hits_1 = hits_3 = hits_10 = mrr = 0
    num_samples = len(test_data)

    for i in range(num_samples):
        test_query = test_data[i]
        candidates = get_final_candidates(args, test_query, all_llm_candidates, all_rule_candidates, i, num_entities, test_numpy, score_numpy)
        candidates = filter_candidates(test_query, candidates, test_data)
        rank = calculate_rank(test_query[2], candidates, num_entities)

        hits_1, hits_3, hits_10, mrr = update_metrics(hits_1, hits_3, hits_10, mrr, rank)

    return hits_1, hits_3, hits_10, mrr

def get_final_candidates(args, test_query, all_llm_candidates, all_rule_candidates, i, num_entities, test_numpy, score_numpy):
    # if args['graph_reasoning_type'] in ['TiRGN', 'REGCN']:
    #     return get_candidates(args, test_query, all_rule_candidates, i, num_entities, test_numpy, score_numpy)
    # else:
    # return get_rule_llm_candidates(args, i, all_llm_candidates, all_rule_candidates, num_entities)
    return all_rule_candidates[i]

def get_rule_llm_candidates(args, i, all_llm_candidates, all_rule_candidates, num_entities):
    temp_candidates = {k: 0 for k in range(num_entities)}
    rule_candidates = all_rule_candidates[i]
    rule_candidates = {**temp_candidates, **rule_candidates}
    llm_candidates = all_llm_candidates[i]
    candidates = {}
    for k in rule_candidates:
        rule_score = rule_candidates.get(k, 0.0)
        llm_score = llm_candidates.get(k, 0.0)
        candidates[k] = args['llm_weight'] * llm_score + args['rule_weight'] * rule_score
    return candidates

def get_candidates(args, test_query, all_rule_candidates, i, num_entities, test_numpy, score_numpy):
    temp_candidates = {k: 0 for k in range(num_entities)}
    rule_candidates = all_rule_candidates[i]
    rule_candidates = {**temp_candidates, **rule_candidates}

    indices = np.where((test_numpy == test_query).all(axis=1))[0]
    score = score_numpy[indices[0]]
    regcn_candidates = {index: value for index, value in enumerate(score)}

    candidates = {k: (1 - args['rule_weight']) * regcn_candidates[k] + args['rule_weight'] * rule_candidates[k] for k in
                  rule_candidates}
    return candidates

def update_metrics(hits_1, hits_3, hits_10, mrr, rank):
    if rank <= 10:
        hits_10 += 1
        if rank <= 3:
            hits_3 += 1
            if rank == 1:
                hits_1 += 1
    mrr += 1 / rank
    return hits_1, hits_3, hits_10, mrr

def print_results(hits_1, hits_3, hits_10, mrr):
    print("Hits@1: ", round(hits_1, 6))
    print("Hits@3: ", round(hits_3, 6))
    print("Hits@10: ", round(hits_10, 6))
    print("MRR: ", round(mrr, 6))

def save_evaluation_results(result_dir_path, hits_1, hits_3, hits_10, mrr):
    filename = "result_eval.txt"
    if not os.path.exists(result_dir_path):
        os.makedirs(result_dir_path)
        with open(os.path.join(result_dir_path, filename), "w", encoding="utf-8") as fout:
            fout.write("Hits@1: " + str(round(hits_1, 6)) + "\n")
            fout.write("Hits@3: " + str(round(hits_3, 6)) + "\n")
            fout.write("Hits@10: " + str(round(hits_10, 6)) + "\n")
            fout.write("MRR: " + str(round(mrr, 6)) + "\n")

