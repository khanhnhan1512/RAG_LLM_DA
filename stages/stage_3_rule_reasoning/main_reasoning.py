import os.path
import time
import argparse
import itertools
import numpy as np
import torch
import pandas as pd
from joblib import Parallel, delayed

from stages.stage_1_learn_rules_from_data.data_loader import DataLoader
from stages.stage_1_learn_rules_from_data.temporal_walk import TemporalWalker
from stages.stage_1_learn_rules_from_data.temporal_walk import store_edges
from stages.stage_1_learn_rules_from_data.rule_learning import RuleLearner, rules_statistics
from stages.stage_3_rule_reasoning import rule_application as ra
from stages.stage_3_rule_reasoning.score_function import score_12, score_13, score_14
from utils import str_to_bool
from utils import load_json_data, get_win_subgraph, load_learn_data

def apply_rules(i, num_queries, parsed, test_data, window_subgraph, rules_dict, args, score_func, top_k):
    """
    Apply rules (multiprocessing possible).

    Parameters:
        i (int): process number
        num_queries (int): minimum number of queries for each process

    Returns:
        all_candidates (list): answer candidates with corresponding confidence scores
        no_cands_counter (int): number of queries with no answer candidates
    """
    print("Start process", i, "...")
    if torch.cuda.is_available() and parsed['gpu'] < torch.cuda.device_count():
        torch.cuda.set_device(parsed['gpu'])
    else:
        print(f"Warning: GPU {parsed['gpu']} is not available. Using CPU instead.")
    all_candidates = [dict() for _ in range(len(args))]
    all_timestamp = [dict() for _ in range(len(args))]

    no_candidates_counter = 0

    test_queries_idx = range(i * num_queries, (i+1) * num_queries) if i < parsed['num_processes'] - 1 else range(i*num_queries, len(test_data))

    if len(test_queries_idx) == 0:
        return all_candidates, no_candidates_counter, all_timestamp
    
    cur_ts = test_data[test_queries_idx[0]][3]
    edges = window_subgraph[cur_ts]

    it_start = time.time()
    for j in test_queries_idx:
        test_query = test_data[j]
        cands_dict = [dict() for _ in range(len(args))]
        timestamp_dict = [dict() for _ in range(len(args))]

        if test_query[3] != cur_ts:
            cur_ts = test_query[3]
            edges = window_subgraph[cur_ts]
        
        if test_query[1] in rules_dict:
            dicts_idx = list(range(len(args)))
            for rule_idx, rule in enumerate(rules_dict[test_query[1]]):
                walk_edges = ra.match_body_relations(rule, edges, test_query, is_sample=parsed["is_sampled"])

                corre = 0

                if 0 not in [len(x) for x in walk_edges]:
                    if parsed['evaluation_type'] != 'end':
                        rule_walks = ra.get_walks(rule, walk_edges, parsed['is_relax_time'])
                    else:
                        rule_walks = ra.get_walks_end(rule, walk_edges, parsed['is_relax_time'])
                    
                    if rule['var_constraints']:
                        rule_walks = ra.check_var_constraints(rule['var_constraints'], rule_walks)
                    
                    if not rule_walks.empty:
                        cands_dict, timestamp_dict = ra.get_candidates(
                            rule,
                            rule_walks,
                            cur_ts,
                            cands_dict,
                            score_func,
                            args,
                            dicts_idx,
                            corre,
                            parsed['is_return_timestamp'],
                            parsed['evaluation_type'],
                            timestamp_dict,
                        )
                        for s in dicts_idx:
                            if parsed['is_return_timestamp']:
                                for x in cands_dict[s].keys():
                                    # Get the sorted list and its index in cands_dict
                                    sorted_indices, sorted_cands = zip(
                                        *sorted(enumerate(cands_dict[s][x]), key=lambda pair: pair[1], reverse=True)
                                    )
                                    sorted_indices = list(sorted_indices)
                                    sorted_cands = list(sorted_cands)

                                    # Rearrange the corresponding list in timestamp_dict
                                    timestamp_dict[s][x] = [timestamp_dict[s][x][i] for i in sorted_indices]

                                    # Update the list in cands_dict to a sorted list
                                    cands_dict[s][x] = sorted_cands

                                # Sort cands_dict[s].
                                sorted_items = sorted(cands_dict[s].items(), key=lambda item: item[1], reverse=True)

                                # Update Cands_Dict[s]
                                cands_dict[s] = dict(sorted_items)

                                # Update timestamp_dict[s] in the same order
                                timestamp_dict[s] = {k: timestamp_dict[s][k] for k,_ in sorted_items}

                            else:
                                cands_dict[s] = {
                                    x: sorted(cands_dict[s][x], reverse=True)
                                    for x in cands_dict[s].keys()
                                }
                                cands_dict[s] = dict(sorted(cands_dict[s].items(), key=lambda item: item[1], reverse=True))
                            
                            if parsed['is_rule_priority'] is False:
                                top_k_scores = [v for _, v in cands_dict[s].items()][:top_k]
                                unique_scores = list(
                                    scores for scores, _ in itertools.groupby(top_k_scores)
                                )
                                if len(unique_scores) >= top_k:
                                    dicts_idx.remove(s)
                            else:
                                if rule_idx >= top_k:
                                    dicts_idx.remove(s)

                        if not dicts_idx:
                            break

            if cands_dict[0]:
                for s in range(len(args)):
                    # calculate noisy-or score
                    scores = calculate_scores(cands_dict[s], parsed)

                    scores = [float(x) for x in scores]
                    cands_scores = dict(zip(cands_dict[s].keys(), scores))
                    noisy_or_cands = dict(
                        sorted(cands_scores.items(), key=lambda x:x[1], reverse=True)
                    )

                    temp_timestamp = {}
                    if parsed['is_return_timestamp']:
                        for time_key, time_value in timestamp_dict[s].items():
                            temp_timestamp[time_key] = max(time_value)
                        all_timestamp[s][j] = temp_timestamp

                    all_candidates[s][j] = noisy_or_cands
            else:  # No candidates found by applying rules
                no_candidates_counter += 1
                for s in range(len(args)):
                    all_candidates[s][j] = dict()
                    if parsed['is_return_timestamp']:
                        all_timestamp[s][j] = dict()
        else: # No rule exists for this relation
            no_candidates_counter += 1
            for s in range(len(args)):
                all_candidates[s][j] = dict()
                if parsed['is_return_timestamp']:
                    all_timestamp[s][j] = dict()
        
        if not (j - test_queries_idx[0] + 1) % 100:
            it_end = time.time()
            it_time = round(it_end - it_start, 6)
            print(
                "Process {0}: test sample finished: {1}/{2}, {3} secs".format(
                    i, j - test_queries_idx[0] + 1, len(test_queries_idx), it_time
                )
            )
            it_start = time.time()
    
    return all_candidates, no_candidates_counter, all_timestamp

def calculate_scores(cands_dict, parsed):
    if parsed["score_type"] == 'noisy-or':
        scores = list(
            map(
                lambda x: 1 - np.prod(1 - np.array(x)),
                cands_dict.values()
            )
        )
    elif parsed['score_type'] == 'sum':
        scores = [sum(sublist) for sublist in list(cands_dict.values())]
    elif parsed['score_type'] == 'mean':
        scores = [sum(sublist) / len(sublist) for sublist in list(cands_dict.values())]
    elif parsed['score_type'] == 'min':
        scores = [min(sublist) for sublist in list(cands_dict.values())]
    elif parsed['score_type'] == 'max':
        scores = [max(sublist) for sublist in list(cands_dict.values())]
    return scores

def load_rules(rules_file, dir_path):
    rules_dict = load_json_data(os.path.join(dir_path, rules_file))
    return {int(k): v for k, v in rules_dict.items()}

def get_score_func(parsed):
    if parsed['evaluation_type'] == 'origin':
        parsed['coor_weight'] = 0
        return score_12
    elif parsed['evaluation_type'] == 'transformer':
        return score_13
    elif parsed['evaluation_type'] == 'end':
        parsed['coor_weight'] = 0
        return score_14
    
def apply_rules_in_parallel(parsed, test_data, window_subgraph, rules_dict, args, score_func):
    final_all_candidates = [dict() for _ in range(len(args))]
    final_all_timestamp = [dict() for _ in range(len(args))]
    final_no_cands_counter = 0

    start = time.time()

    num_queries = len(test_data) // parsed['num_processes']
    output = Parallel(n_jobs=parsed['num_processes'])(
        delayed(apply_rules)(i, num_queries, parsed, test_data, window_subgraph, rules_dict, args, score_func, parsed['top_k'])
        for i in range(parsed['num_processes'])
    )
    # output = []
    # for i in range(parsed['num_processes']):
    #     output.append(apply_rules(i, num_queries, parsed, test_data, window_subgraph, rules_dict, args, score_func, parsed['top_k']))

    for s in range(len(args)):
        for i in range(parsed['num_processes']):
            final_all_candidates[s].update(output[i][0][s])
            output[i][0][s].clear()

            final_all_timestamp[s].update(output[i][2][s])
            output[i][2][s].clear()

    for i in range(parsed['num_processes']):
        final_no_cands_counter += output[i][1]
    
    end = time.time()
    total_time = round(end - start, 6)
    print("Application finished in {} seconds.".format(total_time))

    return final_all_candidates, final_all_timestamp, final_no_cands_counter

def print_final_statistics(final_no_cands_counter, final_all_candidates):
    print("No candidates: ", final_no_cands_counter, " queries")

def save_results(final_all_candidates, final_all_timestamp, parsed, dir_path, rules_file, args, score_func):
    for s in range(len(args)):
        score_func_str = ''
        # score_func_str = f'{score_func.__name__}{args[s]}'.replace(" ", "")
        # score_func_str = f'{score_func_str}_rule_{parsed["is_rule_priority"]}_top_{parsed["top_k"]}_et_{parsed["evaluation_type"]}_sorted_{parsed["is_sorted"]}_bgkg_{parsed["bgkg"]}_start_{parsed["win_start"]}_relax_{parsed["is_relax_time"]}_sample_{parsed["is_sampled"]}'
        ra.save_candidates(rules_file, dir_path, final_all_candidates[s], parsed["max_rule_length"], parsed["window"], score_func_str, final_all_timestamp[s])

def parse_arguments():
    global parsed
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="icews14", type=str)
    parser.add_argument("--result_rules_path", default="./result/icews14/stage_2/filtered_historical_rules_and_current_rules_0_3.csv", type=str)
    parser.add_argument("--test_data", default="test", type=str)
    parser.add_argument("--rules", "-r", default="reasoning_result_2_half", type=str)
    parser.add_argument("--max_rule_length", "-l", default=3, type=int, nargs="+")
    parser.add_argument("--window", "-w", default=0, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--num_processes", "-p", default=8, type=int)
    parser.add_argument("--rule_files", "-f", default="", type=str)
    parser.add_argument("--confidence_type", default="Common", type=str,
                        choices=['Common', 'LLM', 'And', 'Or'])
    parser.add_argument("--weight", default=0.0, type=float)
    parser.add_argument("--weight_0", default=0.5, type=float)
    parser.add_argument("--min_conf", default=0.01, type=float)
    parser.add_argument("--coor_weight", default=0, type=float)
    parser.add_argument("--lmbda", default=0.1, type=float)
    parser.add_argument("--bgkg", default="all", type=str,
                        choices=['all', 'train', 'valid', 'test', 'train_valid', 'train_test', 'valid_test'])
    parser.add_argument("--score_type", default="noisy-or", type=str,
                        choices=['noisy-or', 'sum', 'mean', 'min', 'max'])
    parser.add_argument("--is_relax_time", default='no', type=str_to_bool)
    parser.add_argument("--is_sorted", default='Yes', type=str_to_bool)
    parser.add_argument("--is_return_timestamp", default='no', type=str_to_bool)
    parser.add_argument('--evaluation_type', type=str, default='origin', choices=['transformer', 'origin', 'end'])
    parser.add_argument("--win_start", default=0, type=int)
    parser.add_argument("--is_sampled", default='no', type=str_to_bool)
    parser.add_argument("--is_rule_priority", default='no', type=str_to_bool)

    return vars(parser.parse_args())

def stage_3_main():
    parsed = parse_arguments()
    rule_lengths = (torch.arange(parsed['max_rule_length']) + 1).tolist()

    dataset_dir = os.path.join(".", "datasets", parsed["dataset"])
    dir_path = os.path.join(".", "result", parsed["dataset"], "stage_3")

    data = DataLoader(dataset_dir, parsed)
    test_data = data.test_data_idx[:7371] if parsed["test_data"] == "test" else data.valid_data_idx

    # Load rules
    rule_regex = load_json_data("config/rule_regex.json")['icews14']
    df = pd.read_csv(parsed["result_rules_path"])
    temporal_walk_data = load_learn_data(data, 'all')
    temporal_walk = TemporalWalker(temporal_walk_data, data.inverse_rel_idx, 'exp')
    rl = RuleLearner(temporal_walk.edges, data.relation2id, data.id2entity, data.id2relation, data.inverse_rel_idx, 
                        parsed["dataset"], len(temporal_walk_data), dir_path)
    for _, entry in df.iterrows():
        rl.create_rule_from_series_df(entry=entry, rule_regex=rule_regex)
    rules_dict = rl.rules_dict
    print("Rules statistics:")
    rules_statistics(rules_dict)

    score_func = get_score_func(parsed)
    args = [[parsed['lmbda'], parsed['weight_0'], parsed['confidence_type'], parsed['weight'], parsed['min_conf'], parsed['coor_weight']]]

    new_rules_dict, sort_rules_dict = ra.filter_rules(
        rules_dict, min_conf=parsed["min_conf"], min_body_supp=2, rule_lengths=rule_lengths
    )

    rules_dict = new_rules_dict if not parsed["is_sorted"] else sort_rules_dict

    print("Rules statistics after pruning:")
    rules_statistics(rules_dict)
    learn_edges = store_edges(data.train_data_idx)

    window_subgraph = get_win_subgraph(test_data, data, learn_edges, parsed["window"], parsed["win_start"])

    final_all_candidates, final_all_timestamp, final_no_cands_counter = apply_rules_in_parallel(parsed, test_data, window_subgraph, rules_dict, args, score_func)

    print_final_statistics(final_no_cands_counter, final_all_candidates)

    save_results(final_all_candidates, final_all_timestamp, parsed, dir_path, parsed["rules"], args, score_func)

