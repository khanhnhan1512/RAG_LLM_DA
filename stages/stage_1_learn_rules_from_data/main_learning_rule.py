import argparse
import os
import numpy as np
import torch
import time
from joblib import Parallel, delayed
from datetime import datetime

from stages.stage_1_learn_rules_from_data.data_loader import DataLoader
from stages.stage_1_learn_rules_from_data.utils import load_learn_data, calculate_relation_similarity, load_json_data, save_json_data
from openai_llm.llm_init import LLM_Model
from stages.stage_1_learn_rules_from_data.temporal_walk import TemporalWalker
from stages.stage_1_learn_rules_from_data.rule_learning import RuleLearner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets', help='path to the dataset')
    parser.add_argument('--dataset', "-d", type=str, default='icews14', help='dataset name')
    parser.add_argument('--transition_choice', type=str, default='exp', help='transition choice')
    parser.add_argument('--max_rule_length', type=int, default=3, help='Maximum length of a rule')
    parser.add_argument('--random_walk', type=int, default=200, help='Number of random walks')
    parser.add_argument('--num_process', type=int, default=16, help='Number of learning rule processes')
    parser.add_argument('--seed', '--s', type=int, default=42, help='random seed')
    parser.add_argument("--version", default="train", type=str,
                        choices=['train', 'test', 'train_valid', 'valid'])
    parser.add_argument("--is_relax_time", default=False, type=bool)
    parser = vars(parser.parse_args())
    return parser


def stage_1_main():
    args = parse_args()
    data_path = args['data_path']
    dataset = args['dataset']
    transition_choice = args['transition_choice']
    rule_length = (torch.arange(args['max_rule_length']) + 1).tolist()
    num_walks = args['random_walk']
    num_process = args['num_process']
    seed = args['seed']
    data_dir = os.path.join(data_path, dataset)

    data_loader = DataLoader(data_dir)
    llm_instance = LLM_Model()

    temporal_walk_data = load_learn_data(data_loader, 'train')
    temporal_walk = TemporalWalker(temporal_walk_data, data_loader.inverse_rel_idx, transition_choice)

    rl = RuleLearner(temporal_walk.edges, data_loader.id2relation, data_loader.inverse_rel_idx, dataset, len(temporal_walk_data))
    
    all_rels = sorted(temporal_walk.edges.keys())
    all_rels = [int(rel) for rel in all_rels]
    rel2idx = data_loader.relation2id

    def learn_rules_for_each_relation(rel, length, use_relax_time):
        for _ in range(num_walks):
            walk_successful, walk = temporal_walk.sample_walk(length+1, rel, use_relax_time)
            if walk_successful:
                rl.create_rule(walk, use_relax_time=use_relax_time)

    def learn_rules(i, num_relations, use_relax_time=False):
        """
        Learn rules with optional relax time (multiprocessing possible).

        Parameters:
            i (int): process number
            num_relations (int): minimum number of relations for each process
            use_relax_time (bool): Whether to use relax time in sampling

        Returns:
            rl.rules_dict (dict): rules dictionary
        """
        np.random.seed(seed)
        
        if i < num_relations - 1:
            relation_idx = range(i*num_relations, (i+1)*num_relations)
        else:
            relation_idx = range(i*num_relations, len(all_rels))

        for k in relation_idx:
            rel = all_rels[k]
            for length in rule_length:
                start = time.time()
                learn_rules_for_each_relation(rel, length, use_relax_time)
                end = time.time()
                total_time = round(end - start, 4)

                print(f"Process {i}: relation {k}, length {length}: {total_time} secs")
        
        return rl.rules_dict
    
    start = time.time()
    num_relations_per_process = len(all_rels) // num_process
    output = Parallel(n_jobs=num_process)(
        delayed(learn_rules)(i, num_relations_per_process, args['is_relax_time']) for i in range(num_process)
    )
    end =time.time()
    all_graph = output[0]
    for i in range(1, num_process):
        all_graph.update(output[i])
    print(f"Total time to learn rules: {round(end - start, 4)} secs.")

    rl.rules_dict = all_graph
    rl.sort_rules_dict()
    rl.remove_low_quality_rules()
    dt = datetime.now().strftime("%d%m%y%H%M%S")
    rl.save_rules_csv(dt, rule_length, num_walks, transition_choice, seed)
    rl.rules_statistics()
    # calculate_relation_similarity(llm_instance, list(data_loader.relation2id.keys()), rl.output_dir)

