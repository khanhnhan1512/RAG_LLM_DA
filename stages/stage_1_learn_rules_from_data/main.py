import argparse
import os
import numpy as np
import torch
import time

from data_loader import DataLoader
from utils import load_learn_data, calculate_relation_similarity, load_json_data, save_json_data
from openai_llm.llm_init import LLM_Model
from stages.stage_1_learn_rules_from_data.temporal_walk import TemporalWalker
from stages.stage_1_learn_rules_from_data.rule_learning import RuleLearner


def main(args):
    data_path = args['data_path']
    dataset = args['dataset']
    output_path = args['output_path']
    transition_choice = args['transition_choice']
    rule_length = (torch.arange(args['max_rule_length']) + 1).tolist()
    num_walks = args['random_walk']
    seed = args['seed']
    data_dir = os.path.join(data_path, dataset)

    data_loader = DataLoader(data_dir)
    llm_instance = LLM_Model()
    calculate_relation_similarity(llm_instance, list(data_loader.relation2id.keys()), output_path)

    temporal_walk_data = load_learn_data(data_loader, 'train')
    temporal_walk = TemporalWalker(temporal_walk_data, data_loader.inverse_rel_idx, transition_choice)

    rl = RuleLearner(temporal_walk.edges, data_loader.id2relation, data_loader.inverse_rel_idx, dataset)
    
    all_rels = sorted(temporal_walk.edges.keys())
    all_rels = [int(rel) for rel in all_rels]
    rel2idx = data_loader.relation2id

    regex_config = load_json_data('./config/regex.json')
    relation_regex = regex_config['relation_regex'][dataset]

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

        num_rules = [0]

        for k in relation_idx:
            rel = all_rels[k]
            for length in rule_length:
                start = time.time()
                learn_rules_for_each_relation(rel, length, use_relax_time)
                end = time.time()
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets', help='path to the dataset')
    parser.add_argument('--dataset', "-d", type=str, default='icews14', help='dataset name')
    parser.add_argument('--output_path', type=str, default='rules', help='path to save the rules')
    parser.add_argument('--transition_choice', type=str, default='exp', help='transition choice')
    parser.add_argument('--max_rule_length', type=int, default=3, help='Maximum length of a rule')
    parser.add_argument('--random_walk', type=int, default=200, help='Number of random walks')
    parser.add_argument('--seed', '--s', type=int, default=42, help='random seed')
    parser = vars(parser.parse_args())
    main(parser)
