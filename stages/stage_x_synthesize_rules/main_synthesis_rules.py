import argparse
import os
from datetime import datetime

from stages.stage_1_learn_rules_from_data.rule_learning import RuleLearner, rules_statistics
from stages.stage_1_learn_rules_from_data.temporal_walk import TemporalWalker
from stages.stage_1_learn_rules_from_data.data_loader import DataLoader
from utils import load_json_data, load_learn_data
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets', help='path to the dataset')
    parser.add_argument('--dataset', "-d", type=str, default='icews14', help='dataset name')
    parser.add_argument('--transition_choice', type=str, default='exp', help='transition choice')
    parser.add_argument("--version", default="valid", type=str,
                        choices=['train', 'test', 'train_valid', 'valid', 'all'])
    parser = vars(parser.parse_args())
    return parser

def stage_3_main():

    # args for rule learning
    args = parse_args()
    data_path = args['data_path']
    dataset = args['dataset']
    version = args['version']
    transition_choice = args['transition_choice']
    data_dir = os.path.join(data_path, dataset)
    output_dir = "./result/" + dataset + "/stage_3/"
    rule_regex = load_json_data("config/rule_regex.json")[dataset]

    # for example, the llm-generated rules will be saved in a json file
    generated_rules = load_json_data(f"result/{dataset}/stage_2/generated_rules_added_output.json")
    
    # Code to update generated rules 
    data_loader = DataLoader(data_dir)
    temporal_walk_data = load_learn_data(data_loader, version)
    temporal_walk = TemporalWalker(temporal_walk_data, data_loader.inverse_rel_idx, transition_choice)
    rl = RuleLearner(temporal_walk.edges, data_loader.relation2id, data_loader.id2entity, data_loader.id2relation, data_loader.inverse_rel_idx, 
                     dataset, len(temporal_walk_data), output_dir)

    for rel in generated_rules:
        for rule in generated_rules[rel]:
            rl.create_llm_rule(rule, rule_regex)
    
    rl.sort_rules_dict()
    dt = datetime.now().strftime("%Y%m%d")
    rl.save_rules_csv(dt, "llm", metrics=["confidence_score"])
    rules_statistics(rl.rules_dict)