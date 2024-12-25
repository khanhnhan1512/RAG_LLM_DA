import pandas as pd
import os

from stages.stage_1_learn_rules_from_data.rule_learning import RuleLearner, rules_statistics
from stages.stage_1_learn_rules_from_data.temporal_walk import TemporalWalker
from stages.stage_1_learn_rules_from_data.data_loader import DataLoader
from utils import load_json_data, load_learn_data


rule_regex = load_json_data("config/rule_regex.json")['icews14']
df = pd.read_csv('./01_only_Markovian_merged_results.csv')

# Code to update generated rules 
data_loader = DataLoader(os.path.join('datasets', 'icews14'))
temporal_walk_data = load_learn_data(data_loader, 'train')
temporal_walk = TemporalWalker(temporal_walk_data, data_loader.inverse_rel_idx, 'exp')
rl = RuleLearner(temporal_walk.edges, data_loader.relation2id, data_loader.id2entity, data_loader.id2relation, data_loader.inverse_rel_idx, 
                    'icews14', len(temporal_walk_data), "./result/" + "icews14" + "/stage_x/")
for _, entry in df.iterrows():
    rl.create_rule_from_series_df(entry=entry, rule_regex=rule_regex)

rules_statistics(rl.rules_dict)