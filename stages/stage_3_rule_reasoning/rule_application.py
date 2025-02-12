import json
import os
import numpy as np
import pandas as pd

from stages.stage_1_learn_rules_from_data.temporal_walk import store_edges

def filter_rules(rules_dict, min_conf, min_body_supp, rule_lengths):
    """
    Filter for rules with a minimum confidence, minimum body support, and
    specified rule lengths.

    Parameters.
        rules_dict (dict): rules
        min_conf (float): minimum confidence value
        min_body_supp (int): minimum body support value
        rule_lengths (list): rule lengths

    Returns:
        new_rules_dict (dict): filtered rules
    """
    new_rules_dict = dict()
    sort_rules_dict = dict()

    for k in rules_dict:
        new_rules_dict[k] = []
        for rule in rules_dict[k]:
            conditions = (
                (rule["conf"] >= min_conf) and
                (rule["body_supp_count"] >= min_body_supp) and
                (len(rule["body_rels"]) in rule_lengths)
            )
            if conditions:
                new_rules_dict[k].append(rule)
    
    for relation_id, rules in new_rules_dict.items():
        new_rules = sorted(rules, key=lambda x: x['conf'], reverse=True)
        sort_rules_dict[relation_id] = new_rules

    return new_rules_dict, sort_rules_dict

def get_window_edges(all_data, test_query_ts, learn_edges, window=-1):
    """
    Get the edges in the data (for rule application) that occur in the specified time window.
    If window is 0, all edges before the test query timestamp are included.
    If window is -1, the edges on which the rules are learned are used.
    If window is an integer n > 0, all edges within n timestamps before the test query
    timestamp are included.

    Parameters:
        all_data (np.ndarray): complete dataset (train/valid/test)
        test_query_ts (np.ndarray): test query timestamp
        learn_edges (dict): edges on which the rules are learned
        window (int): time window used for rule application

    Returns:
        window_edges (dict): edges in the window for rule application
    """
    if window > 0:
        mask = (all_data[:, 3] < test_query_ts *
                (all_data[:, 3] >= test_query_ts - window))
        window_edges = store_edges(all_data[mask])
    elif window == 0:
        mask = all_data[:, 3] < test_query_ts
        window_edges = store_edges(all_data[mask])
    elif window == -1:
        window_edges = learn_edges

    return window_edges

def get_window_edges_for_subject_object(all_data, current_ts, subject_object, window=-1):
    """
    Get the edges in the data (for rule application) that occur in the specified time window.
    If window is 0, all edges before the test query timestamp are included.
    If window is -1, the edges on which the rules are learned are used.
    If window is an integer n > 0, all edges within n timestamps before the test query
    timestamp are included.

    Parameters:
        all_data (np.ndarray): complete dataset (train/valid/test)
        current_ts (np.ndarray): test query timestamp
        window (int): time window used for rule application

    Returns:
        window_edges (dict): edges in the window for rule application
    """
    all_mask = (all_data[:, 0] == subject_object[0]) * (all_data[:, 2] == subject_object[1])

    if window > 0:
        mask = (all_data[:, 3] < current_ts) * (all_data[:, 3] >= current_ts - window)
        mask = mask * all_mask
        window_edges = all_data[mask]
    elif window == 0:
        mask = all_data[:, 3] < current_ts
        mask = mask * all_mask
        window_edges = all_data[mask]
    elif window == -1:
        window_edges = all_data
    
    return window_edges.tolist()

def sampling_edges(cur_edges, is_sample=False):
    """
    Edges are sampled based on timestamps.

    Parameters:
        cur_edges (list of np.ndarrays): edges that could constitute rule walks

    Returns:
        sampled_edges (np.ndarray): sampled edges
    """
    try:
        if not is_sample:
            return cur_edges
        cur_ts = cur_edges[:, 2]

        min_val = np.min(cur_ts)
        max_val = np.max(cur_ts)

        # Check that the minimum and maximum values are equal, and if so, manually specify a non-zero range
        if min_val == max_val:
            return cur_edges
        
        normalized_data = (cur_ts - min_val) / (max_val - min_val)
        # Avoid 0s and 1s
        normalized_data = normalized_data * (1 - 2e-10) + 1e-10
        samples = np.random.binomial(1, normalized_data).astype(bool)
        sample_edges = cur_edges[samples]

    except Exception as e:
        sample_edges = cur_edges
    
    return sample_edges

def match_body_relations(rule, edges, test_query, is_sample=False):
    """
    Find edges that could constitute walks (starting from the test query subject)
    that match the rule.
    First, find edges whose subject match the query subject and the relation matches
    the first relation in the rule body. Then, find edges whose subjects match the
    current targets and the relation matches the next relation in the rule body.
    Memory-efficient implementation.

    Parameters:
        rule (dict): rule from rules_dict
        edges (dict): edges for rule application
        test_query_sub (int): test query subject

    Returns:
        walk_edges (list of np.ndarrays): edges that could constitute rule walks
    """
    rels = rule["body_rels"]
    # Match query subject and first body relation
    try:
        test_query_sub = test_query[0]

        rel_edges = edges[rels[0]]
        mask = rel_edges[:, 0] == test_query_sub
        new_edges = rel_edges[mask]
        # make [subject, object, timestamp] 
        cur_edges = np.hstack((new_edges[:, 0:1], new_edges[:, 2:4]))
        sample_edges = sampling_edges(cur_edges, is_sample)
        cur_targets = np.array(list(set(sample_edges[:, 1])))

        walk_edges = []
        walk_edges.append(sample_edges)

        for i in range(1, len(rels)):
            try:
                rel_edges = edges[rels[i]]
                mask = np.any(rel_edges[:, 0] == cur_targets[:, None], axis=0)
                new_edges = rel_edges[mask]
                cur_edges = np.hstack((new_edges[:, 0:1], new_edges[:, 2:4]))
                sample_edges = sampling_edges(cur_edges)
                cur_targets = np.array(list(set(sample_edges[:, 1])))
                walk_edges.append(sample_edges) # [subject, object, timestamp]
            except Exception as e:
                walk_edges.append([])
                break
    except Exception as e:
        walk_edges = [[]]
    
    return walk_edges

def match_body_relations_complete(rule, edges, test_query_sub):
    """
    Find edges that could constitute walks (starting from the test query subject)
    that match the rule.
    First, find edges whose subject match the query subject and the relation matches
    the first relation in the rule body. Then, find edges whose subjects match the
    current targets and the relation the next relation in the rule body.

    Parameters:
        rule (dict): rule from rules_dict
        edges (dict): edges for rule application
        test_query_sub (int): test query subject

    Returns:
        walk_edges (list of np.ndarrays): edges that could constitute rule walks
    """
    rels = rule["body_rels"]
    # Match query subject and first body relation
    try:
        rel_edges = edges[rels[0]]
        mask = rel_edges[:, 0] == test_query_sub
        new_edges = rel_edges[mask]
        walk_edges = [new_edges]
        cur_targets = np.array(list(set(walk_edges[0][:, 2])))

        for i in range(1, len(rels)):
            # Match current targets and next body relation
            try:
                rel_edges = edges[rels[i]]
                mask = np.any(rel_edges[:, 0] == cur_targets[:, None], axis=0)
                new_edges = rel_edges[mask]
                walk_edges.append(new_edges)
                cur_targets = np.array(list(set(walk_edges[i][:, 2])))
            except KeyError:
                walk_edges.append([])
                break
    except KeyError:
        walk_edges = [[]]
    
    return walk_edges

def get_walks(rule, walk_edges, is_relax_time=False):
    """
    Get walks for a given rule. Take the time constraints into account.
    Memory-efficient implementation.

    Parameters:
        rule (dict): rule from rules_dict
        walk_edges (list of np.ndarrays): edges from match_body_relations

    Returns:
        rule_walks (pd.DataFrame): all walks matching the rule
    """
    df_edges = []
    df = pd.DataFrame(
        walk_edges[0],
        columns=["entity_"+str(0), "entity_"+str(1), "timestamp_"+str(0)],
        dtype=np.uint16,
    )
    if not rule["var_constraints"]:
        del df["entity_" + str(0)]
    df_edges.append(df)
    df = df[0:0] #  the original df DataFrame is cleared of all its data while retaining its column structure

    for i in range(1, len(walk_edges)):
        df = pd.DataFrame(
            walk_edges[i],
            columns=["entity_"+str(i), "entity_"+str(i+1), "timestamp_"+str(i)],
            dtype=np.uint16
        )
        df_edges.append(df)
        df = df[0:0]

    rule_walks = df_edges[0]
    df_edges[0] = df_edges[0][0:0]
    for i in range(1, len(df_edges)):
        rule_walks = pd.merge(rule_walks, df_edges[i], on=["entity_"+str(i)])
        if is_relax_time is False:
            rule_walks = rule_walks[
                rule_walks["timestamp_"+str(i-1)] <= rule_walks["timestamp_"+str(i)]
            ]
    
        if not rule["var_constraints"]:
            del rule_walks["entity_"+str(i)]
        df_edges[i] = df_edges[i][0:0]
    
    for i in range(1, len(rule["body_rels"])):
        del rule_walks["timestamp_"+str(i)]
    
    return rule_walks

def get_walks_end(rule, walk_edges, is_relax_time=False):
    """
    Get walks for a given rule. Take the time constraints into account.
    Memory-efficient implementation.

    Parameters:
        rule (dict): rule from rules_dict
        walk_edges (list of np.ndarrays): edges from match_body_relations

    Returns:
        rule_walks (pd.DataFrame): all walks matching the rule
    """
    df_edges = []
    df = pd.DataFrame(
        walk_edges[0],
        columns=["entity_"+str(0), "entity_"+str(1), "timestamp_"+str(0)],
        dtype=np.uint16,
    )
    if not rule["var_constraints"]:
        del df["entity_" + str(0)]
    
    df_edges.append(df)
    df = df[0:0]

    for i in range(1, len(walk_edges)):
        df = pd.DataFrame(
            walk_edges[i],
            columns=["entity_"+str(i), "entity_"+str(i+1), "timestamp_"+str(i)],
            dtype=np.uint16,
        )
        df_edges.append(df)
        df = df[0:0]

    rule_walks = df_edges[0]
    df_edges[0] = df_edges[0][0:0]
    for i in range(1, len(df_edges)):
        rule_walks = pd.merge(rule_walks, df_edges[i], on=["entity_"+str(i)])
        if is_relax_time is False:
            rule_walks = rule_walks[
                rule_walks["timestamp_"+str(i-1)] <= rule_walks["timestamp_"+str(i)]
            ]
        if not rule["var_constraints"]:
            del rule_walks["entity_"+str(i)]
        df_edges[i] = df_edges[i][0:0]
    
    for i in range(0, len(rule["body_rels"]) - 1):
        del rule_walks["timestamp_"+str(i)]
    
    return rule_walks

def get_walks_complete(rule, walk_edges):
    """
    Get complete walks for a given rule. Take the time constraints into account.

    Parameters:
        rule (dict): rule from rules_dict
        walk_edges (list of np.ndarrays): edges from match_body_relations

    Returns:
        rule_walks (pd.DataFrame): all walks matching the rule
    """

    df_edges = []
    df = pd.DataFrame(
        walk_edges[0],
        columns=["entity_"+str(0), "relation_"+str(0), "entity_"+str(1), "timestamp_"+str(0)],
        dtype=np.uint16,
    )
    df_edges.append(df)

    for i in range(1, len(walk_edges)):
        df =pd.DataFrame(
            walk_edges[i],
            columns=["entity_"+str(i), "relation_"+str(i), "entity_"+str(i+1), "timestamp_"+str(i)],
            dtype=np.uint16,
        )
        df_edges.append(df)

    rule_walks = df_edges[0]
    for i in range(1, len(df_edges)):
        rule_walks = pd.merge(rule_walks, df_edges[i], on=["entity_"+str(i)])
        rule_walks = rule_walks[
            rule_walks["timestamp_"+str(i-1)] <= rule_walks["timestamp_"+str(i)]
        ]
    
    return rule_walks

# def get_walks(rule, walk_edges, is_relax_time=False):  
#     """  
#     Optimized memory-efficient version with incremental merging and type optimization  
#     """  
#     # Initialize with first edge  
#     rule_walks = pd.DataFrame(  
#         walk_edges[0],  
#         columns=["entity_0", "entity_1", "timestamp_0"],  
#         dtype=np.uint32  # Use uint32 for better timestamp compatibility  
#     )  
    
#     if not rule["var_constraints"]:  
#         rule_walks.drop(columns=["entity_0"], inplace=True)  

#     # Process subsequent edges incrementally  
#     for i in range(1, len(walk_edges)):  
#         # Create and immediately merge next edge  
#         next_edge = pd.DataFrame(  
#             walk_edges[i],  
#             columns=[f"entity_{i}", f"entity_{i+1}", f"timestamp_{i}"],  
#             dtype=np.uint32  
#         )  
        
#         # Merge with existing walks and immediately clean up  
#         rule_walks = pd.merge(  
#             rule_walks,  
#             next_edge,  
#             on=[f"entity_{i}"],  
#             copy=False  # Reduce memory overhead  
#         )  
        
#         # Apply time constraint before next iteration  
#         if not is_relax_time:  
#             rule_walks = rule_walks.query(f"timestamp_{i-1} <= timestamp_{i}")  
        
#         # Clean up columns early  
#         if not rule["var_constraints"]:  
#             rule_walks.drop(columns=[f"entity_{i}"], inplace=True)  
        
#         # Explicit memory cleanup  
#         del next_edge  
#         if i > 1:  # Keep previous timestamp for constraints  
#             rule_walks.drop(columns=[f"timestamp_{i-2}"], inplace=True)  
    
#     # Final timestamp cleanup  
#     keep_timestamps = [f"timestamp_{len(walk_edges)-1}"]  
#     if len(walk_edges) > 1:  
#         keep_timestamps.append(f"timestamp_{len(walk_edges)-2}")  
    
#     return rule_walks[rule_walks.columns.difference(  
#         [c for c in rule_walks if c.startswith('timestamp_') and c not in keep_timestamps]  
#     )]  

# def get_walks_end(rule, walk_edges, is_relax_time=False):  
#     """  
#     Optimized version for end walks with early column dropping  
#     """  
#     # Reuse the main get_walks logic  
#     rule_walks = get_walks(rule, walk_edges, is_relax_time)  
    
#     # Different timestamp cleanup pattern  
#     ts_to_keep = [f"timestamp_{len(rule['body_rels'])-1}"]  
#     return rule_walks[rule_walks.columns.difference(  
#         [c for c in rule_walks if c.startswith('timestamp_') and c not in ts_to_keep]  
#     )]  

# def get_walks_complete(rule, walk_edges):  
#     """  
#     Memory-optimized complete walks with relation tracking  
#     """  
#     rule_walks = None  
#     for i, edges in enumerate(walk_edges):  
#         # Build edge with relation information  
#         edge_df = pd.DataFrame(  
#             edges,  
#             columns=[  
#                 f"entity_{i}",  
#                 f"relation_{i}",  
#                 f"entity_{i+1}",   
#                 f"timestamp_{i}"  
#             ],  
#             dtype=np.uint32  
#         )  
        
#         if rule_walks is None:  
#             rule_walks = edge_df  
#         else:  
#             # Merge incrementally  
#             rule_walks = pd.merge(  
#                 rule_walks,  
#                 edge_df,  
#                 on=[f"entity_{i}"],  
#                 copy=False  
#             )  
#             # Apply time constraint and clean previous timestamp  
#             rule_walks = rule_walks.query(f"timestamp_{i-1} <= timestamp_{i}")  
#             rule_walks.drop(columns=[f"timestamp_{i-1}"], inplace=True)  
        
#         # Clean up previous edge data  
#         if i > 0:  
#             rule_walks.drop(columns=[f"entity_{i}"], inplace=True)  
        
#         del edge_df  
    
#     return rule_walks

def check_var_constraints(var_constraints, rule_walks):
    """
    Check variable constraints of the rule.

    Parameters:
        var_constraints (list): variable constraints from the rule
        rule_walks (pd.DataFrame): all walks matching the rule

    Returns:
        rule_walks (pd.DataFrame): all walks matching the rule including the variable constraints
    """
    for constraint in var_constraints:
        for i in range(len(constraint) - 1):
            rule_walks = rule_walks[
                rule_walks["entity_"+str(constraint[i])]
                == rule_walks["entity_"+str(constraint[i+1])]
            ]
    
    return rule_walks

def get_candidates(rule, rule_walks, test_query_ts, cands_dict, score_func, args, dicts_idx, corre, is_return_ts,
                   eval_type, timestamp_dict):
    """
    Get from the walks that follow the rule the answer candidates.
    Add the confidence of the rule that leads to these candidates.

    Parameters:
        rule (dict): rule from rules_dict
        rule_walks (pd.DataFrame): rule walks (satisfying all constraints from the rule)
        test_query_ts (int): test query timestamp
        cands_dict (dict): candidates along with the confidences of the rules that generated these candidates
        score_func (function): function for calculating the candidate score
        args (list): arguments for the scoring function
        dicts_idx (list): indices for candidate dictionaries

    Returns:
        cands_dict (dict): updated candidates
    """
    max_entity = "entity_" + str(len(rule["body_rels"]))
    cands = set(rule_walks[max_entity])

    for cand in cands:
        cands_walk = rule_walks[rule_walks[max_entity] == cand]
        for s in dicts_idx:
            score = score_func(rule, cands_walk, test_query_ts, corre, *args[s]).astype(np.float32)

            try:
                cands_dict[s][cand].append(score)
            except KeyError:
                cands_dict[s][cand] = [score]
            
            if is_return_ts and eval_type != 'end':
                max_cands_ts = max(cands_walk["timestamp_0"])
                try:
                    timestamp_dict[s][cand].append(max_cands_ts)
                except KeyError:
                    timestamp_dict[s][cand] = [max_cands_ts]
            
            if is_return_ts and eval_type == 'end':
                max_cands_ts = max(cands_walk[f"timestamp_{len(rule['body_rels']) - 1}"])
                try:
                    timestamp_dict[s][cand].append(max_cands_ts)
                except KeyError:
                    timestamp_dict[s][cand] = [max_cands_ts]
    return cands_dict, timestamp_dict

def save_candidates(rules_file, dir_path, all_candidates, rule_lengths, window, score_func_str, all_timestamp):
    """
    Save the candidates.

    Parameters:
        rules_file (str): name of rules file
        dir_path (str): path to output directory
        all_candidates (dict): candidates for all test queries
        rule_lengths (list): rule lengths
        window (int): time window used for rule application
        score_func_str (str): scoring function

    Returns:
        None
    """
    all_candidates = {int(k): v for k,v in all_candidates.items()}
    for k in all_candidates:
        all_candidates[k] = {int(cand): v for cand, v in all_candidates[k].items()}
    filename = "{0}.json".format(
        rules_file
    )
    filename = filename.replace(" ", "")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(os.path.join(dir_path, filename), "w", encoding="utf-8") as f:
            json.dump(all_candidates, f, indent=4)
    
    # all_timestamp = {int(k): v for k, v in all_timestamp.items()}
    # for k in all_timestamp:
    #     all_timestamp[k] = {int(cand): v for cand, v in all_timestamp[k].items()}
    # filename = "{0}_cands_r{1}_w{2}_{3}_timestamp.json".format(
    #     rules_file[:-11], rule_lengths, window, score_func_str
    # )            
    # filename = filename.replace(" ", "")
    # with open(dir_path + filename, "w", encoding="utf-8") as f:
    #     json.dump(all_timestamp, f)

def verbalize_walk(walk, data):
    """
    Verbalize walk from rule application.

    Parameters:
        walk (pandas.core.series.Series): walk that matches the rule body from get_walks
        data (grapher.Grapher): graph data

    Returns:
        walk_str (str): verbalized walk
    """
    l = len(walk) // 3
    walk = walk.values.tolist()

    walk_str = data.id2entity[walk[0]] + '\t'
    for j in range(l):
        walk_str += data.id2relation[walk[3*j + 1]] +'\t'
        walk_str += data.id2entity[walk[3*j + 2]] + '\t'
        walk_str += data.id2ts[walk[3*j + 3]] + '\t'

    return walk_str[:-1]
            



        
        
