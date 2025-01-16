import os
import json
import itertools
import numpy as np
import pandas as pd
from collections import Counter

import copy
import re
import traceback
from utils import save_json_data, write_to_file

class RuleLearner(object):
    def __init__(self, edges, relation2id, id2entity, id2relation, inv_relation_id, dataset, total_num_fact, output_dir):
        """
        Initialize rule learner object.

        Parameters:
            edges (dict): edges for each relation
            id2relation (dict): mapping of index to relation
            inv_relation_id (dict): mapping of relation to inverse relation
            dataset (str): dataset name

        Returns:
            None
        """

        self.edges = edges
        self.relation2id = relation2id
        self.id2entity = id2entity
        self.id2relation = id2relation
        self.inv_relation_id = inv_relation_id
        self.total_num_fact = total_num_fact
        self.num_individual = 0
        self.num_shared = 0
        self.num_original = 0

        self.found_rules = []
        self.rule2confidence_dict = {}
        self.original_found_rules = []
        self.rules_dict = dict()
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def create_rule(self, walk, confidence=0, use_relax_time=False):
        """
        Create a rule given a cyclic temporal random walk.
        The rule contains information about head relation, body relations,
        variable constraints, confidence, rule support, and body support.
        A rule is a dictionary with the content
        {"head_rel": int, "body_rels": list, "var_constraints": list,
         "conf": float, "rule_supp": int, "body_supp": int} 

        Parameters:
            walk (dict): cyclic temporal random walk
                         {"entities": list, "relations": list, "timestamps": list}
            confidence (float): confidence value
            use_relax_time (bool): whether the rule is created with relaxed time

        Returns:
            rule (dict): created rule
        """

        rule = dict()
        rule["head_rel"] = int(walk["relations"][0])
        rule["body_rels"] = [
            self.inv_relation_id[x] for x in walk["relations"][1:][::-1]
        ]
        rule["var_constraints"], _ = self.define_var_constraints(
            walk["entities"][1:][::-1]
        )

        rule["example_entities"] = [self.id2entity[entity_id] for entity_id in walk["entities"][1:][::-1]]
        rule["example"] = verbalize_example_rule(rule, self.id2relation)
        del rule["example_entities"]
        

        if rule not in self.found_rules:
            self.found_rules.append(rule.copy())
            (
                rule["conf"],
                rule["rule_supp_count"],
                rule["body_supp_count"],
                # rule["head_supp_count"],
                # rule["lift"],
                # rule["conviction"],
                # rule["kulczynski"],
                # rule["IR_score"]
            ) = self.estimate_metrics(rule, is_relax_time=use_relax_time)

            rule["llm_confidence"] = confidence

            if rule["conf"] or confidence:
                self.update_rules_dict(rule)

    def create_llm_rule(self, verbalized_rule, rule_regex, confidence=0, use_relax_time=False):
        """
        
        """
        walk = parse_verbalized_rule_to_walk(verbalized_rule, self.relation2id, self.inv_relation_id, rule_regex)
        rule = dict()
        rule["head_rel"] = int(walk["relations"][0])
        rule["body_rels"] = [
            self.inv_relation_id[x] for x in walk["relations"][1:][::-1]
        ]
        rule["var_constraints"], _ = self.define_var_constraints(
            walk["entities"][1:][::-1]
        )

        if rule not in self.found_rules:
            self.found_rules.append(rule.copy())
            (
                rule["conf"],
                rule["rule_supp_count"],
                rule["body_supp_count"],
                # rule["head_supp_count"],
                # rule["lift"],
                # rule["conviction"],
                # rule["kulczynski"],
                # rule["IR_score"]
            ) = self.estimate_metrics(rule, is_relax_time=use_relax_time)

            rule["llm_confidence"] = confidence

            if rule["conf"] or confidence:
                self.update_rules_dict(rule)

    def define_var_constraints(self, entities):
        """
        Define variable constraints, i.e., state the indices of reoccurring entities in a walk.

        Parameters:
            entities (list): entities in the temporal walk

        Returns:
            var_constraints (list): list of indices for reoccurring entities
        """

        entity_occurances = []
        for ent in set(entities):
            all_idx = [idx for idx, x in enumerate(entities) if x == ent]
            entity_occurances.append(all_idx)
        var_constraints = [x for x in entity_occurances if len(x) > 1]

        return sorted(var_constraints), entity_occurances

    def estimate_metrics(self, rule, num_samples=2000, is_relax_time=False):
        """
        Estimate the metrics of the rule by sampling bodies and checking the rule support.

        Parameters:
            rule (dict): rule
                         {"head_rel": int, "body_rels": list, "var_constraints": list}
            num_samples (int): number of samples

        Returns:
            confidence (float): confidence of the rule, rule_support/body_support
            rule_support (int): rule support
            body_support (int): body support
        """

        if any(body_rel not in self.edges for body_rel in rule["body_rels"]):
            return 0, 0, 0, 0, 0, 0, 0, 0

        if rule['head_rel'] not in self.edges:
            return 0, 0, 0, 0, 0, 0, 0, 0

        all_bodies = []
        for _ in range(num_samples):
            sample_successful, body_ents_tss = self.sample_body(
                rule["body_rels"], rule["var_constraints"], is_relax_time
            )
            if sample_successful:
                all_bodies.append(body_ents_tss)

        all_bodies.sort()
        unique_bodies = list(x for x, _ in itertools.groupby(all_bodies))
        body_support_count = len(unique_bodies)
        body_support = round(body_support_count / self.total_num_fact, 6)

        # confidence, rule_support, lift, conviction, kulczynski, IR_score = 0, 0, 0, 100, 0, 0
        confidence, rule_support = 0, 0
        # head_supp_count = self.calculate_head_supp(rule["head_rel"])
        # head_supp = round(head_supp_count / self.total_num_fact, 6)
        rule_support_count = self.calculate_rule_support(unique_bodies, rule["head_rel"])
        if body_support_count:
            rule_support = round(rule_support_count / self.total_num_fact, 6)
            confidence = round(rule_support / body_support, 6)

        
        # lift = round(confidence / head_supp, 6)
        # kulczynski = round(0.5 * (confidence + (rule_support/head_supp)), 6)
        # if confidence < 1:
        #     conviction = round((1 - head_supp) / (1 - confidence), 6)
        
        # IR_score = round(abs(body_support - head_supp)/(body_support + head_supp - rule_support), 6)

        # return confidence, rule_support_count, body_support_count, head_supp_count, lift, conviction, kulczynski, IR_score
        return confidence, rule_support_count, body_support_count

    def sample_body(self, body_rels, var_constraints, use_relax_time=False):
        """
        Sample a walk according to the rule body.
        The sequence of timesteps should be non-decreasing.

        Parameters:
            body_rels (list): relations in the rule body
            var_constraints (list): variable constraints for the entities
            use_relax_time (bool): whether to use relaxed time sampling

        Returns:
            sample_successful (bool): if a body has been successfully sampled
            body_ents_tss (list): entities and timestamps (alternately entity and timestamp)
                                  of the sampled body
        """

        sample_successful = True
        body_ents_tss = []
        cur_rel = body_rels[0]
        rel_edges = self.edges[cur_rel]
        next_edge = rel_edges[np.random.choice(len(rel_edges))]
        cur_ts = next_edge[3]
        cur_node = next_edge[2]
        body_ents_tss.append(next_edge[0])
        body_ents_tss.append(cur_ts)
        body_ents_tss.append(cur_node)

        for cur_rel in body_rels[1:]:
            next_edges = self.edges[cur_rel]
            if use_relax_time:
                mask = (next_edges[:, 0] == cur_node)
            else:
                mask = (next_edges[:, 0] == cur_node) * (next_edges[:, 3] >= cur_ts)

            filtered_edges = next_edges[mask]

            if len(filtered_edges):
                next_edge = filtered_edges[np.random.choice(len(filtered_edges))]
                cur_ts = next_edge[3]
                cur_node = next_edge[2]
                body_ents_tss.append(cur_ts)
                body_ents_tss.append(cur_node)
            else:
                sample_successful = False
                break

        if sample_successful and var_constraints:
            # Check variable constraints
            body_var_constraints, _ = self.define_var_constraints(body_ents_tss[::2])
            if body_var_constraints != var_constraints:
                sample_successful = False

        return sample_successful, body_ents_tss

    # def calculate_head_supp(self, head_rel):
    #     """
    #     Calculate the head support.
    #     Parameters:
    #         head_rel: relation in head part of a rule

    #     Returns:
    #         sample_successful (bool): if a body has been successfully sampled
    #         body_ents_tss (list): entities and timestamps (alternately entity and timestamp)
    #                               of the sampled body
    #     """
    #     if head_rel not in self.edges:
    #         return 0
    #     return len(self.edges[head_rel])

    def calculate_rule_support(self, unique_bodies, head_rel):
        """
        Calculate the rule support. Check for each body if there is a timestamp
        (larger than the timestamps in the rule body) for which the rule head holds.

        Parameters:
            unique_bodies (list): bodies from self.sample_body
            head_rel (int): head relation

        Returns:
            rule_support (int): rule support
        """

        rule_support = 0
        try:
            head_rel_edges = self.edges[head_rel]
        except Exception as e:
            print(head_rel)
        for body in unique_bodies:
            mask = (
                    (head_rel_edges[:, 0] == body[0])
                    * (head_rel_edges[:, 2] == body[-1])
                    * (head_rel_edges[:, 3] > body[-2])
            )

            if True in mask:
                rule_support += 1

        return rule_support

    def is_new_rule(self, new_rule):
        """
        Check if a rule is new or has been found before.

        Parameters:
            rule (dict): rule from self.create_rule

        Returns:
            is_new (bool): if the rule is new
        """

        key = new_rule['head_rel']
        for rule in self.rules_dict[key]:
            if new_rule["body_rels"] == rule["body_rels"]:
                return False
        return True

    def update_rules_dict(self, rule):
        """
        Update the rules if a new rule has been found.

        Parameters:
            rule (dict): generated rule from self.create_rule

        Returns:
            None
        """
        key = rule["head_rel"]
        if key in self.rules_dict:
            if self.is_new_rule(rule):
                self.rules_dict[rule["head_rel"]].append(rule)
        else:
            self.rules_dict[rule["head_rel"]] = [rule]

    def remove_low_quality_rules(self, min_confidence=0.25):
        """
        Remove rules with confidence lower than a given threshold.

        Parameters:
            min_confidence (float): minimum confidence

        Returns:
            None
        """

        for rel in self.rules_dict:
            self.rules_dict[rel] = [
                x for x in self.rules_dict[rel] if x["conf"] >= min_confidence
            ]
        
    def sort_rules_dict(self):
        """
        Sort the found rules for each head relation by decreasing confidence.

        Parameters:
            None

        Returns:
            None
        """

        for rel in self.rules_dict:
            self.rules_dict[rel] = sorted(
                self.rules_dict[rel], key=lambda x: x["conf"], reverse=True
            )

    def save_rules_csv(self, dt, rule_type, rule_lengths=0, num_walks=0, transition_distr=None, seed=None, metrics=["confidence_score"]):
        """
        Save all rules in a csv file.

        Parameters:
            dt (str): time now
            rule_lengths (list): rule lengths
            num_walks (int): number of walks
            transition_distr (str): transition distribution
            seed (int): random seed

        Returns:
            None
        """
        

        if rule_type == "random_walk":
            filename = "{0}_r{1}_n{2}_{3}_s{4}_{5}_random_rules.csv".format(
                dt, rule_lengths, num_walks, transition_distr, seed, rule_type
            )
            # full_columns = ["kulczynski", "IR_score", "lift_score", "conviction_score", "confidence_score", "rule_supp_count", "body_supp_count", "head_supp_count",
            #        "rule", "head_rel", "example"]
            # default_columns = ["rule_supp_count", "body_supp_count", "head_supp_count", "rule", "head_rel", "example"]
            full_columns = ["confidence_score", "rule_supp_count", "body_supp_count",
                   "rule", "head_rel", "example"]
            default_columns = ["rule_supp_count", "body_supp_count", "head_supp_count", "rule", "head_rel", "example"]

        elif rule_type == "llm":
            filename = "{0}_{1}_generated_llm_rules.csv".format(dt, rule_type)
            # full_columns = ["kulczynski", "IR_score", "lift_score", "conviction_score", "confidence_score", "rule_supp_count", "body_supp_count", "head_supp_count",
            #        "rule", "head_rel"]
            # default_columns = ["rule_supp_count", "body_supp_count", "head_supp_count", "rule", "head_rel"]
            full_columns = ["confidence_score", "rule_supp_count", "body_supp_count",
                   "rule", "head_rel"]
            default_columns = ["rule_supp_count", "body_supp_count", "head_supp_count", "rule", "head_rel"]
            
        filename = filename.replace(" ", "")
        output_path = self.output_dir + filename
        
        df = pd.DataFrame(columns=full_columns)
        entries = []

        for rel in self.rules_dict:
            for rule in self.rules_dict[rel]:
                rule_str = verbalize_rule(rule, self.id2relation, rule_type)
                entry = rule_str.split("\t")
                entries.append(entry)
        
        df = pd.concat([df, pd.DataFrame(entries, columns=full_columns)], ignore_index=True)

        # Convert relevant columns to numeric types
        df['rule_supp_count'] = pd.to_numeric(df['rule_supp_count'], errors='coerce')
        df['body_supp_count'] = pd.to_numeric(df['body_supp_count'], errors='coerce')

        # Step 1: Calculate head_supp_count
        df['head_supp_count'] = df.groupby('head_rel')['rule_supp_count'].transform('sum')

        # Step 2: Calculate additional metrics
        total_num_fact = self.total_num_fact

        df['kulczynski'] = 0.0
        df['IR_score'] = 0.0
        df['lift_score'] = 0.0
        df['conviction_score'] = 100.0  # Default conviction score

        for index, row in df.iterrows():
            rule_supp = row['rule_supp_count']
            body_supp = row['body_supp_count']
            head_supp = row['head_supp_count']

            if pd.isna(body_supp) or pd.isna(head_supp) or body_supp == 0 or head_supp == 0:
                continue

            confidence = round(rule_supp / body_supp, 6) if body_supp != 0 else 0
            head_supp_ratio = round(head_supp / total_num_fact, 6)
            rule_supp_ratio = round(rule_supp / total_num_fact, 6)

            kulczynski = round(0.5 * (confidence + (rule_supp_ratio / head_supp_ratio)), 6)
            IR_score = round(abs(body_supp - head_supp) / (body_supp + head_supp - rule_supp), 6) if (body_supp + head_supp - rule_supp) != 0 else 0
            lift_score = round(confidence / head_supp_ratio, 6) if head_supp_ratio != 0 else 0
            
            # Update conviction score only if confidence < 1
            if confidence < 1:
                conviction_score = round((1 - head_supp_ratio) / (1 - confidence), 6)
                df.at[index, 'conviction_score'] = conviction_score

            # Assign calculated values to the DataFrame
            df.at[index, 'kulczynski'] = kulczynski
            df.at[index, 'IR_score'] = IR_score
            df.at[index, 'lift_score'] = lift_score

        columns_to_keep = metrics + default_columns

        df = df[columns_to_keep]
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Rules have been saved to {output_path}")

    def rules_statistics(self):
        """
        Show statistics of the rules.

        Parameters:
            rules_dict (dict): rules

        Returns:
            None
        """

        print(
            "Number of relations with rules: ", len(self.rules_dict)
        )  # Including inverse relations
        print("Total number of rules: ", sum([len(v) for k, v in self.rules_dict.items()]))

        lengths = []
        for rel in self.rules_dict:
            lengths += [len(x["body_rels"]) for x in self.rules_dict[rel]]
        rule_lengths = [(k, v) for k, v in Counter(lengths).items()]
        print("Number of rules by length: ", sorted(rule_lengths))

def verbalize_rule(rule, id2relation, rule_type):
    """
    Verbalize the rule to be in a human-readable format.

    Parameters:
        rule (dict): rule from Rule_Learner.create_rule
        id2relation (dict): mapping of index to relation

    Returns:
        rule_str (str): human-readable rule
    """

    if rule["var_constraints"]:
        var_constraints = rule["var_constraints"]
        constraints = [x for sublist in var_constraints for x in sublist]
        for i in range(len(rule["body_rels"]) + 1):
            if i not in constraints:
                var_constraints.append([i])
        var_constraints = sorted(var_constraints)
    else:
        var_constraints = [[x] for x in range(len(rule["body_rels"]) + 1)]

    # rule_str = "{0:8.6f}\t{1:8.6f}\t{2:8.6f}\t{3:8.6f}\t{4:8.6f}\t{5:4}\t{6:4}\t{7:4}\t{8}(X0,X{9},T{10})<-"
    rule_str = "{0:8.6f}\t{1:4}\t{2:4}\t{3}(X0,X{4},T{5})<-"
    obj_idx = [
        idx
        for idx in range(len(var_constraints))
        if len(rule["body_rels"]) in var_constraints[idx]
    ][0]
    rule_str = rule_str.format(
        # rule["kulczynski"],
        # rule["IR_score"],
        # rule["lift"],
        # rule["conviction"],
        rule["conf"],
        rule["rule_supp_count"],
        rule["body_supp_count"],
        # rule["head_supp_count"],
        id2relation[rule["head_rel"]],
        obj_idx,
        len(rule["body_rels"]),
    )

    for i in range(len(rule["body_rels"])):
        sub_idx = [
            idx for idx in range(len(var_constraints)) if i in var_constraints[idx]
        ][0]
        obj_idx = [
            idx for idx in range(len(var_constraints)) if i + 1 in var_constraints[idx]
        ][0]
        rule_str += "{0}(X{1},X{2},T{3})&".format(
            id2relation[rule["body_rels"][i]], sub_idx, obj_idx, i
        )

    rule_str = rule_str[:-1]
    if rule_type == "random_walk":
        rule_str += f"\t{id2relation[rule['head_rel']]}\t{rule['example']}"
    elif rule_type == "llm":
        rule_str += f"\t{id2relation[rule['head_rel']]}"
    
    return rule_str

def verbalize_example_rule(rule, id2relation):
    example_str = "{0}({1},{2},T{3})<-".format(
        id2relation[rule["head_rel"]],
        rule["example_entities"][0],
        rule["example_entities"][-1],
        len(rule["body_rels"])
    )
    for i in range(len(rule["body_rels"])):
        sub_idx = i
        obj_idx = i + 1
        example_str += f"{id2relation[rule['body_rels'][i]]}({rule['example_entities'][sub_idx]},{rule['example_entities'][obj_idx]},T{i})&"
    return example_str[:-1]

def parse_verbalized_rule_to_walk(verbalized_rule, relation2id, inverse_rel_idx, rule_regex):
    """
    Parse a verbalized rule string to a temporal walk.

    Parameters:
        verbalized_rule (str): verbalized rule string

    Returns:
        walk (dict): parsed temporal walk
                        {"entities": list, "relations": list}
    """
    walk = {
        "entities": [],
        "relations": [],
    }
    # print(verbalized_rule)
    head, body = verbalized_rule.split("<-")
    # parse to get head relation and entities
    head_match = re.search(rule_regex, head)
    walk["entities"].append(head_match.groups()[1])
    walk["relations"].append(relation2id[head_match.groups()[0].strip()])

    parts = body.split("&")
    for part in parts[::-1]:
        match = re.search(rule_regex, part)
        if match:
            walk["relations"].append(inverse_rel_idx[relation2id[match.groups()[0].strip()]])
            walk["entities"].append(match.groups()[2])

    walk["entities"].append(walk["entities"][0])
    return walk

