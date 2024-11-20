import os
import json
import itertools
import numpy as np
from collections import Counter

import copy
import re
import traceback
from utils import save_json_data, write_to_file

class Rule_Learner(object):
    def __init__(self, edges, id2relation, inv_relation_id, dataset):
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
        self.id2relation = id2relation
        self.inv_relation_id = inv_relation_id
        self.num_individual = 0
        self.num_shared = 0

        self.found_rules = []
        self.original_found_rules = []
        self.rule2confidence_dict = dict()
        self.rules_dict = dict()
        self.output_dir = "./stage_1_results/" + dataset + "/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    

    def define_var_constraints(self, entities):
        """
        Define variable constraints, i.e., state the indices of reoccurring entities in a walk.

        Parameters:
            entities (list): entities in the temporal walk

        Returns:
            var_constraints (list): list of indices for reoccurring entities
        """
        var_constraints = []
        for entity in set(entities):
            all_idx = [idx for idx, x in enumerate(entities) if x == entity]
            var_constraints.append(all_idx)
        var_constraints = [x for x in var_constraints if len(x) > 1]
        return sorted(var_constraints)
    

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
                mask = (next_edges[:, 0] == cur_node) * (next_edges[:, 3] >= cur_ts) # AND logic operator
            filtered_edges = next_edge[mask]

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
            body_var_constraints = self.define_var_constraints(body_ents_tss[::2])
            if body_var_constraints != var_constraints:
                sample_successful = False
        return sample_successful, body_ents_tss


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
            print(f"Can't find edges for {head_rel}.")
        for body in unique_bodies:
            mask = (
                (head_rel_edges[:, 0] == body[0]) *
                (head_rel_edges[:, 2] == body[-1]) *
                (head_rel_edges[:, 3] == body[-2])
            )
            if True in mask:
                rule_support += 1
        return rule_support


    def estimate_confidence(self, rule, num_samples=2000, is_relax_time=False):
        """
        Estimate the confidence of the rule by sampling bodies and checking the rule support.

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
            return 0, 0, 0
        
        if rule["head_rel"] not in self.edges:
            return 0, 0, 0
        
        all_bodies = []
        for _ in range(num_samples):
            sample_successful, body_ents_tss = self.sample_body(rule["body_rels"], rule["var_constraints"], is_relax_time)
            if sample_successful:
                all_bodies.append(body_ents_tss)
        all_bodies.sort()
        unique_bodies = list(x for x, _ in itertools.groupby(all_bodies))
        body_support = len(unique_bodies)

        confidence, rule_support = 0, 0
        if body_support:
            rule_support = self.calculate_rule_support(unique_bodies, rule["head_rel"])
            confidence = round(body_support / rule_support, 3)
        return confidence, rule_support, body_support


    def update_rules_dict(self, rule):
        """
        Update the rules if a new rule has been found.

        Parameters:
            rule (dict): generated rule from self.create_rule

        Returns:
            None
        """
        try:
            self.rules_dict[rule["head_rel"]].append(rule)
        except KeyError:
            self.rules_dict[rule["head_rel"]] = [rule]


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
            self.inv_relation_id[x] for x in walk["relations"][1:]
        ]
        rule["var_constraints"] = self.define_var_constraints(
            walk["entities"][1:][::-1]
        )

        if rule not in self.found_rules:
            self.found_rules.append(rule.copy())

            (
                rule["conf"],
                rule["rule_supp"],
                rule["body_supp"],
            ) = self.estimate_confidence(rule, is_relax_time=use_relax_time)

            rule["llm_confidence"] = confidence

            if rule["conf"] or confidence:
                self.update_rules_dict(rule)


    def create_rule_for_merge(self, walk, confidence=0, rule_without_confidence="", rules_var_dict=None,
                             is_merged=False, is_relax_time=False):
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

        Returns:
            rule (dict): created rule
        """
        rule = dict()
        rule["head_rel"] = int(walk["relations"][0])
        rule["body_rels"] = [
            self.inv_relation_id(x) for x in walk["relations"][1:][::-1]
        ]
        rule["var_constraints"] = self.define_var_constraints(walk["entities"][1:][::-1])

        if is_merged:
            if rules_var_dict.get(rule_without_confidence) is None:
                if rule not in self.found_rules:
                    self.found_rules.append(rule.copy())
                    (
                        rule["conf"],
                        rule["rule_supp"],
                        rule["body_supp"]
                    ) = self.estimate_confidence(rule)
                    rule["llm_confidence"] = confidence

                    if rule["conf"] or confidence:
                        self.num_individual += 1
                        self.update_rules_dict(rule)
            
            else:
                rule_var = rules_var_dict[rule_without_confidence]
                rule_var["llm_confidence"] = confidence
                temp_var = {}
                temp_var["head_rel"] = rule_var["head_rel"]
                temp_var["body_rels"] = rule_var["body_rels"]
                temp_var["var_constraints"] = rule_var["var_constraints"]
                if temp_var not in self.original_found_rules:
                    self.original_found_rules.append(temp_var.copy())
                    self.update_rules_dict(rule_var)
                    self.num_shared += 1
        else:
            if rule not in self.found_rules:
                self.found_rules.append(rule.copy())
                (
                    rule["conf"],
                    rule["rule_supp"],
                    rule["body_supp"]
                ) = self.estimate_confidence(rule, is_relax_time=is_relax_time)
                rule["llm_confidence"] = confidence
                if rule["conf"] = or confidence:
                    self.update_rules_dict(rule)
    

    def create_rule_for_merge_for_iteration(self, walk, llm_confidence=0, rule_without_confidence="",
                                            rules_var_dict=None, is_merged=False):
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

        Returns:
            rule (dict): created rule
        """
        rule = dict()
        rule["head_rel"] = int(walk["relations"][0])
        rule["body_rels"] = [
            self.inv_relation_id[x] for x in walk["relations"][1:][::-1]
        ]
        rule["var_constraints"] = self.define_var_constraints(walk["entities"][1:][::-1])

        rule_with_confidence = ""

        if is_merged:
            if rules_var_dict.get(rule_without_confidence) is None:
                if rule not in self.found_rules:
                    self.found_rules.append(rule.copy())
                    (
                        rule["conf"],
                        rule["rule_supp"],
                        rule["body_supp"]
                    ) = self.estimate_confidence(rule)

                    tuple_key = str(rule)
                    self.rule2confidence_dict[tuple_key] = rule["conf"]
                    rule_with_confidence = rule_without_confidence + "&" + str(rule["conf"])

                    rule["llm_confidence"] = llm_confidence

                    if rule["conf"] or llm_confidence:
                        self.num_individual += 1
                        self.update_rules_dict(rule)
                else:
                    tuple_key = tuple(rule)
                    confidence = self.rule2confidence_dict[tuple_key]
                    rule_with_confidence = rule_without_confidence + "&" + confidence

            else:
                rule_var = rules_var_dict[rule_without_confidence]
                rule_var["llm_confidence"] = llm_confidence
                temp_var = {}
                temp_var["head_rel"] = rule_var["head_rel"]
                temp_var["body_rels"] = rule_var["body_rels"]
                temp_var["var_constraints"] = rule_var["var_constraints"]
                if temp_var not in self.original_found_rules:
                    self.original_found_rules.append(temp_var.copy())
                    self.update_rules_dict(rule_var)
                    self.num_shared += 1
        else:
            if rule not in self.found_rules:
                tuple_key = str(rule)
                self.found_rules.append(rule.copy())
                (
                    rule["conf"],
                    rule["rule_supp"],
                    rule["body_supp"]
                ) = self.estimate_confidence(rule)

                self.rule2confidence_dict[tuple_key] = rule["conf"]
                rule_with_confidence = rule_without_confidence + "&" + str(rule["conf"])

                if rule["body_supp"] == 0:
                    rule["body_supp"] = 2
                
                rule["llm_confidence"] = llm_confidence

                if rule["conf"] or llm_confidence:
                    self.update_rules_dict(rule)
            else:
                tuple_key = str(rule)
                confidence = self.rule2confidence_dict[tuple_key]
                rule_with_confidence = rule_without_confidence + "&" + str(confidence)
        return rule_with_confidence
    

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
    

    def save_rules(self, dt, rule_lengths, num_walks, transition_distr, seed):
        """
        Save all rules.

        Parameters:
            dt (str): time now
            rule_lengths (list): rule lengths
            num_walks (int): number of walks
            transition_distr (str): transition distribution
            seed (int): random seed

        Returns:
            None
        """
        rules_dict = {int(k): v for k, v in self.rules_dict.items()}
        filename = f"{dt}_r{rule_lengths}_n{num_walks}_{transition_distr}_s{seed}_rules.json"
        filename = filename.replace(" ", "")
        with open(self.output_dir + filename, "w", encoding="utf-8") as f:
            json.dump(rules_dict, f, indent=4)
        
    
    def verbalize_rules(self):
        rules_str = ""
        rules_var = {}
        for rel in self.rules_dict:
            for rule in self.rules_dict[rel]:
                single_rule =


    def save_rules_verbalized(self, dt, rule_lengths, num_walks, transition_distr, seed, rel2idx, relation_regex):
        """
        Save all rules in a human-readable format.

        Parameters:
            dt (str): time now
            rule_lengths (list): rule lengths
            num_walks (int): number of walks
            transition_distr (str): transition distribution
            seed (int): random seed

        Returns:
            None
        """

        output_original_dir = os.path.join(self.output_dir, 'original/')
        os.makedirs(output_original_dir, exist_ok=True)

        rules_str, rules_var = self.v


def parse_rules_for_path(lines, relations, relation_regex):
    converted_rules = {}
    for line in lines:
        rule = line.strip()
        if not rule:
            continue
        temp_rule = re.sub(r'\s*<-\s*', '&', rule)
        regex_list = temp_rule.split('&')

        head = ""
        body_list = []
        for idx, regex_item in enumerate(regex_list):
            match = re.search(relation_regex, regex_item)
            if match:
                rel_name = match.group(1).strip()
                if rel_name not in relations:
                    raise ValueError(f"Not exist relation: {rel_name}.")
                if idx == 0:
                    head = rel_name
                    paths = converted_rules.setdefault(head, [])
                else:
                    body_list.append(rel_name)
        path = '|'.join(body_list)
        paths.append(path)
    return converted_rules


def parse_rules_for_name(lines, relations, relation_regex):
    rules_dict = {}
    for rule in lines:
        temp_rule = re.sub(r'\s*<-\s*', "&", rule)
        regex_list = temp_rule.split("&")
        match = re.search(relation_regex, regex_list[0])
        if match:
            head = match[1].strip()
            if head not in relations:
                raise ValueError(f"Not exist relation: {head}.")
        else:
            continue

        if head not in rules_dict:
            rules_dict[head] = []
        rules_dict[head].append(rule)
    return rules_dict


def parse_rules_for_id(rules, rel2idx, relation_regex):
    rules_dict = {}
    for rule in rules:
        temp_rule = re.sub(r'\s*<-\s*', "&", rule)
        regex_list = temp_rule.split("&")
        match = re.search(relation_regex, regex_list[0])
        if match:
            head = match[1].strip()
            if head not in rel2idx:
                raise ValueError(f"Not exist relation: {head}.")
        else:
            continue

        rule_id = rule2id(rule.strip