import os
import json
import itertools
import numpy as np
import pandas as pd
from collections import Counter

import copy
import re
import traceback
from stages.stage_1_learn_rules_from_data.utils import save_json_data, write_to_file

class RuleLearner(object):
    def __init__(self, edges, id2relation, inv_relation_id, dataset, total_num_fact):
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
        self.total_num_fact = total_num_fact
        self.num_individual = 0
        self.num_shared = 0
        self.num_original = 0

        self.found_rules = []
        self.rule2confidence_dict = {}
        self.original_found_rules = []
        self.rules_dict = dict()
        self.output_dir = "./result/" + dataset + "/stage_1/"
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
        rule["var_constraints"] = self.define_var_constraints(
            walk["entities"][1:][::-1]
        )

        if rule not in self.found_rules:
            self.found_rules.append(rule.copy())
            (
                rule["conf"],
                rule["rule_supp"],
                rule["body_supp"],
                rule["lift"],
                rule["conviction"]
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

        var_constraints = []
        for ent in set(entities):
            all_idx = [idx for idx, x in enumerate(entities) if x == ent]
            var_constraints.append(all_idx)
        var_constraints = [x for x in var_constraints if len(x) > 1]

        return sorted(var_constraints)

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
            return 0, 0, 0

        if rule['head_rel'] not in self.edges:
            return 0, 0, 0

        all_bodies = []
        for _ in range(num_samples):
            sample_successful, body_ents_tss = self.sample_body(
                rule["body_rels"], rule["var_constraints"], is_relax_time
            )
            if sample_successful:
                all_bodies.append(body_ents_tss)

        all_bodies.sort()
        unique_bodies = list(x for x, _ in itertools.groupby(all_bodies))
        body_support = len(unique_bodies) / self.total_num_fact

        confidence, rule_support, lift, conviction = 0, 0, 0, 0
        head_supp = self.head_supp(rule["head_rel"], is_relax_time) / self.total_num_fact
        if body_support:
            rule_support = self.calculate_rule_support(unique_bodies, rule["head_rel"]) / self.total_num_fact
            confidence = round(rule_support / body_support, 6)

        if head_supp and body_support:
            lift = round(rule_support / (head_supp * body_support), 6)
        if confidence < 1:
            conviction = round((1 - (head_supp/self.total_num_fact)) / (1 - (confidence/self.total_num_fact)), 6)

        return confidence, rule_support, body_support, lift, conviction

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
            body_var_constraints = self.define_var_constraints(body_ents_tss[::2])
            if body_var_constraints != var_constraints:
                sample_successful = False

        return sample_successful, body_ents_tss

    def head_supp(self, head_rel, use_relax_time=False):
        """
        Calculate the head support.
        Parameters:
            head_rel: relation in head part of a rule

        Returns:
            sample_successful (bool): if a body has been successfully sampled
            body_ents_tss (list): entities and timestamps (alternately entity and timestamp)
                                  of the sampled body
        """
        if head_rel not in self.edges:
            return 0
        return len(self.edges[head_rel])

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

    def save_rules_csv(self, dt, rule_lengths, num_walks, transition_distr, seed):
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
        filename = "{0}_r{1}_n{2}_{3}_s{4}_rules.csv".format(
            dt, rule_lengths, num_walks, transition_distr, seed
        )
        filename = filename.replace(" ", "")
        output_path = self.output_dir + filename

        columns = ["lift_score", "conviction_score", "confidence_score", "rule_support", "body_support", "rule", "head_rel"]
        df = pd.DataFrame(columns=columns)
        entries = []

        for rel in self.rules_dict:
            for rule in self.rules_dict[rel]:
                rule_str = verbalize_rule(rule, self.id2relation)
                entry = rule_str.split("\t") + [self.id2relation[rule["head_rel"]]]
                entries.append(entry)
        df = pd.concat([df, pd.DataFrame(entries, columns=columns)], ignore_index=True)
        df.to_csv(output_path, index=False)
        print(f"Rules have been saved to {output_path}")
        
    def generate_filename(self, dt, rule_lengths, num_walks, transition_distr, seed, suffix):
        filename = f"{dt}_r{rule_lengths}_n{num_walks}_{transition_distr}_s{seed}_{suffix}"
        return filename.replace(" ", "")
    
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

        rules_str, rules_var = self.verbalize_rules()
        save_json_data(rules_var, output_original_dir + "rules_var.json")

        filename = self.generate_filename(dt, rule_lengths, num_walks, transition_distr, seed, "rules.txt")
        write_to_file(rules_str, self.output_dir + filename)

        original_rule_txt = self.output_dir + filename
        remove_filename = self.generate_filename(dt, rule_lengths, num_walks, transition_distr, seed,
                                                 "remove_rules.txt")

        rule_id_content = self.remove_first_three_columns(self.output_dir + filename, self.output_dir + remove_filename)

        self.parse_and_save_rules(remove_filename, list(rel2idx.keys()), relation_regex, 'closed_rel_paths.jsonl')
        self.parse_and_save_rules_with_names(remove_filename, rel2idx, relation_regex, 'rules_name.json',
                                             rule_id_content)
        self.parse_and_save_rules_with_ids(rule_id_content, rel2idx, relation_regex, 'rules_id.json')

        self.save_rule_name_with_confidence(original_rule_txt, relation_regex,
                                       self.output_dir + 'relation_name_with_confidence.json', list(rel2idx.keys()))

    def remove_first_three_columns(self, input_path, output_path):
        rule_id_content = []
        with open(input_path, 'r') as input_file, open(output_path, 'w', encoding="utf-8") as output_file:
            for line in input_file:
                columns = line.split()
                new_line = ' '.join(columns[3:])
                new_line_for_rule_id = ' '.join(columns[3:]) + '&' + columns[0] + '\n'
                rule_id_content.append(new_line_for_rule_id)
                output_file.write(new_line + '\n')
        return rule_id_content

    def parse_and_save_rules(self, remove_filename, keys, relation_regex, output_filename):
        output_file_path = os.path.join(self.output_dir, output_filename)
        with open(self.output_dir + remove_filename, 'r') as file:
            lines = file.readlines()
            converted_rules = parse_rules_for_path(lines, keys, relation_regex)
        with open(output_file_path, 'w') as file:
            for head, paths in converted_rules.items():
                json.dump({"head": head, "paths": paths}, file)
                file.write('\n')
        print(f'Rules have been converted and saved to {output_file_path}')
        return converted_rules

    def parse_and_save_rules_with_names(self, remove_filename, rel2idx, relation_regex, output_filename,
                                        rule_id_content):
        input_file_path = os.path.join(self.output_dir, remove_filename)
        output_file_path = os.path.join(self.output_dir, output_filename)
        with open(input_file_path, 'r') as file:
            rules_content = file.readlines()
            rules_name_dict = parse_rules_for_name(rules_content, list(rel2idx.keys()), relation_regex)
        with open(output_file_path, 'w') as file:
            json.dump(rules_name_dict, file, indent=4)
        print(f'Rules have been converted and saved to {output_file_path}')

    def parse_and_save_rules_with_ids(self, rule_id_content, rel2idx, relation_regex, output_filename):
        output_file_path = os.path.join(self.output_dir, output_filename)
        rules_id_dict = parse_rules_for_id(rule_id_content, rel2idx, relation_regex)
        with open(output_file_path, 'w') as file:
            json.dump(rules_id_dict, file, indent=4)
        print(f'Rules have been converted and saved to {output_file_path}')

    def save_rule_name_with_confidence(self, file_path, relation_regex, out_file_path, relations):
        rules_dict = {}
        with open(file_path, 'r') as fin:
            rules = fin.readlines()
            for rule in rules:
                # Split the string by spaces to get the columns
                columns = rule.split()

                # Extract the first and fourth columns
                first_column = columns[0]
                fourth_column = ''.join(columns[3:])
                output = f"{fourth_column}&{first_column}"

                regrex_list = fourth_column.split('<-')
                match = re.search(relation_regex, regrex_list[0])
                if match:
                    head = match[1].strip()
                    if head not in relations:
                        raise ValueError(f"Not exist relation:{head}")
                else:
                    continue

                if head not in rules_dict:
                    rules_dict[head] = []
                rules_dict[head].append(output)
        save_json_data(rules_dict, out_file_path)

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


# def describe_rules(rules, llm_instance):
#         """
#         Describe the rules in natural language.

#         Parameters:
#             None

#         Returns:
#             None
#         """
#         user_query = "Please help me to describe these temporal rules in natural language."
#         user_msg_content = f'''
#                         Here is the user query: {user_query}
#                         Here is the rules that need to be verbalized:
#                         {rules}
#                         '''

def verbalize_rule(rule, id2relation):
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

    rule_str = "{0:8.6f}\t{1:8.6f}\t{2:8.6f}\t{3:4}\t{4:4}\t{5}(X0,X{6},T{7})<-"
    obj_idx = [
        idx
        for idx in range(len(var_constraints))
        if len(rule["body_rels"]) in var_constraints[idx]
    ][0]
    rule_str = rule_str.format(
        rule["lift"],
        rule["conviction"],
        rule["conf"],
        rule["rule_supp"],
        rule["body_supp"],
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

    return rule_str[:-1]

def parse_rules_for_path(lines, relations, relation_regex):
    converted_rules = {}
    for line in lines:
        rule = line.strip()
        if not rule:
            continue
        temp_rule = re.sub(r'\s*<-\s*', '&', rule)
        regrex_list = temp_rule.split('&')

        head = ""
        body_list = []
        for idx, regrex_item in enumerate(regrex_list):
            match = re.search(relation_regex, regrex_item)
            if match:
                rel_name = match.group(1).strip()
                if rel_name not in relations:
                    raise ValueError(f"Not exist relation:{rel_name}")
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
        temp_rule = re.sub(r'\s*<-\s*', '&', rule)
        regrex_list = temp_rule.split('&')
        match = re.search(relation_regex, regrex_list[0])
        if match:
            head = match[1].strip()
            if head not in relations:
                raise ValueError(f"Not exist relation:{head}")
        else:
            continue

        if head not in rules_dict:
            rules_dict[head] = []
        rules_dict[head].append(rule)

    return rules_dict


def parse_rules_for_id(rules, rel2idx, relation_regex):
    rules_dict = {}
    for rule in rules:
        temp_rule = re.sub(r'\s*<-\s*', '&', rule)
        regrex_list = temp_rule.split('&')
        match = re.search(relation_regex, regrex_list[0])
        if match:
            head = match[1].strip()
            if head not in rel2idx:
                raise ValueError(f"Relation '{head}' not found in rel2idx")
        else:
            continue

        rule_id = rule2id(rule.rsplit('&', 1)[0], rel2idx, relation_regex)
        rule_id = rule_id + '&' + rule.rsplit('&', 1)[1].strip()
        rules_dict.setdefault(head, []).append(rule_id)
    return rules_dict


def rule2id(rule, relation2id, relation_regex):
    temp_rule = copy.deepcopy(rule)
    temp_rule = re.sub(r'\s*<-\s*', '&', temp_rule)
    temp_rule = temp_rule.split('&')
    rule2id_str = ""

    try:
        for idx, _ in enumerate(temp_rule):
            match = re.search(relation_regex, temp_rule[idx])
            rel_name = match[1].strip()
            subject = match[2].strip()
            object = match[3].strip()
            timestamp = match[4].strip()
            rel_id = relation2id[rel_name]
            full_id = f"{rel_id}({subject},{object},{timestamp})"
            if idx == 0:
                full_id = f"{full_id}<-"
            else:
                full_id = f"{full_id}&"

            rule2id_str += f"{full_id}"
    except KeyError as keyerror:
        # 捕获异常并打印调用栈信息
        traceback.print_exc()
        raise ValueError(f"KeyError: {keyerror}")

    except Exception as e:
        raise ValueError(f"An error occurred: {rule}")

    return rule2id_str[:-1]





