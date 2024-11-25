import os
import numpy as np

from stages.stage_1_learn_rules_from_data.utils import load_json_data


class DataLoader():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.entity2id = load_json_data(os.path.join(data_dir, 'entity2id.json'))
        self.relation2id = load_json_data(os.path.join(data_dir, 'relation2id.json'))
        self.ts2id = load_json_data(os.path.join(data_dir, 'ts2id.json'))
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.id2ts = {v: k for k, v in self.ts2id.items()}
        self.inverse_rel_idx = dict()
        for i in range(len(self.relation2id)):
            self.inverse_rel_idx[i] = len(self.relation2id) + i
        for i in range(len(self.relation2id), 2 * len(self.relation2id)):
            self.inverse_rel_idx[i] = i % len(self.relation2id)
        self.add_inverse_relation()
        self.train_data_idx, self.train_data_text = self.load_fact(os.path.join(data_dir, 'train.txt'))
        self.valid_data_idx, self.valid_data_text = self.load_fact(os.path.join(data_dir, 'valid.txt'))
        self.test_data_idx, self.test_data_text = self.load_fact(os.path.join(data_dir, 'test.txt'))
        

    def add_inverse_relation(self):
        idx = len(self.relation2id)
        for relation in self.relation2id.copy():
            self.relation2id["inv_" + relation] = idx
            idx += 1

    def split_quad(self, quads):
        result = []
        for quad in quads:
            result.append(quad.strip().split('\t'))
        return result

    def map_to_idx(self, quads):
        subs = [self.entity2id[quad[0]] for quad in quads]
        rels = [self.relation2id[quad[1]] for quad in quads]
        objs = [self.entity2id[quad[2]] for quad in quads]
        ts = [self.ts2id[quad[3]] for quad in quads]
        return np.column_stack((subs, rels, objs, ts))
    
    def add_inverse_fact(self, quads_idx, quads_text):
        subs = quads_idx[:, 2]
        inverse_rels = [self.inverse_rel_idx[rel] for rel in quads_idx[:, 1]]
        objs = quads_idx[:, 0]
        ts = quads_idx[:, 3]
        inv_quads_idx = np.column_stack((subs, inverse_rels, objs, ts))
        quads_idx = np.vstack((quads_idx, inv_quads_idx))

        inverse_quads_text = [[quads_text[i][2], "inv_" + quads_text[i][1], quads_text[i][0], quads_text[i][3]] for i in range(len(quads_text))]
        quads_text.extend(inverse_quads_text)
        return quads_idx, quads_text
    
    def load_fact(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                quads = f.readlines()
                quads_text = self.split_quad(quads)
                quads_idx = self.map_to_idx(quads_text)
                quads_idx, quads_text = self.add_inverse_fact(quads_idx, quads_text)
                return quads_idx, quads_text
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            return None, None

    