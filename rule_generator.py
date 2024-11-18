import argparse
import os

from data_loader import DataLoader
from utils import load_learn_data, relation_similarity
from openai_llm.llm_init import LLM_Model

def main(args):
    data_path = args['data_path']
    dataset = args['dataset']
    output_path = args['output_path']
    data_dir = os.path.join(data_path, dataset)

    data_loader = DataLoader(data_dir)
    llm_instance = LLM_Model()
    relation_similarity(llm_instance, list(data_loader.relation2id.keys()), output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets', help='path to the dataset')
    parser.add_argument('--dataset', "-d", type=str, default='icews14', help='dataset name')
    parser.add_argument('--output_path', type=str, default='rules', help='path to save the rules')
    parser.add_argument('--seed', '--s', type=int, default=42, help='random seed')
    parser = vars(parser.parse_args())
    main(parser)
