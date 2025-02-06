import sys
import os
import shutil
import pickle
import re
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stages.stage_1_learn_rules_from_data.data_loader import DataLoader

from openai_llm.llm_init import LLM_Model
from process_embedding.process_data import Process
from utils import load_json_data, copy__file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets', help='path to the dataset')
    parser.add_argument('--dataset', type=str, default='YAGO', help='name of the dataset')
    parser = vars(parser.parse_args())
    return parser

def load_old_embedding_data(settings):
    source_dir_path = settings["output_vector_database_load"]
    try:
        os.makedirs(source_dir_path)
    except FileExistsError:
        pass

    destination_dir_path = settings["output_vector_database_build"]

    copy__file(source_dir_path, destination_dir_path)

    try:
        with open(os.path.join(source_dir_path, 'data.pkl'), 'rb') as f:
            original_data = pickle.load(f)
        print(f"Loaded existing reference points from vector database in {source_dir_path}")
    except:
        print("No existing reference points found. Creating new reference points.")
        original_data = {}
    
    return original_data


def main_process_embedding():
    args = parse_args()
    data_path = os.path.join(args['data_dir'], args['dataset'])
    settings = load_json_data('config/data_embedding.json')
    for k in settings:
        settings[k] = re.sub(r'\bdataset\b', args['dataset'], settings[k])
    original_data = load_old_embedding_data(settings)
    data_loader = DataLoader(data_path)
    llm_instance = LLM_Model()
    process = Process(llm_instance, settings, original_data, data_loader)
    process.main()

if __name__ == '__main__':
    main_process_embedding()