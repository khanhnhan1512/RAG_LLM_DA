import pandas as pd
import numpy as np
import os
import json
import glob
import pickle

def detect_changes_in_data(original_data, data):
    """
    """
    original_data = original_data.reset_index(drop=True)
    data = data.reset_index(drop=True)

    compare_cols = [col for col in original_data.columns if col != 'state']

    data = data.copy()
    for col in compare_cols:
        if col not in data.columns:
            data[col] = np.nan
    
    data = data[compare_cols]
    

def load_rules_data(path):
    """
    """
    data = pd.read_csv(path)
    return data

def load_facts_data(path):
    columns = ['subject', 'relation', 'object', 'timestamp']
    df = pd.read_csv(path, sep=' ', header=None, names=columns)
    return df

def main_load_data(data, original_data, db_directory):
    """
    """
    task_result = ''
    for i, collection in enumerate(data):
        path = data[collection]['path']
        data[collection]['data'] = pd.DataFrame()

        try:
            if collection == 'rules':
                data[collection]['data'] = load_rules_data(path)
            if collection == 'facts':
                data[collection]['data'] = load_facts_data(path)
        except Exception as e:
            print(f"Error while loading data from {path}: {e}")

        if not data[collection]['data'].empty:
            task_result += f"- Loaded '{collection}' data from {path}.\n"
        else:
            task_result += f"- Can not load '{collection}' data from {path}.\n"
    
    data = detect_changes_in_data(original_data, data)

    with open(os.path.join(db_directory, 'data.pkl'), 'wb') as f:
        pickle.dump(data, f)

    return data, task_result
