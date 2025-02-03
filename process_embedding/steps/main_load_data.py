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
    
    for col in compare_cols:
        data[col] = data[col].astype(original_data[col].dtype)
    
    for col in original_data[compare_cols].select_dtypes(include=['object']).columns:
        data[col] = data[col].str.strip()
        original_data[col] = original_data[col].str.strip()

    original_data['_merge_key'] = original_data[compare_cols].apply(lambda row: tuple(row), axis=1)
    data['_merge_key'] = data[compare_cols].apply(lambda row: tuple(row), axis=1)

    # Identify new and unchanged rows
    data['state'] = 'New'
    data.loc[data['_merge_key'].isin(original_data['_merge_key']), 'state'] = 'Unchanged'

    # Identify deleted rows
    deleted_rows = original_data[~original_data['_merge_key'].isin(data['_merge_key'])][compare_cols].copy()
    deleted_rows['state'] = 'Deleted'

    # Identify changed rows
    common_keys = original_data['_merge_key'].isin(data['_merge_key'])
    modified_rows = []
    for idx in original_data[common_keys].index:
        row_df1 = original_data.loc[idx, compare_cols]
        row_df2 = data.loc[data['_merge_key'] == original_data.loc[idx, '_merge_key'], compare_cols].iloc[0]
        if not row_df1.equals(row_df2):
            modified_rows.append(idx)
    data.loc[data['_merge_key'].isin(original_data.loc[modified_rows, '_merge_key']), 'state'] = 'Modified'

    # Combine all states
    result = pd.concat([data, deleted_rows], ignore_index=True)

    # Drop the merge key
    result = result.drop(columns='_merge_key')

    return result

def detect_changes_in_collection(original_data, data):
    for collection in data:
        if collection in original_data:
            data[collection]['data'] = detect_changes_in_data(original_data[collection]['data'], data[collection]['data'])
        else:
            data[collection]['data']['state'] = 'New'
    return data

def load_rules_data(path):
    """
    """
    data = pd.read_csv(path)
    return data

def load_facts_data(path):
    """
    """
    fact_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.inv')]
    df = pd.DataFrame()
    for f in fact_files:
        columns = ['subject', 'relation', 'object', 'timestamp']
        temp_df = pd.read_csv(f, sep='\t', header=None, names=columns)
        df = pd.concat([df, temp_df], ignore_index=True)
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
    
    data = detect_changes_in_collection(original_data, data)

    with open(os.path.join(db_directory, 'data.pkl'), 'wb') as f:
        pickle.dump(data, f)

    return data, task_result
