import pandas as pd
from utils import load_json_data

def add_attributes_for_facts(df, dataset, data_loader):
    """
    """
    transformed_relations = load_json_data(f'result/{dataset}/stage_1/transformed_relations.json')
    describes = []
    subject_id, object_id, relation_id, timestamp_id = [], [], [], []
    for _, row in df.iterrows():
        describes.append(row['subject'] + ' ' + transformed_relations[row['relation']] + ' to/with ' + row['object'] + f" on {row['timestamp']}")
        subject_id.append(data_loader.entity2id[row['subject']])
        object_id.append(data_loader.entity2id[row['object']])
        relation_id.append(data_loader.relation2id[row['relation']])
        timestamp_id.append(data_loader.ts2id[row['timestamp']])

    df['description'] = describes
    df['subject_id'] = subject_id
    df['object_id'] = object_id
    df['relation_id'] = relation_id
    df['timestamp_id'] = timestamp_id
    return df

def add_attribute(data, llm_instance, settings, data_loader):
    """
    """
    task_result = ''
    for i, collection in enumerate(data):
        try:
            df = data[collection]['data']
            if collection == 'facts':
                data[collection]['data'] = add_attributes_for_facts(df, settings['dataset'], data_loader)
            task_result += f'Added attributes for {collection}.\n'
        except Exception as e:
            task_result += f'Failed to add description for {collection}. Error: {e}\n'
            data[collection]['data'] = pd.DataFrame()
    return data, task_result