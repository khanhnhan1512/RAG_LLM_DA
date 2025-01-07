import pandas as pd
from utils import load_json_data

def add_description_for_facts(df, dataset):
    """
    """
    transformed_relations = load_json_data(f'result/{dataset}/stage_1/transformed_relations.json')
    describes = []
    for _, row in df.iterrows():
        describes.append(row['subject'] + ' ' + transformed_relations[row['relation']] + ' to/with ' + row['object'] + f" on {row['timestamp']}")
    df['description'] = describes
    return df

def add_attribute(data, llm_instance, settings):
    """
    """
    task_result = ''
    for i, collection in enumerate(data):
        try:
            df = data[collection]['data']
            if collection == 'facts':
                data[collection]['data'] = add_description_for_facts(df, settings['dataset'])
            task_result += f'Added description for {collection}.\n'
        except Exception as e:
            task_result += f'Failed to add description for {collection}. Error: {e}\n'
            data[collection]['data'] = pd.DataFrame()
    return data, task_result