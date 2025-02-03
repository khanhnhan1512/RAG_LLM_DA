from process_embedding.custom_embedding_function import build_documents

def main_prepare_vector_db(data, llm_instance):
    task_result = ''
    for i, collection in enumerate(data):
        try:
            if not data[collection]['data'].empty:
                new_or_modified_data = data[collection]['data'][data[collection]['data']['state'].isin(['New', 'Modified'])]
                deleted_or_modified = data[collection]['data'][data[collection]['data']['state'].isin(['Deleted', 'Modified'])].index.astype(str).tolist()
                data[collection]['data'] = {'New_Documents': build_documents(new_or_modified_data, collection), 'Deleting_Documents': deleted_or_modified}

            else:
                data[collection]['data'] = {'New_Documents': [], 'Deleting_Documents': []}
        except Exception as e:
            print(f"An error occurred while transforming the data as documents: {e}")
            data[collection]['data'] = {'New_Documents': [], 'Deleting_Documents': []}

        # Success
        if data[collection]['data']:
            task_result += f"- {collection} data transformed successfully.\n"
        else:
            task_result += f"- {collection} data could not be transformed.\n"
    
    return data, task_result