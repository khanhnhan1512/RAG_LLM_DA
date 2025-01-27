import time
import os
import pickle
from langchain_chroma import Chroma
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.vectorstores.utils import filter_complex_metadata
from tqdm import tqdm
from process_embedding.custom_embedding_function import CustomEmbeddingFunction

def add_documents_to_db(db, documents):
    """
    """
    retry = 0
    retry_max = 3
    if documents:
        while retry < retry_max:
            try:
                db.add_documents(filter_complex_metadata(documents))
                break
            except Exception as e:
                retry += 1
                print(f"Error while adding documents to database: {e}. Retrying...({retry}/{retry_max})")
                time.sleep(5)

def delete_documents_from_db(db, delete_ids):
    """
    """
    retry = 0
    retry_max = 3
    while retry < retry_max:
        try:
            db.delete(ids=delete_ids)
            break
        except Exception as e:
            retry += 1
            print(f"Error while deleting documents from database: {e}. Retrying...({retry}/{retry_max})")
            time.sleep(5)

def verification_embedding_step(db, data):
    """
    Its very important to verify if 'New'/'Modified' elements are part of the vector db. They should be embedded and added to the vector db.
    'Deleted' should be deleted from the vector db.
    """
    # Retrieve IDs from the db
    ids_in_db = set(db.get()['ids'])

    # Ensure that 'ids_in_db' contains string representation of IDs
    ids_in_db = set(map(str, ids_in_db))

    # Convert Df indices to strings for consistent comparison
    data.index = data.index.map(str)

    # For 'Deleted' entries, check if their indices are still in the db IDs
    deleted_mask = data['state'] == 'Deleted'
    deleted_indices_in_db = data[deleted_mask].index.intersection(ids_in_db)

    # Update the 'state' of entries that were supposed to be deleted but are in the database IDs
    data.loc[deleted_indices_in_db, 'state'] = 'Unchanged'

    # For 'New' and 'Modified' entries, check if their indices are in the db IDs
    new_modified_mask = data['state'].isin(['New', 'Modified'])
    new_modified_indices_not_in_db = data[new_modified_mask].index.difference(ids_in_db)

    # Handle cases where 'New'/'Modified' entries are not in the db
    if not new_modified_indices_not_in_db.empty:
        print("The following 'New' or 'Modified' entries are missing from the db IDs:")
        print(new_modified_indices_not_in_db.tolist())
        data.loc[new_modified_indices_not_in_db, 'state'] = 'Error'
    
    return data

def main_create_vector_db(data, llm_instance, db_directory):
    """
    """
    task_result = ''
    db = None

    with open(os.path.join(db_directory, 'data.pkl'), 'rb') as f:
        reference_data = pickle.load(f)
    try:
        os.mkdir(db_directory)
    except FileExistsError:
        pass

    for _, collection in enumerate(data):
        documents = data[collection]['data']['New_Documents']
        ids_to_delete = data[collection]['data']['Deleting_Documents']

        collection_persist_directory = os.path.join(db_directory, collection)
        try:
            os.mkdir(collection_persist_directory)
        except FileExistsError:
            pass

        db = Chroma(persist_directory=collection_persist_directory, embedding_function=CustomEmbeddingFunction(llm_instance))

        if ids_to_delete:
            delete_documents_from_db(db, ids_to_delete)
        
        if documents:
            try:
                chunk_size = 20
                num_process = len(documents) // chunk_size 
                with ThreadPoolExecutor(max_workers=5) as executor:
                    document_chunks = [documents[i * chunk_size:(i + 1) * chunk_size] for i in range(num_process)]  
                    if len(documents) % chunk_size != 0:  # Handle the remaining documents  
                        document_chunks.append(documents[num_process * chunk_size:])
                    if len(document_chunks) > 0:
                        futures = {}
                        for _, chunk in enumerate(document_chunks):
                            future = executor.submit(add_documents_to_db, db, chunk)
                            futures[future] = chunk
                        
                        for future in tqdm(as_completed(futures), total=len(futures), desc='Adding documents to db'):
                            chunk = futures[future]
                            try:
                                future.result()
                            except Exception as e:
                                print(f"Error while adding documents to db: {e}")

                task_result += f"- {collection} data added to db successfully.\n"
            except Exception as e:
                task_result += f"- {collection} data could not be added to db.\n"
        else:
            task_result += f"- {collection} data has no document to embed.\n"
        
        # Verify that 'New'/'Modified' entries are in the db
        reference_data[collection]['data'] = verification_embedding_step(db, reference_data[collection]['data'])

    with open(os.path.join(db_directory, 'data.pkl'), 'wb') as f:
        pickle.dump(reference_data, f)
    return db, task_result