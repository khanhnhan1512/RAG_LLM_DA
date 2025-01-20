import os  
import json  
import threading 
import pandas as pd
import numpy as np
import math
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from openai_llm.llm_init import LLM_Model
from stages.stage_1_learn_rules_from_data.data_loader import DataLoader
from stages.stage_4_llm_reasoning.score_function import score_1
from utils import load_json_data, save_json_data, load_vectorstore_db, lookup_vector_db


def safe_save_results(results, base_filename):  
    """  
    Lưu kết quả một cách an toàn với khóa threading  
    """  
    # Tạo thư mục nếu chưa tồn tại  
    os.makedirs(os.path.dirname(base_filename), exist_ok=True)  
    
    # Sử dụng lock để đảm bảo an toàn khi ghi file  
    lock = threading.Lock()  
    
    with lock:  
        # Ghi từng phần kết quả  
        with open(base_filename, 'a') as f:  
            json.dump(results, f)  
            f.write('\n')  # Xuống dòng để phân tách các lần ghi

def convert_query_to_natural_question(test_query, transformed_relations):
    if "inv_" in test_query[1]:
        question = f"{test_query[0]} {transformed_relations[test_query[1]]} by whom on {test_query[3]}?"
    else:
        question = f"{test_query[0]} {transformed_relations[test_query[1]]} to/with whom on {test_query[3]}?"
    return question

def get_top_k_relations(similarity_matrix, relation_id, data, top_k=10):
    relation_similarity = similarity_matrix[relation_id]
    # print(relation_similarity[3])
    top_k_relations_id = np.argsort(relation_similarity)[::-1][:top_k].tolist()
    top_k_relations_id.append(relation_id)
    top_k_relations = [data.id2relation[rel_id] for rel_id in top_k_relations_id]
    return top_k_relations_id, top_k_relations

def get_top_k_entities(similarity_matrix, entity_id, data, top_k=5):
    entity_similarity = similarity_matrix[entity_id]
    top_k_entity_id = np.argsort(entity_similarity)[::-1][:top_k].tolist()
    top_k_entity = [data.id2entity[entity_id] for entity_id in top_k_entity_id]
    return top_k_entity_id, top_k_entity

def filter_related_rules(rules_dict, rel_id, top_k_id):
    related_rules = []
    rules = rules_dict[rel_id]
    # print(len(rules))
    for rule in rules:
        if list(set(top_k_id) & set(rule['body_rels'])):
            related_rules.append(rule['verbalized_rule'])

    return related_rules

def get_facts_of_related_entity(related_entities, vector_db, llm_instance, search_content):
    result = []
    for ent in related_entities:
        filter = {"subject": ent}
        docs = lookup_vector_db(search_content, filter, vector_db, llm_instance, top_k=10)
        result.extend([doc.page_content for doc in docs])
    return result

def get_facts_between_subject_entity_end_its_relate_entities(related_entities, subject_entity, vector_db, llm_instance, search_content):
    result = []
    for ent in related_entities:
        filter = {"$and": [
            {"object": {"$in": [subject_entity, ent]}},
            {"subject": {"$in": [subject_entity, ent]}}
            ]   
        }
        docs = lookup_vector_db(search_content, filter, vector_db, llm_instance, top_k=5)
        for doc in docs:
            result.append(doc.page_content)
    
    return result

def get_related_facts(search_content, vector_db, llm_instance, related_facts, candidates_dict, related_relations, seen_entities=None, n=0):  
    # Khởi tạo sets theo dõi nếu là lần gọi đầu tiên  
    if seen_entities is None:  
        seen_entities = set()  
    
    # Điều kiện dừng  
    if n == 2:  
        return related_facts  
    
    cands = candidates_dict[n]  
    seen_entities.update(cands)
    candidates_dict[n+1] = set()  
    
    # Xử lý từng candidate  
    for can in cands:   
        filter = {"subject": can}
        docs = lookup_vector_db(search_content, filter, vector_db, llm_instance, top_k=20//(len(cands)))  
        
        for doc in docs:  
            fact_content = doc.page_content  
            new_entity = doc.metadata['object']
            related_relations.add(doc.metadata['relation'])  
            related_facts.append(fact_content)      
            # Chỉ thêm entity mới vào candidates cho hop tiếp theo  
            if new_entity not in seen_entities:  
                candidates_dict[n+1].add(new_entity)  
    
    # Nếu không còn candidates mới, dừng sớm  
    if not candidates_dict[n+1]:  
        return related_facts  
        
    return get_related_facts(  
        search_content,   
        vector_db,   
        llm_instance,   
        related_facts,   
        candidates_dict, 
        related_relations,  
        seen_entities,      
        n+1  
    )
 
def candidate_reasoning( question, related_facts, top_k_entity, facts_between_entity_subject_and_related_entities, related_entity_facts, llm_instance):
    system_msg_content = f'''
    You are an expert in Temporal Knowledge Graphs, utilizing data consisting of events and activities worldwide involving countries, organizations, famous individuals, etc.   
    Your task is Temporal Knowledge Graph Reasoning, which involves predicting the missing object in a given fact from the test dataset. A fact is represented as a quadruple: subject, relation, object, and time.

    In this context:  
    - "subject" entity is the entity mentioned in the query  
    - "relation" is an action/event performed by the subject  
    - "object" entity is the entity you need to infer through reasoning  
    - "timestamp": The temporal aspect of the fact

    To support your reasoning process in finding the missing object, you will be provided with relevant facts:  
    1. Primary Information:  
    - A sequence of events (facts), known as "Reasoning Paths", related to the query's subject and relation  
    2. Secondary Information:  
    - Most related entities to the entity subject  
    - Facts between these related entities with the entity subject.
    This information aids your reasoning process, especially because:     
    - These entities can be candidates because they might have relationships with the entity subject, which are also similar to the relationship between the entity subject and the missing object,
    3. Additional Information:
    - Facts about entities "related" to the "subject" entity.
    These facts are also very useful when you have too few or no facts directly related to the "subject" entity. If so, the pattern of these facts can be viewed as belonging to the "subject" entity, and the actions that these similar entities have performed can also be considered as actions of the subject entity, thereby helping to find candidates for the given query.

    You should follow these reasoning Process Guidelines:  
    1. Primary Analysis:  
    - Analyze direct reasoning paths connecting to the subject  
    - Identify temporal patterns and their significance  
    - Evaluate the strength of direct evidence  

    2. Multi-hop Reasoning:  
    - Consider indirect connections through intermediate entities  
    - Evaluate path length and relevance  
    - Consider temporal sequence of connected facts  
    - Weight evidence based on path length and temporal proximity  

    3. Related Entity Analysis: 
    If direct evidence is insufficient, consider related entities:
    - Consider most related entities as potential candidates because they might have relationships with the entity subject in the past.
    - Examine patterns/facts from semantically related entities and apply successful patterns from them  

    For the candidate selection criteria:  
    1. Evidence Strength:  
    - Direct path evidence (highest weight)  
    - Multi-hop reasoning paths 
    - Related entity and their patterns/facts  
        
    2. Temporal Relevance:  
    - Recency of connections  
    - Pattern consistency over time  
    - Temporal proximity to query time  
    
    Finally, remember that when there are too few or no facts directly related to the "subject" entity, use the "Additional Information" to find candidates for the given query.
    Your answer should be in the following JSON format:  
    {{  
        "candidates": // An ordered list of up to 10 candidates, from highest to lowest likelihood of being the correct answer.   
                    // Each candidate should be an entity name exactly as it appears in the given facts.  
                    // The list should be ordered by decreasing probability of being the correct answer.
                    // You will not be allowed to give the empty list as an answer. Remember that when there are too few or no facts directly related to the "subject" entity, use the "Additional Information" to find candidates for the given query.
    }}
    '''
    system_msg = SystemMessage(content=system_msg_content)

    user_msg_content = f'''
    Here is the question you need to find the answer:
    - {question}

    For the primary information:
    - Here are facts related to the query's subject and relation:
    {related_facts}

    For the secondary information:
    - Here are the most related entities to the entity "subject" and the facts between these related entities and the entity "subject":
    + {top_k_entity}
    + {facts_between_entity_subject_and_related_entities}

    For the additional information:
    - Here are the facts of the related entities:
    + {related_entity_facts}
    '''
    user_msg = HumanMessage(content=user_msg_content)

    answer_llm = llm_instance.run_task([system_msg, user_msg])
    return answer_llm['candidates']

def get_candidates(test_query, i, transformed_relations, vector_db, llm_instance, data, entity_similarity_matrix):
    """
    
    """
    # Convert query to natural question
    print(f"Processing query {i}...")
    question = convert_query_to_natural_question(test_query, transformed_relations)

    # Get related facts
    search_content = transformed_relations[test_query[1]]
    candidates_dict = {0: {test_query[0]}} 
    related_relations = set()
    try:
        related_facts = get_related_facts(search_content, vector_db, llm_instance, [], candidates_dict, related_relations)
    except:
        related_facts = []
    # Get most related entities and their facts
    query_subject_id = data.entity2id[test_query[0]]
    top_k_entity_id, top_k_entity = get_top_k_entities(entity_similarity_matrix, query_subject_id, data)
    try:
        related_entity_facts = get_facts_of_related_entity(top_k_entity, vector_db, llm_instance, search_content)
    except:
        related_entity_facts = []

    # Get facts between related entities and the entity subject
    try:
        facts_between_entity_subject_and_related_entities = get_facts_between_subject_entity_end_its_relate_entities(
            top_k_entity, test_query[0], vector_db, llm_instance, search_content
        )
    except:
        facts_between_entity_subject_and_related_entities = []

    # Get candidates list
    candidates = candidate_reasoning(question, related_facts, top_k_entity, facts_between_entity_subject_and_related_entities, related_entity_facts, llm_instance)
    candidates_id = [data.entity2id[ent] for ent in candidates if ent in data.entity2id]
    print(f"Finish query {i}...")
    return candidates, candidates_id

def get_entity_max_ts(test_query_subject, candidate, historical_data):
    other_answers = historical_data[
        (historical_data[:, 0] == test_query_subject)
        * (historical_data[:, 2] == candidate)
    ]
    # sort by timestamp in descending order
    other_answers = other_answers[other_answers[:, 3].argsort()[::-1]]
    # return the max timestamp
    if len(other_answers) > 0:
        return other_answers[0, 3]
    else:
        return None

def scoring_candidates(candidates_id, i, test_data_dict, data):
    test_id, test_query = next(iter(test_data_dict[i].items()))
    cands_score_dict = {}
    query_ts = data.ts2id[test_query[3]]
    historical_data = np.vstack((data.train_data_idx, data.valid_data_idx))
    for rank, id in enumerate(candidates_id):
        cand_max_ts = get_entity_max_ts(data.entity2id[test_query[0]], id, historical_data)
        if cand_max_ts:
            cands_score_dict[id] = score_1(rank, cand_max_ts, query_ts, 0.5)
        else:
            cands_score_dict[id] = score_1(rank, 0, query_ts, 1.0)
    # sort cands_score_dict
    cands_score_dict = dict(sorted(cands_score_dict.items(), key=lambda item: item[1], reverse=True))
    return cands_score_dict

def apply_llm_reasonging_parallel(test_data_dict, process, num_queries, transformed_relations, vector_db, llm_instance, data, entity_similarity_matrix, num_process):
    result = dict()
    base_filename = f"result/icews14/stage_4/candidates_part_{process}.jsonl"
    test_query_idx = range(process * num_queries, (process + 1) * num_queries) if process < num_process-1 else range(process * num_queries, len(test_data_dict))
    for j in test_query_idx:
        test_id, test_query = next(iter(test_data_dict[j].items()))
        candidates = get_candidates(test_query, test_id, transformed_relations, vector_db, llm_instance, data, entity_similarity_matrix)[1]  
        
        # Lưu từng phần kết quả ngay lập tức  
        safe_save_results({test_id: candidates}, base_filename)  
        
        result[test_id] = candidates

    return result

def scoring_candidates_parallel(query_cands_dict, process, num_queries, test_data_dict, data, num_process):
    result = dict()
    test_query_idx = range(process * num_queries, (process + 1) * num_queries) if process < num_process-1 else range(process * num_queries, len(test_data_dict))
    for j in test_query_idx:
        result[j] = scoring_candidates(query_cands_dict[str(j)], j, test_data_dict, data)

    return result

def stage_4_main():
    # number of processes
    num_process = 1

    # Load LLm model
    llm_instance = LLM_Model()
    dataset_dir = os.path.join(".", "datasets", 'icews14')
    dir_path = os.path.join(".", "result", 'icews14', "stage_3")

    # Load data and test data
    data = DataLoader(dataset_dir)
    test_data = data.test_data_text

    test_data_dict = [{i:v} for i, v in enumerate(test_data)]
    test_data_dict = test_data_dict[:7371]

    # Load similarity matrix
    relation_similarity_matrix = np.load('result/icews14/stage_1/relation_similarity.npy')
    entity_similarity_matrix = np.load('./entity_similarity.npy')
    transformed_relations = load_json_data('result/icews14/stage_1/transformed_relations.json')

    # Load vectorstore db
    vector_db = load_vectorstore_db(llm_instance, 'icews14')
    for collection in vector_db:
        print(f"{collection}: {len(vector_db[collection]['vector_db'].get()['documents'])} documents")

    ##################################################################################################
    # query_cands_dict = {}
    num_queries = math.ceil(len(test_data_dict) / num_process)
    futures = []
    # with ThreadPoolExecutor(max_workers=num_process) as executor:
    #     for i in range(num_process):
    #         futures.append(executor.submit(apply_llm_reasonging_parallel, test_data_dict, i, num_queries, transformed_relations, vector_db['facts']['vector_db'], llm_instance, data, entity_similarity_matrix, num_process))
    
    #     for future in as_completed(futures):
    #         result = future.result()

    # # Sau khi chạy xong, merge các file kết quả  
    # def merge_result_files():  
    #     final_results = {}  
    #     for i in range(num_process):  
    #         filename = f"result/icews14/stage_4/candidates_part_{i}.jsonl"  
    #         if os.path.exists(filename):  
    #             with open(filename, 'r') as f:  
    #                 for line in f:  
    #                     part_result = json.loads(line)  
    #                     final_results.update(part_result)  
        
    #     # Sắp xếp và lưu kết quả cuối cùng  
    #     sorted_results = dict(sorted(final_results.items(), key=lambda x: int(x[0])))  
    #     with open("result/icews14/stage_4/final_candidates.json", 'w') as f:  
    #         json.dump(sorted_results, f, indent=4)  

    # Gọi hàm merge kết quả  
    # merge_result_files()

    # scoring for candidates
    query_cands_dict = load_json_data("result/icews14/stage_4/final_candidates.json")
    query_cands_score_dict = {}
    scoring = []
    with ThreadPoolExecutor(max_workers=num_process) as executor:
        for i in range(num_process):
            scoring.append(executor.submit(scoring_candidates_parallel, query_cands_dict, i, num_queries, test_data_dict, data, num_process))
       
        for future in as_completed(scoring):
            query_cands_score_dict.update(future.result())
            


    # ##################################################################################################
    # query_cands_dict = dict(sorted(query_cands_dict.items(), key=lambda item: item[0]))
    save_json_data(query_cands_score_dict, "result/icews14/stage_4/candidates_score.json")