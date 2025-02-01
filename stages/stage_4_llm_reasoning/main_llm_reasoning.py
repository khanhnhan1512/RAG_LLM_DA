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
        question = f"{test_query[0]} {transformed_relations[test_query[1]]} whom on {test_query[3]}?"
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

def get_ground_truth_answers(test_query, data, vector_db, llm_instance, test_data_start_timestamp, test_size):
    subject_ent = test_query[0]
    relation = test_query[1]
    query_timestamp_id = data.ts2id[test_query[3]]
    filter = {
        "$and":[
            {"subject": subject_ent}, 
            {"relation": relation},
            {"timestamp_id": {"$lt": query_timestamp_id}},
            {"timestamp_id": {"$gte": test_data_start_timestamp}},
        ]
    }
    retrieved_docs = lookup_vector_db("", filter, vector_db, llm_instance, top_k=test_size)
    ground_truth_answers = [doc.metadata['object'] for doc in retrieved_docs]
    return ground_truth_answers

def get_facts_of_related_entity(related_entities, vector_db, llm_instance, search_content, query_timestamp_id):
    result = []
    for ent in related_entities:
        filter = {
            "$and":[
            {"subject": ent}, 
            {"timestamp_id": {"$lt": query_timestamp_id}}
            ]
        }
        retrieved_docs = lookup_vector_db(search_content, filter, vector_db, llm_instance, top_k=100)
        if len(retrieved_docs) < 10:
            docs = retrieved_docs
        else:
            retrieved_docs = sorted(retrieved_docs, key= lambda doc: doc.metadata['timestamp_id'], reverse=True)
            docs = retrieved_docs[:5]
            docs.extend(retrieved_docs[-5:])

        result.extend([doc.page_content for doc in docs])
    return result

def get_facts_between_subject_entity_end_its_relate_entities(related_entities, subject_entity, vector_db, llm_instance, search_content, query_timestamp_id):
    result = []
    for ent in related_entities:
        filter = {"$and": [
            {"object": {"$in": [subject_entity, ent]}},
            {"subject": {"$in": [subject_entity, ent]}},
            {"timestamp_id": {"$lt": query_timestamp_id}}
            ]   
        }
        docs = lookup_vector_db(search_content, filter, vector_db, llm_instance, top_k=5)
        for doc in docs:
            result.append(doc.page_content)
    
    return result

def get_related_facts(search_content, vector_db, llm_instance, related_facts, candidates_dict, related_relations, query_timestamp_id, seen_entities=None, n=0):  
    # Khởi tạo sets theo dõi nếu là lần gọi đầu tiên  
    if seen_entities is None:  
        seen_entities = set()  
    
    # Điều kiện dừng  
    if n == 2:  
        return related_facts  
    
    cands = candidates_dict[n]  
    seen_entities.update(cands)
    candidates_dict[n+1] = set()  
    num_docs = 30//(len(cands))
    # Xử lý từng candidate  
    for can in cands:   
        filter = {
            "$and":
            [{"subject": can}, 
            {"timestamp_id": {"$lt": query_timestamp_id}}]
        }
        num_recently_docs = num_docs * 2 // 3
        num_history_docs = num_docs - num_recently_docs
        retrieved_docs = lookup_vector_db(search_content, filter, vector_db, llm_instance, top_k=100)  
        retrieved_docs = sorted(retrieved_docs, key= lambda doc: doc.metadata['timestamp_id'], reverse=True)
        if len(retrieved_docs) < num_docs:
            docs = retrieved_docs
        else:
            docs = retrieved_docs[:num_recently_docs]
            docs.extend(retrieved_docs[-num_history_docs:])
        
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
        query_timestamp_id,
        seen_entities,      
        n+1  
    )
 
def candidate_reasoning( question, related_facts, top_k_entity, facts_between_entity_subject_and_related_entities, related_entity_facts, ground_truth_answers, llm_instance):
    system_msg_content = f'''
    You are an advanced reasoning assistant tasked with solving Temporal Knowledge Graph (TKG) reasoning problems. Your goal is to predict the missing object in a query given the subject, relation, and timestamp. To achieve this, you will use a Retrieval-Augmented Generation (RAG) approach, leveraging the following groups of retrieved information:
    ---
    # Instruction for Reasoning:
    1. Understand the Query:
    - The query will always include a subject, a relation, and a timestamp.
    - Example: "Malaysia expressed intent to cooperate to/with whom on 2014-12-09?"
    - Your task is to predict the missing object (e.g., a country, organization, or entity) that best fits the query.
    2. Leverage Multi-Hop Reasoning (Group 1):
    - You will be provided with a sequence of multi-hop facts related to the subject entity. These facts are connected directly or indirectly to the subject and share a semantic similarity with the query's relation.
    - Example facts:
        + "Malaysia were the recipients of expressed intent to cooperate to/with Thailand on 2014-12-02"
        + "Malaysia expressed intent to cooperate to/with China on 2014-04-11"
    - Perform multi-hop reasoning by analyzing these facts and their relationships. Pay close attention to the timestamps of the facts to ensure temporal consistency with the query.
    3. Expand Candidate Entities (Group 2):
    - If the multi-hop facts are insufficient, use additional facts involving semantically similar entities to the subject.
    - Example: For "Malaysia", semantically similar entities might include "Men_(Malaysia)", "Police_(Malaysia)", etc.
    - Example fact: "Police_(Malaysia) Made a statement to/with Malaysia on 2014-12-08".
    - Use these facts to expand the list of candidate objects for the query.
    4. Infer from Semantically Similar Entities (Group 3):
    - If no direct facts about the subject entity exist before the query's timestamp, infer the missing object by analyzing patterns from semantically similar entities.
    - Example: If "Police_(Malaysia) Expressed intent to meet or negotiate to/with Citizen_(Malaysia) on 2014-02-21", you can infer that "Malaysia" might have a similar pattern of cooperation with "Citizen_(Malaysia)".
    - Use this approach to make educated predictions when direct evidence is lacking.
    5. Learn from Historical Query Patterns (Group 4):
    - If the query is part of a series of similar queries with different timestamps, you will be provided with ground truth answers for previous queries.
    - Example:
        + Query: "Malaysia expressed intent to cooperate to/with whom on 2014-12-09?"
        + Ground truth: "Thailand"
    - Use these ground truths as hints to avoid repeating mistakes and improve accuracy for the current query.
    ---
    # Reasoning steps:
    - Analyze the Query: Identify the subject, relation, and timestamp. Determine the type of object you are predicting (e.g., country, organization, person).
    - Retrieve and Prioritize Facts: Start with Group 1 (multi-hop facts) and prioritize facts with timestamps closest to the query's timestamp. If insufficient, move to Group 2 (semantically similar entities) and Group 3 (inference from similar entities).
    - Perform Multi-Hop Reasoning: Trace connections between facts to identify potential candidate objects. Ensure the reasoning process respects the temporal order of events.
    - Expand and Infer: Use facts from semantically similar entities to expand the candidate pool. Infer patterns from similar entities if direct evidence is unavailable.
    - Incorporate Historical Patterns: Use ground truths from similar historical queries to guide your prediction.
    ---
    Query: "Malaysia expressed intent to cooperate to/with whom on 2014-12-09?"
    Retrieved Facts:
        - "Malaysia were the recipients of expressed intent to cooperate to/with Thailand on 2014-12-02"
        - "Police_(Malaysia) Expressed intent to meet or negotiate to/with Citizen_(Malaysia) on 2014-02-21"
        - Ground Truth Hint: "Thailand" (from a similar query)
    Reasoning:
        - The fact from 2014-12-02 shows Malaysia cooperating with Thailand, which is temporally close to the query's timestamp.
        - The fact from 2014-02-21 involves "Police_(Malaysia)", suggesting a pattern of cooperation with domestic entities.
        - The historical ground truth hints that "Thailand" is a likely candidate.
    Prediction: "Thailand"
    ---
    Your answer should be in the following JSON format:  
    {{  
        "candidates": // An ordered list of up to 10 candidates, from highest to lowest likelihood of being the correct answer.   
                    // Each candidate should be an entity name exactly as it appears in the given facts.  
                    // The list should be ordered by decreasing probability of being the correct answer.
                    // You will not be allowed to give the empty list as an answer.
    }}
    '''
    system_msg = SystemMessage(content=system_msg_content)

    user_msg_content = f'''
    Here is the question you need to find the answer:
    - {question}

    Here are multi-hop facts related to the query's subject entity (group 1):
    - {related_facts}

    Here are top {len(top_k_entity)} semantically similar entities to the subject entity and facts between them and the subject entity (Group 2):
    - {top_k_entity}
    - {facts_between_entity_subject_and_related_entities}

    Here are facts from semantically similar entities to the subject entity (group 3):
    - {related_entity_facts}

    Here are ground truth answers for previous queries, which are part of a series of similar queries with different timestamps (Group 4):
    - {ground_truth_answers}
    '''
    user_msg = HumanMessage(content=user_msg_content)

    answer_llm = llm_instance.run_task([system_msg, user_msg])
    return answer_llm['candidates']

def get_candidates(test_query, i, transformed_relations, vector_db, llm_instance, data, entity_similarity_matrix, test_data_start_ts, test_size):
    """
    
    """
    # Convert query to natural question
    print(f"Processing query {i}...")
    question = convert_query_to_natural_question(test_query, transformed_relations)

    # Get related facts
    search_content = transformed_relations[test_query[1]]
    query_timestamp_id = data.ts2id[test_query[3]]
    candidates_dict = {0: {test_query[0]}} 
    related_relations = set()
    try:
        related_facts = get_related_facts(search_content, vector_db, llm_instance, [], candidates_dict, related_relations, query_timestamp_id)
    except Exception as e:
        related_facts = []
    # Get most related entities and their facts
    query_subject_id = data.entity2id[test_query[0]]
    top_k_entity_id, top_k_entity = get_top_k_entities(entity_similarity_matrix, query_subject_id, data)
    try:
        related_entity_facts = get_facts_of_related_entity(top_k_entity, vector_db, llm_instance, search_content, query_timestamp_id)
    except:
        related_entity_facts = []

    # Get facts between related entities and the entity subject
    try:
        facts_between_entity_subject_and_related_entities = get_facts_between_subject_entity_end_its_relate_entities(
            top_k_entity, test_query[0], vector_db, llm_instance, search_content, query_timestamp_id
        )
    except:
        facts_between_entity_subject_and_related_entities = []

    # Get ground truth answers
    ground_truth_answers = get_ground_truth_answers(test_query, data, vector_db, llm_instance, test_data_start_ts, test_size)

    # Get candidates list
    candidates = candidate_reasoning(question, related_facts, top_k_entity, facts_between_entity_subject_and_related_entities, related_entity_facts, ground_truth_answers, llm_instance)
    candidates_id = [data.entity2id[ent] for ent in candidates if ent in data.entity2id]
    print(f"Finish query {i}...")
    return candidates, candidates_id

def get_entity_max_ts(test_query_subject, candidate, historical_data, query_ts):
    historical_data = historical_data[historical_data[:, 3] < query_ts]

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
    historical_data = np.vstack((data.train_data_idx, data.valid_data_idx, data.test_data_idx))
    for rank, id in enumerate(candidates_id):
        cand_max_ts = get_entity_max_ts(data.entity2id[test_query[0]], id, historical_data, query_ts)
        if cand_max_ts:
            cands_score_dict[id] = score_1(rank, cand_max_ts, query_ts, 0.5)
        else:
            cands_score_dict[id] = score_1(rank, 0, query_ts, 1.0)
    # sort cands_score_dict
    cands_score_dict = dict(sorted(cands_score_dict.items(), key=lambda item: item[1], reverse=True))
    return cands_score_dict

def apply_llm_reasonging_parallel(test_data_dict, process, num_queries, transformed_relations, vector_db, llm_instance, 
                                  data, entity_similarity_matrix, num_process, test_data_start_ts=0, test_size=None):
    result = dict()
    base_filename = f"result/GDELT/stage_4/candidates_part_{process}.jsonl"
    test_query_idx = range(process * num_queries, (process + 1) * num_queries) if process < num_process-1 else range(process * num_queries, len(test_data_dict))
    for j in test_query_idx:
        test_id, test_query = next(iter(test_data_dict[j].items()))
        candidates = get_candidates(test_query, test_id, transformed_relations, vector_db, llm_instance, data, entity_similarity_matrix, test_data_start_ts, test_size)[1]  
        
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
    num_process = 8

    # Load LLm model
    llm_instance = LLM_Model()
    dataset_dir = os.path.join(".", "datasets", 'GDELT')

    # Load data and test data
    data = DataLoader(dataset_dir)
    test_data = data.test_data_text

    # timestamp of test data
    test_data_start_timestamp = data.ts2id[test_data[0][3]]

    test_data_dict = [{i:v} for i, v in enumerate(test_data)]
    test_data_dict = test_data_dict[12000:]

    # Load similarity matrix
    relation_similarity_matrix = np.load('result/GDELT/stage_1/relation_similarity.npy')
    entity_similarity_matrix = np.load('result/GDELT/stage_1/entity_similarity.npy')
    transformed_relations = load_json_data('result/GDELT/stage_1/transformed_relations.json')

    # Load vectorstore db
    vector_db = load_vectorstore_db(llm_instance, 'GDELT')
    for collection in vector_db:
        print(f"{collection}: {len(vector_db[collection]['vector_db'].get()['documents'])} documents")

    ##################################################################################################
    query_cands_dict = {}
    num_queries = math.ceil(len(test_data_dict) / num_process)
    futures = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i in range(num_process):
            futures.append(executor.submit(apply_llm_reasonging_parallel, test_data_dict, i, num_queries, 
                                           transformed_relations, vector_db['facts']['vector_db'], llm_instance, 
                                           data, entity_similarity_matrix, num_process, test_data_start_timestamp, len(test_data)//2))
    
        for future in as_completed(futures):
            result = future.result()

    # Sau khi chạy xong, merge các file kết quả  
    def merge_result_files():  
        final_results = {}  
        for i in range(num_process):  
            filename = f"result/GDELT/stage_4/candidates_part_{i}.jsonl"  
            if os.path.exists(filename):  
                with open(filename, 'r') as f:  
                    for line in f:  
                        part_result = json.loads(line)  
                        final_results.update(part_result)  
        
        # Sắp xếp và lưu kết quả cuối cùng  
        sorted_results = dict(sorted(final_results.items(), key=lambda x: int(x[0])))  
        with open("result/GDELT/stage_4/final_candidates.json", 'w') as f:  
            json.dump(sorted_results, f, indent=4)  

    # Gọi hàm merge kết quả  
    merge_result_files()

    # # scoring for candidates
    # query_cands_dict = load_json_data("result/GDELT/stage_4/final_candidates.json")
    # query_cands_score_dict = {}
    # scoring = []
    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     for i in range(num_process):
    #         scoring.append(executor.submit(scoring_candidates_parallel, query_cands_dict, i, num_queries, test_data_dict, data, num_process))
       
    #     for future in as_completed(scoring):
    #         query_cands_score_dict.update(future.result())
            
    # ##################################################################################################
    # query_cands_score_dict = dict(sorted(query_cands_score_dict.items(), key=lambda item: int(item[0])))
    # save_json_data(query_cands_score_dict, "result/GDELT/stage_4/candidates_score.json")