import os
import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from openai_llm.llm_init import LLM_Model
from stages.stage_1_learn_rules_from_data.data_loader import DataLoader
from stages.stage_1_learn_rules_from_data.temporal_walk import TemporalWalker
from stages.stage_1_learn_rules_from_data.temporal_walk import store_edges
from stages.stage_1_learn_rules_from_data.rule_learning import RuleLearner, rules_statistics
from stages.stage_4_llm_reasoning.score_function import score_1
from utils import load_json_data, save_json_data, load_vectorstore_db, load_learn_data, lookup_vector_db

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

def get_top_k_entities(similarity_matrix, entity_id, data, top_k=10):
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
    if n == 3:  
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
 
def candidate_reasoning( question, related_facts, related_rules, top_k_entity, facts_between_entity_subject_and_related_entities, related_entity_facts, llm_instance):
    system_msg_content = f'''
    You are an expert in Temporal Knowledge Graphs, utilizing data consisting of events and activities worldwide involving countries, organizations, famous individuals, etc.   
    Your task is Temporal Knowledge Graph Reasoning, which involves predicting the missing object in a given fact from the test dataset. A fact is represented as a quadruple: subject, relation, object, and time.

    In this context:  
    - "subject" is the entity mentioned in the query  
    - "relation" is an action/event performed by the subject  
    - "object" is the entity you need to infer through reasoning  
    - "timestamp": The temporal aspect of the fact

    To support your reasoning process in finding the missing object, you will be provided with relevant facts:  
    1. Primary Information:  
    - A sequence of events (facts), known as "Reasoning Paths", related to the query's subject and relation  
    - Learned rules from training and validation datasets that represent patterns which events typically follow. An important note is relations with the "inv_" prefix (e.g., "inv_make_statement") indicate passive relations. For example: inv_make_statement(B,A,T) means "B receives a statement from A at time T.
    Using this source of information, your task is to infer the missing "object" through multi-hop reasoning.  
    2. Secondary Information:  
    - Most related entities to the entity subject  
    - Facts between these related entities with the entity subject.
    This information aids your reasoning process, especially because:     
    - These entities can be candidates because they might have relationships with the entity subject, which are also similar to the relationship between the entity subject and the missing object,
    3. Additional Information:
    - The facts of these related entities.
    This information is useful especiall when there are no facts about the entity subject in the past. So, patterns from these similar entities might apply to the entity subject to infer the missing object.

    You should follow these reasoning Process Guidelines:  
    1. Primary Analysis:  
    - Analyze direct reasoning paths connecting to the subject  
    - Match and apply relevant rules to existing facts  
    - Identify temporal patterns and their significance  
    - Evaluate the strength of direct evidence  

    2. Multi-hop Reasoning:  
    - Consider indirect connections through intermediate entities  
    - Evaluate path length and relevance  
    - Consider temporal sequence of connected facts  
    - Weight evidence based on path length and temporal proximity  

    3. Similar Entity Analysis:  
    - Consider most related entities as potential candidates because they might have relationships with the entity subject in the past.
    - Examine patterns from semantically similar entities  
    - Apply successful patterns from similar entities  

    Finally, for the candidate selection criteria:  
    1. Evidence Strength:  
    - Direct path evidence (highest weight)  
    - Rule application matches  
    - Multi-hop reasoning paths 
    - Related entity patterns  
        
    2. Temporal Relevance:  
    - Recency of connections  
    - Pattern consistency over time  
    - Temporal proximity to query time  

    3. Confidence Scoring:  
    - Direct evidence: High confidence  
    - Rule-based inference: Medium-high confidence  
    - Related entity and their patterns: Medium confidence  
    - Multi-hop paths: Weighted by path length

    Your answer should be in the following JSON format:  
    {{  
        "candidates": // An ordered list of up to 10 candidates, from highest to lowest likelihood of being the correct answer.   
                    // Each candidate should be an entity name exactly as it appears in the given facts.  
                    // The list should be ordered by decreasing probability of being the correct answer.
    }}
    '''
    system_msg = SystemMessage(content=system_msg_content)

    user_msg_content = f'''
    Here is the question you need to find the answer:
    - {question}

    For the primary information:
    - Here are facts related to the query's subject and relation:
    {related_facts}
    - Here are the learned rules related to query's relation:
    {related_rules}

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

def get_candidates(test_query, i, transformed_relations, vector_db, llm_instance, data, rules_dict, entity_similarity_matrix):
    """
    
    """
    # Convert query to natural question
    question = convert_query_to_natural_question(test_query, transformed_relations)

    # Get related facts
    search_content = transformed_relations[test_query[1]]
    candidates_dict = {0: {test_query[0]}} 
    related_relations = set()
    related_facts = get_related_facts(search_content, vector_db, llm_instance, [], candidates_dict, related_relations)
    related_rels_id = [data.relation2id[rel] for rel in related_relations]

    # Get related rules
    related_rules = filter_related_rules(rules_dict, data.relation2id[test_query[1]], related_rels_id)

    # Get most related entities and their facts
    query_subject_id = data.entity2id[test_query[0]]
    top_k_entity_id, top_k_entity = get_top_k_entities(entity_similarity_matrix, query_subject_id, data)
    related_entity_facts = get_facts_of_related_entity(top_k_entity, vector_db, llm_instance, search_content)

    # Get facts between related entities and the entity subject
    facts_between_entity_subject_and_related_entities = get_facts_between_subject_entity_end_its_relate_entities(
        top_k_entity, test_query[0], vector_db, llm_instance, search_content
    )

    # Get candidates list
    candidates = candidate_reasoning(question, related_facts, related_rules, top_k_entity, facts_between_entity_subject_and_related_entities, related_entity_facts, llm_instance)
    candidates_id = [data.entity2id[ent] for ent in candidates]

    return i, candidates, candidates_id

def get_entity_max_ts(vector_db, llm_instance, test_query, data, entity, transformed_relations):
    search_conent = transformed_relations[test_query[1]]
    filter = {"$and": 
        [
            {"object": {"$in": [test_query[0], data.id2entity[entity]]}},
            {"subject": {"$in": [test_query[0], data.id2entity[entity]]}},
        ]
    }
    docs = lookup_vector_db(search_conent, filter, vector_db, llm_instance, top_k=100)
    sorted_docs = sorted(docs, key=lambda doc: doc.metadata['timestamp_id'], reverse=True)
    if sorted_docs:
        return sorted_docs[0].metadata['timestamp_id']
    else:
        return None

def scoring_candidates(candidates_id, test_query, data, vector_db, llm_instance, transformed_relations):
    cands_score_dict = {}
    query_ts = data.ts2id[test_query[3]]
    for rank, id in enumerate(candidates_id):
        cand_max_ts = get_entity_max_ts(vector_db, llm_instance, test_query, data, id, transformed_relations)
        if cand_max_ts:
            cands_score_dict[id] = score_1(rank, cand_max_ts, query_ts, 0.5)
        else:
            cands_score_dict[id] = score_1(rank, 0, query_ts, 1.0)
    return cands_score_dict

def stage_4_main():
    # Load LLm model
    llm_instance = LLM_Model()
    dataset_dir = os.path.join(".", "datasets", 'icews14')
    dir_path = os.path.join(".", "result", 'icews14', "stage_3")

    # Load data and test data
    data = DataLoader(dataset_dir)
    test_data = data.test_data_text

    # Load rules
    rule_regex = load_json_data("config/rule_regex.json")['icews14']
    temporal_walk_data = load_learn_data(data, 'all')
    temporal_walk = TemporalWalker(temporal_walk_data, data.inverse_rel_idx, 'exp')
    rl = RuleLearner(temporal_walk.edges, data.relation2id, data.id2entity, data.id2relation, data.inverse_rel_idx, 
                        'icews14', len(temporal_walk_data), dir_path)
    rules_df = pd.read_csv('result/icews14/stage_2/01_only_Markovian_merged_results.csv')
    for _, entry in rules_df.iterrows():
        rl.create_rule_from_series_df(entry=entry, rule_regex=rule_regex)
    rules_dict = rl.rules_dict
    print("Rules statistics:")
    rules_statistics(rules_dict)

    # Load similarity matrix
    relation_similarity_matrix = np.load('result/icews14/stage_1/relation_similarity.npy')
    entity_similarity_matrix = np.load('./entity_similarity.npy')
    transformed_relations = load_json_data('result/icews14/stage_1/transformed_relations.json')

    # Load vectorstore db
    vector_db = load_vectorstore_db(llm_instance, 'icews14')
    for collection in vector_db:
        print(f"{collection}: {len(vector_db[collection]['vector_db'].get()['documents'])} documents")

    ##################################################################################################
    query_cands_dict = {}
    for i, test_query in enumerate(test_data[:1]):
        i, candidates, candidates_id = get_candidates(test_query, i, transformed_relations, vector_db['facts']['vector_db'], llm_instance, data, rules_dict, entity_similarity_matrix)
        query_cands_dict[i] = candidates_id
        print(f"Query {i}: {test_query} - Candidates: {candidates}")
    
    # scoring
    for query_id, cands_id in query_cands_dict.items():
        candidates_score = scoring_candidates(cands_id, test_data[query_id], data, vector_db['facts']['vector_db'], llm_instance, transformed_relations)
        query_cands_dict[query_id] = candidates_score
    
    ##################################################################################################
    save_json_data("result/icews14/stage_4/candidates_score.json", query_cands_dict)