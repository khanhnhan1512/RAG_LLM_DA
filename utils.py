import json
import argparse
import os
import shutil
import re
import numpy as np
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from sklearn.metrics.pairwise import cosine_similarity
from stages.stage_3_rule_reasoning import rule_application as ra
from process_embedding.custom_embedding_function import CustomEmbeddingFunction

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="icews14", type=str)
    parser.add_argument("--test_data", default="test", type=str)
    parser.add_argument("--candidates", "-c", default="", type=str)
    parser.add_argument("--timestamp", "-t", default="", type=str)
    parser.add_argument("--checkpoint", "-ch", default="", type=str)
    parser.add_argument("--is_known", default="No", type=str_to_bool)
    parser.add_argument('--graph_reasoning_type', type=str,
                        choices=['transformer', 'timestamp', 'based_source_with_timestamp', 'origin', 'fusion',
                                 'fusion_with_weight', 'fusion_with_source', 'fusion_with_relation', 'TADistmult',
                                 'TADistmult_with_recent', 'frequcy_only', 'new_origin_frequency', 'TiRGN', 'REGCN'],
                        default='timestamp')
    parser.add_argument("--rule_weight", default=1.0, type=float)
    parser.add_argument("--model_weight", default=0.5, type=float)
    parser.add_argument("--interval", default=70, type=int)
    parser.add_argument("--index", default=-1, type=int)
    parser.add_argument("--group_size", default=0, type=int)

    #TiRGN
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=50,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph", action='store_true', default=False,
                        help="use the info of static graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")

    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10,
                        help="evolution speed")

    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=False,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=20,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=20,
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("--grid-search", action='store_true', default=False,
                        help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str, default="history_len,n_layers,dropout,n_bases,angle,history_rate",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")

    # configuration for global history
    parser.add_argument("--history-rate", type=float, default=0.3,
                        help="history rate")

    parser.add_argument("--save", type=str, default="one",
                        help="number of save")

    parsed = parser.parse_args()

    return parsed

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_lines_from_file(file_path):
    """
    Reads each line from a text file and returns them as a list of strings.

    :param file_path: Path to the text file to read.
    :return: A list of strings, each representing a line from the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # Strip newline characters from each line
    lines = [line.strip() for line in lines]
    return lines

def write_lines_to_file(file_path, lines):
    """
    Writes each element of a list into a text file, with each element on a new line.

    :param file_path: Path to the text file to write.
    :param lines: List of strings to write to the file.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in lines:
            file.write(f"{line}\n")

def print_sorted_params(params):
    params_dict = vars(params)
    sorted_params = sorted(params_dict.items(), key=lambda x: x[0])

    for key, value in sorted_params:
        print(f"{key}: {value}")

def load_json_data(file_path):
    try:
        print(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        data = None
    return data

def save_json_data(data, file_path):
    try:
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Data has been converted to JSON and saved to {file_path}")
    except Exception as e:
        print(f"Error saving JSON data to {file_path}")

def write_to_file(content, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)

def copy__file(source_dirt_path, destination_path):
    try:
        shutil.rmtree(destination_path)
        print(f"Deleted existing files/data at {destination_path}")
    except:
        print(f"No existing files/data at {destination_path}")
    
    try:
        os.mkdir(destination_path)
    except:
        pass

    try:
        shutil.copytree(source_dirt_path, destination_path, dirs_exist_ok=True)
        print(f"Copied files from {source_dirt_path} to {destination_path}")
    except Exception as e:
        print(f"Error while copying files from {source_dirt_path} to {destination_path}: {e}")

def load_learn_data(data, type):
    data_map = {
        'all': np.array(data.train_data_idx.tolist() + data.valid_data_idx.tolist() + data.test_data_idx.tolist()),
        'train': np.array(data.train_data_idx.tolist()),
        'valid': np.array(data.valid_data_idx.tolist()),
        'test': np.array(data.test_data_idx.tolist()),
        'train_valid': np.array(data.train_data_idx.tolist() + data.valid_data_idx.tolist())
    }
    return data_map[type]

def transform_relations(all_rels, llm_instance, output_dir):
    batch_size = 10

    result = dict()

    system_msg_content = '''
    You are an expert in Temporal Knowledge Graph. Your task is to transform the given relations into a natural language term.
    - Each given relation represents for an action between two entities in the past.
    - Remove underscores and technical formatting  
    - Use simple past tense (historical context)

    Your answer should be in a json format as below:
    {{
        'original_relations': // list of original relations
        'transformed_relations': // list of transformed relations corresponding to the original relations
    }}
    '''
    system_msg = SystemMessage(content=system_msg_content)

    for i in range(0, len(all_rels), batch_size):
        user_msg_content = f'''
            The list of relations need to be transformed: {all_rels[i:i+batch_size]}
        '''
        user_msg = HumanMessage(content=user_msg_content)

        answer_llm = llm_instance.run_task([system_msg, user_msg])

        for k, v in zip(answer_llm['original_relations'], answer_llm['transformed_relations']):
            result[k] = v
    inv_result = {}
    for k, v in result.items():
        inv_result[f"inv_{k}"] = f"was {v} by"
    result.update(inv_result)
    save_json_data(result, os.path.join(output_dir, 'transformed_relations.json'))
    return result.values()

def calculate_similarity(llm_instance, all_data, output_dir, filename):
    embedding_A = llm_instance.run_embeddings(all_data)
    embedding_B = llm_instance.run_embeddings(all_data)
    similarity = cosine_similarity(embedding_A, embedding_B)
    np.fill_diagonal(similarity, 0)
    np.save(os.path.join(output_dir, filename), similarity)

def load_vectorstore_db(llm_instance, dataset):
    data_dict = {}
    
    settings = load_json_data('config/data_embedding.json')
    for k in settings:
        settings[k] = re.sub(r'\bdataset\b', dataset, settings[k])

    collections = ['facts', 'rules']
    path_to_chroma_db = settings['output_vector_database_load']

    for collection in collections:
        path_to_collection = os.path.join(path_to_chroma_db, collection)
        data_dict[collection] = {}
        data_dict[collection]['vector_db'] = Chroma(persist_directory=path_to_collection, embedding_function=CustomEmbeddingFunction(llm_instance))

    return data_dict

def lookup_vector_db(query_search, filter, vector_db, llm_instance, top_k=None, threshold=None):
    """
    """
    documents = []
    search_kwargs = {
        'filter': filter,
        'k': 20,
    }

    if top_k:
        search_kwargs['k'] = top_k
    if threshold:
        search_kwargs['score_threshold'] = threshold
    
    try:
        retriever = vector_db.as_retriever(
            search_type='similarity_threshold' if threshold else 'similarity',
            search_kwargs=search_kwargs
        )
    except Exception as e:
        print(f"Error while filtering data from vectorstore: {e}")
        return documents

    try:
        documents = retriever.invoke(query_search)
    except Exception as e:
        print(f"Error while invoking retriever: {e}")
    
    return documents

def get_win_subgraph(test_data, data, learn_edges, window, win_start=0):
    unique_timestamp_id = np.unique(test_data[:, 3])
    win_subgraph = {}
    for timestamp_id in unique_timestamp_id:
        subgraph = ra.get_window_edges(data.all_idx, timestamp_id - win_start, learn_edges, window)
        win_subgraph[timestamp_id] = subgraph
    return win_subgraph

def filter_candidates(test_query, candidates, test_data):
    """
    Filter out those candidates that are also answers to the test query
    but not the correct answer.

    Parameters:
        test_query (np.ndarray): test_query
        candidates (dict): answer candidates with corresponding confidence scores
        test_data (np.ndarray): test dataset

    Returns:
        candidates (dict): filtered candidates
    """

    other_answers = test_data[
        (test_data[:, 0] == test_query[0])
        * (test_data[:, 1] == test_query[1])
        * (test_data[:, 2] != test_query[2])
        * (test_data[:, 3] == test_query[3])
    ]

    if len(other_answers):
        objects = other_answers[:, 2]
        for obj in objects:
            candidates.pop(obj, None)

    return candidates

def calculate_rank(test_query_answer, candidates, num_entities, setting="best"):
    """
    Calculate the rank of the correct answer for a test query.
    Depending on the setting, the average/best/worst rank is taken if there
    are several candidates with the same confidence score.

    Parameters:
        test_query_answer (int): test query answer
        candidates (dict): answer candidates with corresponding confidence scores
        num_entities (int): number of entities in the dataset
        setting (str): "average", "best", or "worst"

    Returns:
        rank (int): rank of the correct answer
    """

    rank = num_entities
    if test_query_answer in candidates:
        conf = candidates[test_query_answer]
        all_confs = list(candidates.values())
        all_confs = sorted(all_confs, reverse=True)
        ranks = [idx for idx, x in enumerate(all_confs) if x == conf]

        try:

            if setting == "average":
                rank = (ranks[0] + ranks[-1]) // 2 + 1
            elif setting == "best":
                rank = ranks[0] + 1
            elif setting == "worst":
                rank = ranks[-1] + 1
        except Exception as e:
            ranks

    return rank

