import threading
import time
import tiktoken
import json
from openai_llm.llm_connect import connect_llm_chat, connect_llm_embeddings

from openai_llm.utils import read_settings

MAX_TOKENS_VALUES = {
    'gpt-4o': {'accepted': 64_000, 'total': 500_000, 'nb_requests': 2_000},
    'gpt-4o-mini': {'accepted': 64_000, 'total': 1_000_000, 'nb_requests': 250_000},
    'text-embedding-3-large': {'accepted': 64_000, 'total': 200_000, 'nb_requests': 1000}
}

COST_TOKENS_1000TK = {
    'gpt-4o': {'input': 0.005, 'output': 0.015},
    'gpt-4o-mini': {'input': 0.000153, 'output': 0.00062},
    'text-embedding-3-large': {'output': 0.000122}
}

class LLM_Model:
    def __init__(self):
        self.settings = read_settings("config/llm.config")

        self.connect()

        self.reset_llm_instance()

        self.lock = threading.Lock()

    def connect(self):
        # Model connection
        self.chat_model = connect_llm_chat(self.settings["chat_model"])
        self.embedding_model = connect_llm_embeddings(self.settings["embedding_model"])

        # Token management
        self.tokenizer_chat_model = tiktoken.encoding_for_model(self.settings["chat_model"]["model"])
        self.tokenizer_embedding_model = tiktoken.encoding_for_model(self.settings["embedding_model"]["model"])
        self.cost_chat_tokens_info = COST_TOKENS_1000TK[self.settings["chat_model"]["model"]]
        self.cost_embedding_tokens_info = COST_TOKENS_1000TK[self.settings["embedding_model"]["model"]]

    def reset_llm_instance(self):

        # Cost tracking
        self.costs_chat = {
            'name': self.settings["chat_model"]["model"],
            'input': {'total_tokens': 0, 'cost': 0},
            'output': {'total_tokens': 0, 'cost': 0}
        }

        self.costs_embedding = {
            'name': self.settings["embedding_model"]["model"],
            'output': {'total_tokens': 0, 'cost': 0}
        }

        # Quota information / Limit information
        max_tokens_query = MAX_TOKENS_VALUES[self.settings["chat_model"]["model"]]
        self.quota_query = {
            'timer': None,
            'free_tokens': max_tokens_query['total'],
            'accepted_tokens': max_tokens_query['accepted'],
            'quota_limit_tokens': max_tokens_query['total'],
            'request_timer': None,
            'current_requests': 0,
            'max_nb_requests': max_tokens_query['nb_requests']
        }

        max_tokens_embeddings = MAX_TOKENS_VALUES[self.settings["embedding_model"]["model"]]
        self.quota_embeddings = {
            'timer': None,
            'free_tokens': max_tokens_embeddings['total'],
            'accepted_tokens': max_tokens_embeddings['accepted'],
            'quota_limit_tokens': max_tokens_embeddings['total'],
            'request_timer': None,
            'current_requests': 0,
            'max_nb_requests': max_tokens_embeddings['nb_requests']
        }

        self.max_documents_retrieve = 20

    def num_tokens(self, tokenizer, text):
        return len(tokenizer.encode(text))
    
    def pause(self, seconds=2):
        time.sleep(seconds)

    def check_and_update_quota(self, input_tokens, quota_info, current_time, time_interval=60):
        with self.lock:
            if quota_info['timer'] is None or (current_time - quota_info['timer']) > time_interval:
                quota_info['timer'] = current_time
                quota_info['free_tokens'] = quota_info['quota_limit_tokens']
            
            if quota_info['request_timer'] is None or (current_time - quota_info['request_timer']) > time_interval:
                quota_info['request_timer'] = current_time
                quota_info['current_requests'] = 0

            # Check if we have enough free tokens and requests
            if (input_tokens <= quota_info["free_tokens"] and 
                quota_info["current_requests"] < quota_info["max_nb_requests"]):
                quota_info["free_tokens"] -= input_tokens
                quota_info["current_requests"] += 1
                proceed = True
                msg = None
            else:
                msg = []
                if input_tokens > quota_info["free_tokens"]:
                    remaining_time_tokens = time_interval - (current_time - quota_info["timer"])
                    msg.append(f"Waiting for free tokens. Free tokens: {quota_info['free_tokens']}, "
                               f"Input tokens: {input_tokens}, Remaining time: {remaining_time_tokens:.2f} seconds")
                if quota_info["current_requests"] >= quota_info["max_nb_requests"]:
                    remaining_time_requests = time_interval - (current_time - quota_info["request_timer"])
                    msg.append(f"Waiting for free requests. Max requests reached. ",
                               f"Remaining time: {remaining_time_requests:.2f} seconds")
        return proceed, msg
    
    def run_task(self, msg):
        answer_llm = {}

        retry = 0
        max_retry = 1

        while retry < max_retry:
            try: 
                answer_llm = self.run_query(msg)
                answer_llm = json.loads(answer_llm.content)

                if answer_llm == {}:
                    retry += 1
                    continue
                else:
                    break
            except Exception as e:
                print(f"Error running task: {e}")
                retry += 1
                self.pause()
        return answer_llm

    def run_query(self, msg):
        input_tokens = sum(self.num_tokens(self.tokenizer_chat_model, m.content) for m in msg)

        while True:
            current_time = time.time()
            proceed, msg_list = self.check_and_update_quota(input_tokens, self.quota_query, current_time)
            if proceed:
                break
            else:
                if msg_list:
                    for msg in msg_list:
                        print(msg)
                self.pause(1)

        result = self.chat_model.invoke(msg)

        with self.lock:
            self.update_cost_chat_model(result)
        return result

    def run_embedding(self, documents):
        input_tokens = sum(self.num_tokens(self.tokenizer_embedding_model, doc) for doc in documents)

        while True:
            current_time = time.time()
            proceed, msg_list = self.check_and_update_quota(input_tokens, self.quota_embeddings, current_time)
            if proceed:
                break
            else:
                if msg_list:
                    for msg in msg_list:
                        print(msg)
                self.pause(1)

        doc_result = self.embedding_model.embed_documents(documents)

        with self.lock:
            self.update_cost_embedding_model(documents)
        return doc_result
    
    def update_cost_chat_model(self, result):
        self.costs_chat['input']['total_tokens'] += result.usage_metadata['input_tokens']
        self.costs_chat['output']['total_tokens'] += result.usage_metadata['output_tokens']

        self.costs_chat['input']['cost'] = round(self.costs_chat['input']['total_tokens'] * self.cost_chat_tokens_info['input'] / 1000, 4)
        self.costs_chat['output']['cost'] = round(self.costs_chat['output']['total_tokens'] * self.cost_chat_tokens_info['output'] / 1000, 4)

    def update_cost_embedding_model(self, documents):
        total_tokens = sum(self.num_tokens(self.tokenizer_embedding_model, doc) for doc in documents)
        self.costs_embedding['output']['total_tokens'] += total_tokens

        self.costs_embedding['output']['cost'] = round(self.costs_embedding['output']['total_tokens'] * self.cost_embedding_tokens_info['output'] / 1000, 4)
    
    def get_cost(self):
        return {
            'chat_model': self.costs_chat,
            'embedding_model': self.costs_embedding
        }




