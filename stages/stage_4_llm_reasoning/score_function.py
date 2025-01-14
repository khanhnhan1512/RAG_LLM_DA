import numpy as np

def score_1(llm_rank, candidate_ts, test_query_ts, weight=0.0):
    return weight * (1.0 - llm_rank/10.0) + (1 - weight) * np.exp(np.abs(candidate_ts - test_query_ts))