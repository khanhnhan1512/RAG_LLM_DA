import numpy as np

def score_1(llm_rank, candidate_ts, test_query_ts, weight=0.0, lmbda=0.1):
    return weight * (1.0 - llm_rank/10.0) + (1.0 - weight) * np.exp(lmbda * (candidate_ts - test_query_ts))