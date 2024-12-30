import sys
import os
import shutil
import pickle
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from openai_llm.llm_init import LLM_Model
from process_embedding.process_data import Process