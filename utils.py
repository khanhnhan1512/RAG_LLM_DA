import json
import os

def load_json_data(file_path):
    try:
        print(f"Loading data from {file_path}")
        with open(file_path, 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        data = None
    return data
