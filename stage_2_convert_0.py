import csv
from collections import defaultdict
import os
import pandas as pd
import json

def read_csv_to_dict(file_path):
    # Initialize a defaultdict to hold sets for each unique head_rel
    result_dict = defaultdict(set)

    # Open the CSV file and read its contents
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        # Iterate through each row in the CSV
        for row in reader:
            head_rel = row['head_rel']
            rule = row['rule']
            # Add the rule to the set corresponding to the head_rel key
            result_dict[head_rel].add(rule)

    # Convert sets back to lists to maintain the desired output format
    return {k: list(v) for k, v in result_dict.items()}

def save_dict_to_json(data, output_file):
    # Save the dictionary to a JSON file
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

input_file_path = 'result/icews14/stage_1/151224093916_r[1,2,3]_n200_exp_s42_rules.csv'
output_file_path = 'result/icews14/stage_2/rule_dict_output.json'

# Read CSV and create dictionary
head_rel_dict = read_csv_to_dict(input_file_path)

# Save the resulting dictionary to a JSON file
save_dict_to_json(head_rel_dict, output_file_path)