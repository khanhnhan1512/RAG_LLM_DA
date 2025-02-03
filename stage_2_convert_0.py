import csv
from collections import defaultdict
import json

def read_csv_to_dict(file_path):
    # Initialize a defaultdict to hold lists for each unique head_rel
    result_dict = defaultdict(list)

    # Open the CSV file and read its contents
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        # Iterate through each row in the CSV
        for row in reader:
            head_rel = row['head_rel']
            rule = row['rule']
            # Append the rule to the list corresponding to the head_rel key
            result_dict[head_rel].append(rule)

    # Return the dictionary with lists
    return result_dict

def save_dict_to_json(data, output_file):
    # Save the dictionary to a JSON file
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

# input_file_path = 'result\icews14\stage_2\common_rule_heads_historical_data.csv'
# output_file_path = 'result\icews14\stage_2\historical_data_rule_dict_output.json'

# # input_file_path = r'result\icews14\stage_2\merged_results\merged_historical_rules_and_current_rules_0,4.csv'
# # output_file_path = 'result/icews14/stage_2/high_quality_example_rules.json'


# input_file_path = 'result\GDELT\stage_2\common_rule_heads_historical_data.csv'
# output_file_path = 'result\GDELT\stage_2\historical_data_rule_dict_output.json'


input_file_path = 'result\YAGO\stage_2\common_rule_heads_historical_data.csv'
output_file_path = 'result\YAGO\stage_2\historical_data_rule_dict_output.json'

# input_file_path = r'result\icews14\stage_2\merged_results\merged_historical_rules_and_current_rules_0,4.csv'
# output_file_path = 'result/icews14/stage_2/high_quality_example_rules.json'

# Read CSV and create dictionary
head_rel_dict = read_csv_to_dict(input_file_path)

# Save the resulting dictionary to a JSON file
save_dict_to_json(head_rel_dict, output_file_path)
