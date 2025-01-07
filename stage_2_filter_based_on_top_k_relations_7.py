import pandas as pd
import json
import re

def filter_rules(csv_file_path, json_file_path, output_file_path):
    # Load the JSON file containing inverse relations
    with open(json_file_path, 'r') as json_file:
        inv_relations = json.load(json_file)

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Function to check if any rule matches the inv relations for head_rel
    def matches_inv_rule(rule_part, head_rel):
        # Split rule_part by '&' to check each individual rule
        rules = rule_part.split('&')
        
        # Check against inv relations for each rule
        for key, values in inv_relations.items():
            if key.lower() == f'inv_{head_rel.lower()}' or key.lower() == f'{head_rel.lower()}':
                for value in values:
                    for rule in rules:
                        # Extracting the action part from the rule (e.g., Make_statement)
                        action = re.sub(r'\(.*\)', '', rule).strip()  # Remove parameters from rules
                        if action.lower() == value.lower():
                            return True
        return False

    # Filter rows based on matching rules
    filtered_rows = []

    for index, row in df.iterrows():
        # Extract rule part after '<-'
        rule_part = row['rule'].split('<-')[1] if '<-' in row['rule'] else ''
        
        # Check if head_rel matches any inv rule
        if matches_inv_rule(rule_part, row['head_rel']):
            filtered_rows.append(row)

    # Create a DataFrame from filtered rows
    filtered_df = pd.DataFrame(filtered_rows)

    # Write filtered results to a new CSV file
    filtered_df.to_csv(output_file_path, index=False)

# Define file paths
csv_file_path = r'C:\Users\Admin\Documents\Graduation thesis\RAG_LLM_DA\result\icews14\stage_2\merged_results\merged_historical_rules_and_current_rules_0,1.csv'
json_file_path = r'C:\Users\Admin\Documents\Graduation thesis\RAG_LLM_DA\result\icews14\stage_2\get_top_k_relations.json'
output_file_path = r'C:\Users\Admin\Documents\Graduation thesis\RAG_LLM_DA\result\icews14\stage_2\filtered_results_based_on_top_k_relations.csv'

# Call the function to filter rules
filter_rules(csv_file_path, json_file_path, output_file_path)

print(f'Filtered results written to {output_file_path}')
