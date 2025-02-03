import csv
import re

def read_unique_relations(file_path):
    """Read unique relations from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return {line.strip() for line in file}  # Use a set for fast lookup

def split_rules_by_relation(input_csv_path, unique_relations, common_output_path, specific_output_path):
    """Split rules based on their head_rel and write to respective files."""
    with open(input_csv_path, mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        
        # Prepare output files
        with open(common_output_path, mode='w', encoding='utf-8', newline='') as common_file, \
             open(specific_output_path, mode='w', encoding='utf-8', newline='') as specific_file:
            
            common_writer = csv.writer(common_file)
            specific_writer = csv.writer(specific_file)

            # Write header to both output files
            common_writer.writerow(reader.fieldnames)
            specific_writer.writerow(reader.fieldnames)

            # Iterate through each row in the CSV
            for row in reader:
                head_rel = row['head_rel']
                # Check if head_rel is in unique relations or matches after removing "inv_"
                if head_rel in unique_relations or re.sub(r'^inv_', '', head_rel) in unique_relations:
                    common_writer.writerow(row.values())  # Write to common file
                else:
                    specific_writer.writerow(row.values())  # Write to specific file

# File paths
# unique_relations_file = r'result\icews14\stage_2\unique_relations.txt'
# input_csv_file = r'result\icews14\stage_1\kulc_05_historical_random_walk_random_rules.csv'
# common_output_file = r'result\icews14\stage_2\common_rule_heads_historical_data.csv'
# specific_output_file = r'result\icews14\stage_2\saved_results\kulc_05_scored_historical_rules_on_historical_data.csv'


# unique_relations_file = r'result\GDELT\stage_2\unique_relations.txt'
# input_csv_file = r'result\GDELT\stage_1\08_historical_random_walk_random_rules.csv'
# common_output_file = r'result\GDELT\stage_2\common_rule_heads_historical_data.csv'
# specific_output_file = r'result\GDELT\stage_2\saved_results\08_scored_historical_rules_on_historical_data.csv'

unique_relations_file = r'result\YAGO\stage_2\unique_relations.txt'
input_csv_file = r'result\YAGO\stage_1\08_historical_random_walk_random_rules.csv'
common_output_file = r'result\YAGO\stage_2\common_rule_heads_historical_data.csv'
specific_output_file = r'result\YAGO\stage_2\saved_results\08_scored_historical_rules_on_historical_data.csv'


# Step 1: Read unique relations
unique_relations = read_unique_relations(unique_relations_file)

# Step 2: Split rules by relation and write to output files
split_rules_by_relation(input_csv_file, unique_relations, common_output_file, specific_output_file)
