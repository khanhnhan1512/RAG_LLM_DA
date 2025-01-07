# import csv
# from collections import defaultdict

# def read_csv_to_dict(file_path):
#     """Read a CSV file and return a list of dictionaries."""
#     with open(file_path, mode='r', encoding='utf-8') as csv_file:
#         reader = csv.DictReader(csv_file)
#         return [row for row in reader]

# def filter_rows(rows, thresholds):
#     """Filter rows based on given thresholds."""
#     filtered = []
#     for row in rows:
#         if all(float(row[col]) >= threshold for col, threshold in thresholds.items()):
#             filtered.append(row)
#     return filtered

# def merge_results(column_names, thresholds, *file_paths):
#     """Merge results from multiple CSV files based on thresholds."""
#     merged_results = defaultdict(list)  # To hold rows grouped by head_rel
#     unique_rules = set()  # To track unique rules

#     # Read and filter each file
#     for file_path in file_paths:
#         rows = read_csv_to_dict(file_path)
#         filtered_rows = filter_rows(rows, thresholds)

#         for row in filtered_rows:
#             head_rel = row['head_rel']
#             rule = row['rule']

#             # Only add if the rule is not already included
#             if rule not in unique_rules:
#                 unique_rules.add(rule)
#                 # Keep only the required columns
#                 merged_results[head_rel].append({col: row[col] for col in column_names})

#     return merged_results

# def write_merged_results(merged_results, output_file_path):
#     """Write merged results to a CSV file."""
#     with open(output_file_path, mode='w', encoding='utf-8', newline='') as outfile:
#         writer = csv.writer(outfile)

#         # Write header
#         writer.writerow(next(iter(merged_results.values()))[0].keys())

#         # Write rows in order of head_rel
#         for head_rel in merged_results.keys():
#             for row in merged_results[head_rel]:
#                 writer.writerow(row.values())

# # File paths
# scored_historical_rules_file = r'result\icews14\stage_2\saved_results\01_merged_results.csv'
# scored_current_rules_file = r'result\icews14\stage_2\saved_results\02_merged_results.csv'
# output_file_path = r'result\icews14\stage_2\saved_results\0102_merged_results.csv'

# # Define column names to project and thresholds for filtering
# column_names = ['confidence_score', 'rule_supp_count', 'body_supp_count', 'head_supp_count', 'rule', 'head_rel']
# thresholds = {
#     'confidence_score': 0.1,
#     'body_supp_count': 2
# }

# # Step 3: Merge results from both files
# merged_results = merge_results(column_names, thresholds, scored_historical_rules_file, scored_current_rules_file)

# # Step 4: Write the final result to a CSV file
# write_merged_results(merged_results, output_file_path)

import csv
from collections import defaultdict

def read_csv_to_dict(file_path):
    """Read a CSV file and return a list of dictionaries."""
    with open(file_path, mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        return [row for row in reader]

def filter_rows(rows, thresholds):
    """Filter rows based on given thresholds."""
    filtered = []
    for row in rows:
        if all(float(row[col]) >= threshold for col, threshold in thresholds.items()):
            filtered.append(row)
    return filtered

def merge_results(column_names, thresholds, file_paths):
    """Merge results from multiple CSV files based on thresholds."""
    merged_results = defaultdict(list)  # To hold rows grouped by head_rel
    unique_rules = set()  # To track unique rules

    # Read and filter each file
    for file_path in file_paths:
        rows = read_csv_to_dict(file_path)
        filtered_rows = filter_rows(rows, thresholds)

        for row in filtered_rows:
            head_rel = row['head_rel']
            rule = row['rule']

            # Only add if the rule is not already included
            if rule not in unique_rules:
                unique_rules.add(rule)
                # Keep only the required columns
                merged_results[head_rel].append({col: row[col] for col in column_names})

    return merged_results

def write_merged_results(merged_results, output_file_path):
    """Write merged results to a CSV file."""
    with open(output_file_path, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)

        # Write header (assuming all head_rel have the same structure)
        writer.writerow(next(iter(merged_results.values()))[0].keys())

        # Write rows in order of head_rel
        for head_rel in merged_results.keys():
            for row in merged_results[head_rel]:
                writer.writerow(row.values())

# File paths for multiple CSV files
file_paths = [
    # r'result\icews14\stage_2\saved_results\01_scored_current_rules_on_current_data.csv',
    # r'result\icews14\stage_2\saved_results\02_scored_current_rules_on_current_data.csv',
    # r'result\icews14\stage_2\saved_results\03_scored_current_rules_on_current_data.csv',
    # r'result\icews14\stage_2\saved_results\04_scored_current_rules_on_current_data.csv',
    # r'result\icews14\stage_2\saved_results\05_scored_current_rules_on_current_data.csv',
    # r'result\icews14\stage_2\saved_results\06_scored_current_rules_on_current_data.csv',
    # r'result\icews14\stage_2\saved_results\07_scored_current_rules_on_current_data.csv',
    # r'result\icews14\stage_2\saved_results\08_scored_current_rules_on_current_data.csv'

    # r'result\icews14\stage_2\merged_results\merged_current_rules_on_current_data.csv',
    # r'result\icews14\stage_2\merged_results\merged_historical_rules_on_current_data.csv',
    # r'result\icews14\stage_2\merged_results\merged_historical_rules_on_historical_data.csv',
    r'result\icews14\stage_2\filtered_results.csv',
]

# Define column names to project and thresholds for filtering
column_names = ['confidence_score', 'rule_supp_count', 'body_supp_count', 'head_supp_count', 'rule', 'head_rel']
thresholds = {
    'confidence_score': 0.3,
    'body_supp_count': 3
}

# Step 3: Merge results from all specified files
merged_results = merge_results(column_names, thresholds, file_paths)

# Step 4: Write the final result to a CSV file
output_file_path = r'result\icews14\stage_2\merged_results\filtered_historical_rules_and_current_rules_0,3.csv'
write_merged_results(merged_results, output_file_path)

print(f"Merged results written to {output_file_path}")

