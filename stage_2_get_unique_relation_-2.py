# def extract_unique_relations(input_file_path, output_file_path):
#     seen_relations = set()  # To keep track of seen relations
#     unique_relations = []   # To maintain the order of unique relations

#     with open(input_file_path, 'r', encoding='utf-8') as infile:
#         for line in infile:
#             parts = line.strip().split('\t')  # Split the line by tab
#             if len(parts) > 1:  # Ensure there is at least two parts
#                 relation = parts[1]  # The second object is the relation
#                 if relation not in seen_relations:
#                     seen_relations.add(relation)  # Add to seen set
#                     unique_relations.append(relation)  # Maintain order

#     # Write the unique relations to the output file
#     with open(output_file_path, 'w', encoding='utf-8') as outfile:
#         for relation in unique_relations:
#             outfile.write(f"{relation}\n")  # Write each relation on a new line

# input_file_path = r'datasets\icews14\valid.txt'
# output_file_path = r'result\icews14\stage_2\unique_relations.txt'

# # Extract unique relations and save to file
# extract_unique_relations(input_file_path, output_file_path)


def extract_unique_relations(valid_file_path, train_file_path, output_file_path):
    seen_relations = set()  # To keep track of seen relations
    unique_relations = []   # To maintain the order of unique relations

    # First, process valid.txt to ensure its relations are prioritized
    with open(valid_file_path, 'r', encoding='utf-8') as valid_file:
        for line in valid_file:
            parts = line.strip().split('\t')  # Split the line by tab
            if len(parts) > 1:  # Ensure there is at least two parts
                relation = parts[1]  # The second object is the relation
                if relation not in seen_relations:
                    seen_relations.add(relation)  # Add to seen set
                    unique_relations.append(relation)  # Maintain order

    # Then, process train.txt to add any new unique relations
    with open(train_file_path, 'r', encoding='utf-8') as train_file:
        for line in train_file:
            parts = line.strip().split('\t')  # Split the line by tab
            if len(parts) > 1:  # Ensure there is at least two parts
                relation = parts[1]  # The second object is the relation
                if relation not in seen_relations:
                    seen_relations.add(relation)  # Add to seen set
                    unique_relations.append(relation)  # Maintain order

    # Write the unique relations to the output file
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for relation in unique_relations:
            outfile.write(f"{relation}\n")  # Write each relation on a new line

# File paths
valid_file_path = r'datasets\icews14\valid.txt'
train_file_path = r'datasets\icews14\train.txt'
output_file_path = r'result\icews14\stage_2\unique_relations.txt'

# Extract unique relations and save to file
extract_unique_relations(valid_file_path, train_file_path, output_file_path)
