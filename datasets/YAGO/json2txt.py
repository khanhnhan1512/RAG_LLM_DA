import json

# Specify the path to your JSON file
dataset = "../icews18/"
files = ["entity2id","relation2id"]
file = files[0]
json_file = file+'.json'
txt_file = file+'.txt'
json_file_path = dataset + json_file
txt_file_path = dataset + txt_file

# Open the JSON file in read mode
with open(json_file_path, 'r', encoding='utf-8') as file:
    # Load JSON data from the file into a variable
    data = json.load(file)

# Now 'data' contains the contents of the JSON file as a Python dictionary or list
# You can access and use this data as needed
print("Contents of the JSON file:")
print(data)

text_lines = []
# Example: Accessing specific data from the loaded JSON
# Assuming 'data' is a dictionary as per your example
if isinstance(data, dict):
    for key, value in data.items():
        text_lines.append(f"{key}\t{value}")

text_lines.append("\n")
result_text = "\n".join(text_lines)

# Write the result to a text file
with open(txt_file_path, "w", encoding='utf-8') as file:
    file.write(result_text)
print("saved at ", txt_file_path)