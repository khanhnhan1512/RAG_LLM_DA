import ast

def read_settings(file_path):
    """
    
    """
    with open(file_path, 'r') as f:
        setting_dict = f.read()
        setting_dict = ast.literal_eval(setting_dict)
    return setting_dict