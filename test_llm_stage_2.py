import pandas as pd
import json
from openai_llm.llm_init import LLM_Model
from langchain_core.messages import HumanMessage, SystemMessage

# Step 1: Read the content from draph.txt and extract 'rule' column values
def extract_rules_from_file(filename):
    # Read the CSV file
    df = pd.read_csv(filename)
    # Extracting 'rule' column values
    return df['rule'].tolist()

# Step 2: Create LLM prompt
def create_llm_prompt(rules):
    user_query = "Identify the top 10 rules that are not correct about reality based on the given data."
    user_msg_content = f'''
    Here is the user query: {user_query}

    Here are the rules that need to be evaluated:
    {rules}
    '''
    system_msg_content = '''
    You are an expert in rule validation and factual accuracy. You will be given some rules extracted from the data.
    Your task is to identify the top 10 rules that do not align with reality.
    Your answer should be in JSON format:
    {
        "identified_rules": // a list of top 10 rules that are not correct about reality.
    }
    '''
    
    return user_msg_content, system_msg_content

# Step 3: Save results to JSON file
def save_results_to_json(results, filename):
    with open(filename, 'w') as file:
        json.dump(results, file, indent=4)

# Main execution
if __name__ == "__main__":
    # Extract rules from draph.txt
    rules = extract_rules_from_file("draph.txt")
    
    # Initialize LLM instance
    llm_instance = LLM_Model()
    
    # Create prompt for LLM
    user_msg_content, system_msg_content = create_llm_prompt(rules)
    
    # Create messages for LLM input
    user_message = HumanMessage(content=user_msg_content)
    system_message = SystemMessage(content=system_msg_content)
    
    # Get response from LLM
    answer_llm = llm_instance.run_task([system_message, user_message])
    
    # Assuming answer_llm contains a valid JSON response with identified rules
    identified_rules = answer_llm.get("identified_rules", [])
    
    # Prepare results for saving
    results = {
        "identified_rules": identified_rules
    }
    
    # Save results to results.json
    save_results_to_json(results, "results.json")