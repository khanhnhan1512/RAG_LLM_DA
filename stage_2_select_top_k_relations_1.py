
import os
import json
from openai_llm.llm_init import LLM_Model
from langchain_core.messages import HumanMessage, SystemMessage

def load_relations(file_path):
    """Load relations from a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def load_rule_head_dict(file_path):
    """Load rule head dictionary from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_llm_prompt(rule_head, rules, relation_list, k=10):
    """Create prompt for LLM."""
    user_query = f"Identify the top {k} relevant relations from the following list that can induce or combine to induce the rule head '{rule_head}' based on reality knowledge."
    user_msg_content = f'''
    Here is the user query: {user_query}

    Here are the current logical combinations of rules associated with the rule head '{rule_head}':
    {rules}

    Here is the list of relations to consider:
    {relation_list}
    
    Please ensure to select relations that are not already included in the current rules.
    '''
    
    system_msg_content = '''
    Definition: "Temporal Logical Rules:\n Temporal Logical Rules \"{head}(X0,Xl,Tl)<-R1(X0,X1,T0)&...&Rl(X(l-1),Xl,T(l-1))\" are rules used in temporal knowledge graph reasoning to predict relations between entities over time. They describe how the relation \"{head}\" between entities \"X0\" and \"Xl\" evolves from past time steps \"Ti (i={{0,...,(l-1)}})\"(rule body) to the next \"Tl\" (rule head), strictly following the constraint \"T0 <= \u00b7\u00b7\u00b7 <= T(l-1) < Tl\".\n\n",
    You are an expert in rule validation and factual accuracy. You will be given some rules extracted from the data.
    Your task is to identify the top {k} relevant relations that can logically induce or combine to induce the rule head '{rule_head}'.
    Your answer should be in JSON format:
    {
        "identified_relation_list": // a list of top {k} relevant relations.
    }
    '''

    return user_msg_content, system_msg_content

def select_top_k_most_relevant(rule_head_dict, relation_list, k, llm_instance):
    """Select top k most relevant relations for each rule head using an LLM."""
    top_k_relations = {}
    n = 0

    for rule_head, rules in rule_head_dict.items():
        # Create prompt for LLM using rule_head and relation_list
        user_msg_content, system_msg_content = create_llm_prompt(rule_head, rules, relation_list, k)
        
        # Create messages for LLM input
        user_message = HumanMessage(content=user_msg_content)
        system_message = SystemMessage(content=system_msg_content)
        
        # Get response from LLM
        answer_llm = llm_instance.run_task([system_message, user_message])

        # Assuming answer_llm returns a JSON-like structure with identified_relation_list
        identified_relation_list = answer_llm.get("identified_relation_list", [])

        # Store top k relevant relations in the output dictionary
        top_k_relations[rule_head] = [{relation: 0} for relation in identified_relation_list[:k]]

        if n == 3:
            break

        n += 1

    return top_k_relations

def transform_output(top_k_relations):
    """Transform output to required format."""
    transformed_output = {}
    for rule_head, relations in top_k_relations.items():
        transformed_output[rule_head] = [{relation: 0} for relation in relations]
    return transformed_output

# Example usage
relation_file_path = 'datasets/icews14/relations.txt'
rule_dict_file_path = 'result/icews14/stage_2/rule_dict_output.json'

# Load relations and rule heads with rules
relations = load_relations(relation_file_path)
rule_head_dict = load_rule_head_dict(rule_dict_file_path)

# Initialize LLM instance
llm_instance = LLM_Model()

# Select top k relevant relations
k = 10
result = select_top_k_most_relevant(rule_head_dict, relations, k, llm_instance)

# Output directory
output_dir = 'result\icews14\stage_2'

# Save the result to a JSON file
output_file_path = os.path.join(output_dir, 'top_k_relations.json')
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)