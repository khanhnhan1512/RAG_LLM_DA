# import pandas as pd
# import json
# from openai_llm.llm_init import LLM_Model
# from langchain_core.messages import HumanMessage, SystemMessage

# # Step 1: Read the content from draph.txt and extract 'rule' column values
# def extract_rules_from_file(filename):
#     # Read the CSV file
#     df = pd.read_csv(filename)
#     # Extracting 'rule' column values
#     return df['rule'].tolist()

# # Step 2: Create LLM prompt
# def create_llm_prompt(rules):
#     user_query = "Identify the top 10 rules that are not correct about reality based on the given data."
#     user_msg_content = f'''
#     Here is the user query: {user_query}

#     Here are the rules that need to be evaluated:
#     {rules}
#     '''
#     system_msg_content = '''
#     You are an expert in rule validation and factual accuracy. You will be given some rules extracted from the data.
#     Your task is to identify the top 10 rules that do not align with reality.
#     Your answer should be in JSON format:
#     {
#         "identified_rules": // a list of top 10 rules that are not correct about reality.
#     }
#     '''
    
#     return user_msg_content, system_msg_content

# # Step 3: Save results to JSON file
# def save_results_to_json(results, filename):
#     with open(filename, 'w') as file:
#         json.dump(results, file, indent=4)

# # Main execution
# if __name__ == "__main__":
#     # Extract rules from draph.txt
#     rules = extract_rules_from_file("draph.txt")
    
#     # Initialize LLM instance
#     llm_instance = LLM_Model()
    
#     # Create prompt for LLM
#     user_msg_content, system_msg_content = create_llm_prompt(rules)
    
#     # Create messages for LLM input
#     user_message = HumanMessage(content=user_msg_content)
#     system_message = SystemMessage(content=system_msg_content)
    
#     # Get response from LLM
#     answer_llm = llm_instance.run_task([system_message, user_message])
    
#     # Assuming answer_llm contains a valid JSON response with identified rules
#     identified_rules = answer_llm.get("identified_rules", [])
    
#     # Prepare results for saving
#     results = {
#         "identified_rules": identified_rules
#     }
    
#     # Save results to results.json
#     save_results_to_json(results, "results.json")

# import os
# import pandas as pd

# def create_rule_dict(file_path):
#     # Read the CSV file into a DataFrame
#     df = pd.read_csv(file_path)

#     # Initialize an empty dictionary to hold the results
#     rule_dict = {}

#     # Iterate through each row in the DataFrame
#     for index, row in df.iterrows():
#         head_rel = row['head_rel']
#         rule = row['rule']

#         # If the head_rel is not already in the dictionary, initialize it with an empty list
#         if head_rel not in rule_dict:
#             rule_dict[head_rel] = []

#         # Append the rule to the list corresponding to the head_rel key
#         rule_dict[head_rel].append(rule)

#     return rule_dict

# def save_rule_dict_to_csv(rule_dict, output_file_path):
#     # Ensure the directory exists
#     os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
#     # Convert the dictionary to a DataFrame
#     df = pd.DataFrame([(key, rule) for key, rules in rule_dict.items() for rule in rules], columns=['head_rel', 'rule'])
    
#     # Save the DataFrame to a CSV file
#     df.to_csv(output_file_path, index=False)

# file_path = 'result/icews14/stage_1/151224093916_r[1,2,3]_n200_exp_s42_rules.csv'
# result_dict = create_rule_dict(file_path)
# output_file_path = 'result/icews14/stage_1/rule_dict_output.csv'
# save_rule_dict_to_csv(result_dict, output_file_path)

# import os
# import json
# from openai_llm.llm_init import LLM_Model
# from langchain_core.messages import HumanMessage, SystemMessage

# def load_json(file_path):
#     """Load JSON data from a file."""
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return json.load(f)

# def create_llm_prompt(rule_head, extracted_rules, candidate_relations):
#     """Create prompt for LLM."""
#     user_msg_content = f'''
#     Please generate as many temporal logical rules as possible related to '{rule_head}' based on extracted temporal rules.

#     Here are a few examples:
#     Example 1:
#     Rule Head:
#     {rule_head}(X, Y, T)
#     Extracted Rules from Historical Data:
#     {extracted_rules}
#     '''

#     system_msg_content = f'''
#     Definition: "Temporal Logical Rules:\n Temporal Logical Rules \"{rule_head}(X0,Xl,Tl)<-R1(X0,X1,T0)&...&Rl(X(l-1),Xl,T(l-1))\" are rules used in temporal knowledge graph reasoning to predict relations between entities over time. They describe how the relation \"{rule_head}\" between entities \"X0\" and \"Xl\" evolves from past time steps \"Ti (i={{0,...,(l-1)}})\"(rule body) to the next \"Tl\" (rule head), strictly following the constraint \"T0 <= \u00b7\u00b7\u00b7 <= T(l-1) < Tl\".\n\n",
#     Context: "You are an expert in temporal knowledge graph reasoning, and please generate as many temporal logical rules as possible related to \"Rl\" based on extracted temporal rules.\n\nHere are a few examples: \n\nRule head: inv_Provide_humanitarian_aid(X0,Xl,Tl)\nSampled rules:\n\tinv_Provide_humanitarian_aid(X0,X1,T1)<-inv_Engage_in_diplomatic_cooperation(X0,X1,T0)\n\tinv_Provide_humanitarian_aid(X0,X3,T3)<-inv_Criticize_or_denounce(X0,X1,T0)&inv_Demand(X1,X2,T1)&Make_a_visit(X2,X3,T2)\n\tinv_Provide_humanitarian_aid(X0,X2,T2)<-Make_a_visit(X0,X1,T0)&Make_a_visit(X1,X2,T1)\nGenerated Temporal logic rules:\n\tinv_Provide_humanitarian_aid(X0,X1,T1)<-inv_Provide_aid(X0,X1,T0)\n\tinv_Provide_humanitarian_aid(X0,X2,T2)<-Make_an_appeal_or_request(X0,X1,T0)&inv_Consult(X1,X2,T1)\n\tinv_Provide_humanitarian_aid(X0,X3,T3)<-inv_Return,_release_person(s)(X0,X1,T0)&Return,_release_person(s)(X1,X2,T1)&Accuse(X2,X3,T2)\n\nRule head: Appeal_for_change_in_institutions,_regime(X0,Xl,Tl)\nSampled rules:\n\tAppeal_for_change_in_institutions,_regime(X0,X1,T1)<-inv_Engage_in_symbolic_act(X0,X1,T0)\n\tAppeal_for_change_in_institutions,_regime(X0,X2,T2)<-inv_Criticize_or_denounce(X0,X1,T0)&Make_pessimistic_comment(X1,X2,T1)\nGenerated Temporal logic rules:\n\tAppeal_for_change_in_institutions,_regime(X0,X1,T1)<-Make_an_appeal_or_request(X0,X1,T0)\n\tAppeal_for_change_in_institutions,_regime(X0,X2,T2)<-inv_Rally_support_on_behalf_of(X0,X1,T0)&Praise_or_endorse(X1,X2,T1)\n\tAppeal_for_change_in_institutions,_regime(X0,X3,T3)<-Appeal_for_change_in_institutions,_regime(X0,X1,T0)&Host_a_visit(X1,X2,T1)&inv_Criticize_or_denounce(X2,X3,T2)\n\tAppeal_for_change_in_institutions,_regime(X0,X2,T2)<-inv_Engage_in_symbolic_act(X0,X1,T0)&inv_Consult(X1,X2,T1)\n\nRule head: Appeal_for_economic_aid(X0,Xl,Tl)\nSampled rules:\n\tAppeal_for_economic_aid(X0,X1,T1)<-inv_Reduce_or_stop_military_assistance(X0,X1,T0)\n\tAppeal_for_economic_aid(X0,X2,T2)<-inv_Make_an_appeal_or_request(X0,X1,T0)&Make_statement(X1,X2,T1)\n\tAppeal_for_economic_aid(X0,X3,T3)<-inv_Demand(X0,X1,T0)&inv_Accede_to_demands_for_change_in_leadership(X1,X2,T1)&Accuse(X2,X3,T2)\nGenerated Temporal logic rules:\n\tAppeal_for_economic_aid(X0,X2,T2)<-Make_an_appeal_or_request(X0,X1,T0)&Appeal_for_military_aid(X1,X2,T1)\n\tAppeal_for_economic_aid(X0,X2,T2)<-inv_Express_intent_to_cooperate(X0,X1,T0)&Make_statement(X1,X2,T1)\n\tAppeal_for_economic_aid(X0,X1,T1)<-Make_an_appeal_or_request(X0,X1,T0)\n\n",

#     Temporal Logical Rules: Temporal Logical Rules \"{rule_head}(X0,Xl,Tl)<-R1(X0,X1,T0)&...&Rl(X(l-1),Xl,T(l-1))\" are rules used in temporal knowledge graph reasoning to predict relations between entities over time. They describe how the relation \"{rule_head}\" between entities \"X0\" and \"Xl\" evolves from past time steps \"Ti (i={{0,...,(l-1)}})\" (rule body) to the next \"Tl\" (rule head), strictly following the constraint \"T0 <= \u00b7\u00b7\u00b7 <= T(l-1) < Tl\".

#     For the relations in rule body, you are going to choose from the following candidates: {candidate_relations}.

#     Let's think step-by-step, please generate as many as possible most relevant temporal rules that are relative to \"{rule_head}(X0,Xl,Tl)\" based on the above sampled rules.

#     Return in JSON format:
#     {{
#         "{rule_head}": [
#             // list of new rules generated from the combination of candidate relations, the structure is like: "{rule_head}(X0,Xl,Tl)<-R1(X0,X1,T0)&...&Rl(X(l-1),Xl,T(l-1))"
#         ]
#     }}
# '''

#     return user_msg_content, system_msg_content

# def add_generated_rules_to_rule_dict(generated_rules, rule_dict):
#     """Add generated new rules to the existing rule_dict."""
#     for rule_head, new_rules in generated_rules.items():
#         if rule_head in rule_dict:
#             rule_dict[rule_head].extend(new_rules)
#         else:
#             rule_dict[rule_head] = new_rules
#     return rule_dict

# def generate_new_rules(top_k_relations, rule_dict, llm_instance):
#     """Generate new rules using LLM based on top k relations and existing rules."""
#     generated_rules = {}

#     for rule_head, candidate_relations in top_k_relations.items():
#         extracted_rules = rule_dict.get(rule_head, [])
        
#         # Create prompt for LLM
#         user_msg_content, system_msg_content = create_llm_prompt(rule_head, extracted_rules, candidate_relations)
        
#         # Create messages for LLM input
#         user_message = HumanMessage(content=user_msg_content)
#         system_message = SystemMessage(content=system_msg_content)
        
#         # Get response from LLM
#         answer_llm = llm_instance.run_task([system_message, user_message])

#         if isinstance(answer_llm, dict):
#             generated_rules.update(answer_llm)  # Merge generated rules into our dictionary

#     # Update the rule_dict with the new generated rules
#     updated_rule_dict = add_generated_rules_to_rule_dict(generated_rules, rule_dict)

#     return updated_rule_dict

# # Example usage
# top_k_relations_file_path = 'result/icews14/stage_2/top_k_relations.json'
# rule_dict_file_path = 'result/icews14/stage_2/rule_dict_output.json'

# # Load top k relations and current rule heads with rules
# top_k_relations = load_json(top_k_relations_file_path)
# rule_dict = load_json(rule_dict_file_path)

# # Initialize LLM instance
# llm_instance = LLM_Model()

# # Generate new rules based on top k relations and existing rules
# new_generated_rules = generate_new_rules(top_k_relations, rule_dict, llm_instance)

# # Output directory
# output_dir = 'result/icews14/stage_2'

# # Save the generated rules in the specified format
# output_file_path = os.path.join(output_dir, 'generated_rules_added_output.json')
# with open(output_file_path, 'w', encoding='utf-8') as f:
#     json.dump(new_generated_rules, f, ensure_ascii=False, indent=4)


import pandas as pd

def filter_llm_generated_rules(file_path, thresholds):
    """
    Filters LLM generated rules based on given thresholds for specified columns.

    Parameters:
    - file_path: str, path to the CSV file containing LLM generated rules.
    - thresholds: dict, mapping of column names to their corresponding threshold values.

    Returns:
    - None; writes two files: one for kept content and one for eliminated content.
    """
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Filter based on thresholds
    temp_df = df.copy()

    # Apply filtering based on each threshold
    for column, threshold in thresholds.items():
        if column in temp_df.columns:
            # Calculate the cutoff value based on the threshold
            cutoff_index = int(len(temp_df) * threshold)
            # Sort by the column and keep the top values
            temp_df = temp_df.nlargest(cutoff_index, column)

    # Get kept and eliminated content
    kept_content = temp_df
    eliminated_content = df[~df.index.isin(temp_df.index)]

    # Define output file paths
    kept_file_path = 'result/icews14/stage_2/kept_llm_generated_rules.csv'
    eliminated_file_path = 'result/icews14/stage_2/eliminated_llm_generated_rules.csv'

    # Write kept content to a new CSV file
    kept_content.to_csv(kept_file_path, index=False)

    # Write eliminated content to a new CSV file
    eliminated_content.to_csv(eliminated_file_path, index=False)

# Example usage
file_path = 'result/icews14/stage_2/20241219_llm_generated_llm_rules.csv'
thresholds = {
    'confidence_score': 0.9,
    'body_supp_count': 0.8
}

filter_llm_generated_rules(file_path, thresholds)

