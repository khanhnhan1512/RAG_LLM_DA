import os
import json
from openai_llm.llm_init import LLM_Model
from langchain_core.messages import HumanMessage, SystemMessage

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_llm_prompt(rule_head, extracted_rules, candidate_relations, k):
    """Create prompt for LLM."""
    user_msg_content = f'''
    Let's think step-by-step, please generate as many as possible temporal logical rules related to '{rule_head}' based on extracted temporal rules.

    Rule Head:
    {rule_head}(X, Y, T)
    Extracted Rules from Current Data:
    {extracted_rules}
    '''
    system_msg_content = f'''
    Definition: "Temporal Logical Rules:\n Temporal Logical Rules \"{rule_head}(X0,Xl,Tl)<-R1(X0,X1,T0)&...&Rl(X(l-1),Xl,T(l-1))\" are rules used in temporal knowledge graph reasoning to predict relations between entities over time. They describe how the relation \"{rule_head}\" between entities \"X0\" and \"Xl\" evolves from past time steps \"Ti (i={{0,...,(l-1)}})\"(rule body) to the next \"Tl\" (rule head), strictly following the constraint \"T0 <= \u00b7\u00b7\u00b7 <= T(l-1) < Tl\".\n\n",
    Context: "You are an expert in temporal knowledge graph reasoning, and please generate as many temporal logical rules as possible related to \"Rl\" based on extracted temporal rules.\n\nHere are a few examples: \n\nRule head: inv_Provide_humanitarian_aid(X0,Xl,Tl)\nSampled rules:\n\tinv_Provide_humanitarian_aid(X0,X1,T1)<-inv_Engage_in_diplomatic_cooperation(X0,X1,T0)\n\tinv_Provide_humanitarian_aid(X0,X3,T3)<-inv_Criticize_or_denounce(X0,X1,T0)&inv_Demand(X1,X2,T1)&Make_a_visit(X2,X3,T2)\n\tinv_Provide_humanitarian_aid(X0,X2,T2)<-Make_a_visit(X0,X1,T0)&Make_a_visit(X1,X2,T1)\nGenerated Temporal logic rules:\n\tinv_Provide_humanitarian_aid(X0,X1,T1)<-inv_Provide_aid(X0,X1,T0)\n\tinv_Provide_humanitarian_aid(X0,X2,T2)<-Make_an_appeal_or_request(X0,X1,T0)&inv_Consult(X1,X2,T1)\n\tinv_Provide_humanitarian_aid(X0,X3,T3)<-inv_Return,_release_person(s)(X0,X1,T0)&Return,_release_person(s)(X1,X2,T1)&Accuse(X2,X3,T2)\n\nRule head: Appeal_for_change_in_institutions,_regime(X0,Xl,Tl)\nSampled rules:\n\tAppeal_for_change_in_institutions,_regime(X0,X1,T1)<-inv_Engage_in_symbolic_act(X0,X1,T0)\n\tAppeal_for_change_in_institutions,_regime(X0,X2,T2)<-inv_Criticize_or_denounce(X0,X1,T0)&Make_pessimistic_comment(X1,X2,T1)\nGenerated Temporal logic rules:\n\tAppeal_for_change_in_institutions,_regime(X0,X1,T1)<-Make_an_appeal_or_request(X0,X1,T0)\n\tAppeal_for_change_in_institutions,_regime(X0,X2,T2)<-inv_Rally_support_on_behalf_of(X0,X1,T0)&Praise_or_endorse(X1,X2,T1)\n\tAppeal_for_change_in_institutions,_regime(X0,X3,T3)<-Appeal_for_change_in_institutions,_regime(X0,X1,T0)&Host_a_visit(X1,X2,T1)&inv_Criticize_or_denounce(X2,X3,T2)\n\tAppeal_for_change_in_institutions,_regime(X0,X2,T2)<-inv_Engage_in_symbolic_act(X0,X1,T0)&inv_Consult(X1,X2,T1)\n\nRule head: Appeal_for_economic_aid(X0,Xl,Tl)\nSampled rules:\n\tAppeal_for_economic_aid(X0,X1,T1)<-inv_Reduce_or_stop_military_assistance(X0,X1,T0)\n\tAppeal_for_economic_aid(X0,X2,T2)<-inv_Make_an_appeal_or_request(X0,X1,T0)&Make_statement(X1,X2,T1)\n\tAppeal_for_economic_aid(X0,X3,T3)<-inv_Demand(X0,X1,T0)&inv_Accede_to_demands_for_change_in_leadership(X1,X2,T1)&Accuse(X2,X3,T2)\nGenerated Temporal logic rules:\n\tAppeal_for_economic_aid(X0,X2,T2)<-Make_an_appeal_or_request(X0,X1,T0)&Appeal_for_military_aid(X1,X2,T1)\n\tAppeal_for_economic_aid(X0,X2,T2)<-inv_Express_intent_to_cooperate(X0,X1,T0)&Make_statement(X1,X2,T1)\n\tAppeal_for_economic_aid(X0,X1,T1)<-Make_an_appeal_or_request(X0,X1,T0)\n\n",

    Temporal Logical Rules: Temporal Logical Rules \"{rule_head}(X0,Xl,Tl)<-R1(X0,X1,T0)&...&Rl(X(l-1),Xl,T(l-1))\" are rules used in temporal knowledge graph reasoning to predict relations between entities over time. They describe how the relation \"{rule_head}\" between entities \"X0\" and \"Xl\" evolves from past time steps \"Ti (i={{0,...,(l-1)}})\" (rule body) to the next \"Tl\" (rule head), strictly following the constraint \"T0 <= \u00b7\u00b7\u00b7 <= T(l-1) < Tl\".

    For the relations in rule body, you are going to choose from the following candidates: {candidate_relations} and the relations of above extracted rules from Current data. Each candidate needs to be selected in all the relations to induce or combine to induce the rule head to make sense in terms of actual semantics.

    Return in JSON format:
    {{
        "{rule_head}": [
            // list of new rules generated from the combination of candidate relations, the structure is like: "{rule_head}(X0,Xl,Tl)<-R1(X0,X1,T0)&...&Rl(X(l-1),Xl,T(l-1))"
        ]
    }}
'''

    return user_msg_content, system_msg_content

def add_generated_rules_to_rule_dict(generated_rules, rule_dict):
    """Add generated new rules to the existing rule_dict."""
    for rule_head, new_rules in generated_rules.items():
        if rule_head in rule_dict:
            rule_dict[rule_head].extend(new_rules)
        else:
            rule_dict[rule_head] = new_rules
    return rule_dict

def generate_new_rules(top_k_relations, k, rule_dict, llm_instance):
    """Generate new rules using LLM based on top k relations and existing rules."""
    generated_rules = {}

    for rule_head, candidate_relations_dict in top_k_relations.items():
        extracted_rules = rule_dict.get(rule_head, [])

        candidate_relations = [list(relation.keys())[0] for relation in candidate_relations_dict]
        
        # Create prompt for LLM
        user_msg_content, system_msg_content = create_llm_prompt(rule_head, extracted_rules, candidate_relations, k)
        
        # Create messages for LLM input
        user_message = HumanMessage(content=user_msg_content)
        system_message = SystemMessage(content=system_msg_content)
        
        # Get response from LLM
        answer_llm = llm_instance.run_task([system_message, user_message])

        if isinstance(answer_llm, dict):
            generated_rules.update(answer_llm)  # Merge generated rules into our dictionary

    return generated_rules


# Set k value for top k relations
k = 10  

# Example usage
top_k_relations_file_path = 'result/icews14/stage_2/top_k_relations.json'
rule_dict_file_path = 'result/icews14/stage_2/rule_dict_output.json'

# Load top k relations and current rule heads with rules
top_k_relations = load_json(top_k_relations_file_path)
rule_dict = load_json(rule_dict_file_path)

# Initialize LLM instance
llm_instance = LLM_Model()

# Generate new rules based on top k relations and existing rules
new_generated_rules = generate_new_rules(top_k_relations, k, rule_dict, llm_instance)

# Output directory
output_dir = 'result/icews14/stage_2'

# Save the generated rules in the specified format
output_file_path = os.path.join(output_dir, 'generated_rules_added_output.json')
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(new_generated_rules, f, ensure_ascii=False, indent=4)
