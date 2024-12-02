from openai_llm.llm_init import LLM_Model
from langchain_core.messages import HumanMessage, SystemMessage

llm_instance = LLM_Model()

rules = ["Praise_or_endorse(X0,X1,T3)<-Reduce_or_stop_military_assistance(X0,X1,T0)&inv_Appeal_to_others_to_settle_dispute(X1,X2,T1)&Appeal_to_others_to_settle_dispute(X2,X1,T2)",
        "Praise_or_endorse(X0,X1,T1)<-Veto(X0,X1,T0)",
        "Praise_or_endorse(X0,X1,T3)<-Provide_economic_aid(X0,X1,T0)&inv_Make_statement(X1,X2,T1)&Arrest,_detain,_or_charge_with_legal_action(X2,X1,T2)"]

user_query = "Please help me to verbalize these temporal rules in natural language."
user_msg_content = f'''
Here is the user query: {user_query}

Here is the rules that need to be verbalized:
{rules}
'''
user_message = HumanMessage(content=user_msg_content)

system_msg_content = '''
You are an expert in temporal knowledge graph. You will be given some temporal rules that are learned from the data.
Your task is to verbalize the rules in natural language.
your answer should be in the json format:
{{
    "original_rules": // a list of original rules that are learned from the data.
    "verbalized_rules": // a list of corresponding verbalized rules in natural language.
}}
'''

system_message = SystemMessage(content=system_msg_content)

answer_llm = llm_instance.run_task([system_message, user_message])
print(answer_llm)