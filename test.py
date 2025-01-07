import re

def parse_rules_for_path(lines, relation_regex):

    converted_rules = {}
    for line in lines:
        rule = line.strip()
        if not rule:
            continue
        temp_rule = re.sub(r'\s*<-\s*', '&', rule)
        regrex_list = temp_rule.split('&')

        head = ""
        body_list = []
        for idx, regrex_item in enumerate(regrex_list):
            match = re.search(relation_regex, regrex_item)
            if match:
                rel_name = match.group(1).strip()
                if idx == 0:
                    head = rel_name
                    paths = converted_rules.setdefault(head, [])
                else:
                    body_list.append(rel_name)

        path = '|'.join(body_list)
        paths.append(path)

    return converted_rules

lines = ["Arrest,_detain,_or_charge_with_legal_action(X0,X1,T1)<-Return,_release_person(s)(X0,X1,T0)",
         "Make_statement(X0,X1,T3)<-Appeal_for_diplomatic_cooperation_(such_as_policy_support)(X0,X1,T0)&Make_a_visit(X1,X2,T1)&inv_Reduce_relations(X2,X1,T2)",
          "Make_statement(X0,X2,T2)<-Make_empathetic_comment(X0,X1,T0)&Obstruct_passage,_block(X1,X2,T1)",]
parse_rules_for_path(lines, "([\\w\\s'\\-\\.,\\(\\)]+)\\((\\w+),\\s*(\\w+),\\s*(\\w+)\\)(&|$)")