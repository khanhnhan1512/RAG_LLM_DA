

class TemporalWalker(object):
    def __init__(self, learn_data, inv_relation_id, transition_choice):
        self.learn_data = learn_data
        self.inv_relation_id = inv_relation_id
        self.transition_choice = transition_choice

    def store_edges(self):
        edges = dict()
        relations = 
