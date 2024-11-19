import numpy as np

class TemporalWalker(object):
    def __init__(self, learn_data, inv_relation_id, transition_choice):
        """
        Initialize temporal random walk object.

        Parameters:
            learn_data (np.ndarray): data on which the rules should be learned
            inv_relation_id (dict): mapping of relation to inverse relation
            transition_distr (str): transition distribution
                                    "unif" - uniform distribution
                                    "exp"  - exponential distribution

        Returns:
            None
        """
        self.learn_data = learn_data
        self.inv_relation_id = inv_relation_id
        self.transition_choice = transition_choice
        self.neighbors = store_neighbor(learn_data)
        self.edges = store_edges(learn_data)

    def sample_start_edge(self, rel_idx):
        """
        Define start edge distribution.

        Parameters:
            rel_idx (int): relation index

        Returns:
            start_edge (np.ndarray): start edge
        """
        rel_edges = self.edges[rel_idx]
        return rel_edges[np.random.choice(len(rel_edges))]
    
    def sample_next_edge(self, filtered_edges, cur_ts):
        """
        Define next edge distribution.

        Parameters:
            filtered_edges (np.ndarray): filtered (according to time) edges
            cur_ts (int): current timestamp

        Returns:
            next_edge (np.ndarray): next edge
        """
        if self.transition_choice == "unif":
            next_edge = filtered_edges[np.random.choice(len(filtered_edges))]
        elif self.transition_choice == "exp":
            tss = filtered_edges[:, 3]
            prob = np.exp(tss - cur_ts)
            try:
                prob = prob  / np.sum(prob)
                next_edge = filtered_edges[np.random.choice(range(len(filtered_edges)), p=prob)]
            except ValueError:
                next_edge = filtered_edges[np.random.choice(len(filtered_edges))]
        return next_edge
    
    def transition_step(self, cur_node, cur_ts, prev_edge, start_node, step, L, target_cur_ts=None):
        """
        Sample a neighboring edge given the current node and timestamp.
        In the second step (step == 1), the next timestamp should be smaller than the current timestamp.
        In the other steps, the next timestamp should be smaller than or equal to the current timestamp.
        In the last step (step == L-1), the edge should connect to the source of the walk (cyclic walk).
        It is not allowed to go back using the inverse edge.

        Parameters:
            cur_node (int): current node
            cur_ts (int): current timestamp
            prev_edge (np.ndarray): previous edge
            start_node (int): start node
            step (int): number of current step
            L (int): length of random walk
            target_cur_ts (int, optional): target current timestamp for relaxed time. Defaults to cur_ts.

        Returns:
            next_edge (np.ndarray): next edge
        """
        next_edges = self.neighbors[cur_node]
        if target_cur_ts is None:
            target_cur_ts = cur_ts
        if step == 1: # the next timestamp should be smaller than the current timestamp
            filtered_edges = next_edges[next_edges[:, 3] < target_cur_ts]
        else: # the next timestamp should be smaller than or equal to the current timestamp
            filtered_edges = next_edges[next_edges[:, 3] <= target_cur_ts]
            # delete the inverse edge
            inv_edge = [
                cur_node,
                self.inv_relation_id[prev_edge[1]],
                prev_edge[0],
                cur_ts
            ]
            row_idx = np.where(np.all(filtered_edges == inv_edge, axis=1))
            filtered_edges = np.delete(filtered_edges, row_idx, axis=0)
        if step == L - 1:
            filtered_edges = filtered_edges[filtered_edges[:, 2] == start_node]
        if len(filtered_edges):
            next_edge = self.sample_next_edge(filtered_edges, target_cur_ts)
        else:
            next_edge = []
        return next_edge

def store_edges(quads):
    """
    Store all edges for each relation.

    Parameters:
        quads (np.ndarray): indices of quadruples

    Returns:
        edges (dict): edges for each relation
    """
    edges = dict()
    relations = list(set(quads[:, 1]))
    for rel in relations:
        edges[rel] = quads[quads[:, 1] == rel]
    return edges

def store_neighbor(quads):
    """
    Store all neighbors (outgoing edges) for each node.

    Parameters:
        quads (np.ndarray): indices of quadruples

    Returns:
        neighbors (dict): neighbors for each node
    """
    neighbors = dict()
    heads = list(set(quads[:, 0]))
    for entity in heads:
        neighbors[entity] = quads[quads[:, 0] == entity]
    return neighbors
