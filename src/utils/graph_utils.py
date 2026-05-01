import numpy as np

def get_blazepose_33_topology():
    num_nodes = 33
    connections = [
        (11, 12), (23, 24), (11, 23), (12, 24), # Torso
        (11, 13), (13, 15), (15, 17), (17, 19), (15, 21), # Left Arm
        (12, 14), (14, 16), (16, 18), (18, 20), (16, 22), # Right Arm
        (0, 11), (0, 12), # Head/Shoulders
        (23, 25), (25, 27), (27, 29), (29, 31), # Left Leg
        (24, 26), (26, 28), (28, 30), (30, 32), # Right Leg
    ]
    return num_nodes, connections

def build_blazepose_33_adjacency_matrix(self_loops=True):
    num_nodes, connections = get_blazepose_33_topology()
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for edge in connections:
        node_u, node_v = edge
        if node_u < num_nodes and node_v < num_nodes:
            adj[node_u, node_v] = 1.0
            adj[node_v, node_u] = 1.0 

    if self_loops:
        adj += np.eye(num_nodes)

    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum + 1e-9, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    norm_adj = adj.dot(r_mat_inv)

    return norm_adj