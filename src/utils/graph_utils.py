import numpy as np


def get_blazepose_33_topology():
    """
    Returns the number of nodes and anatomical edge connections for the
    BlazePose 33-landmark skeleton.

    Note on landmark coverage:
        BlazePose provides 33 landmarks total (indices 0–32), which include
        11 face/head landmarks (0–10) and 22 body landmarks (11–32).
        For exercise recognition focused on compound resistance movements
        (squat, push-up, bench press, pull-up), only the body landmarks are
        structurally relevant to joint-angle computation. Face landmarks are
        excluded from the graph topology because they carry no biomechanical
        signal for limb kinematics and would introduce noise. This is
        consistent with prior ST-GCN implementations for action recognition
        (Yan et al., 2018) that restrict graph edges to load-bearing joints.
        The node count is kept at 33 (not 22) so that landmark indices map
        directly to the BlazePose output without remapping; face nodes simply
        receive no edges and are treated as isolated vertices.

    BlazePose landmark index reference (body only):
        11/12 = left/right shoulder
        13/14 = left/right elbow
        15/16 = left/right wrist
        17–22 = hand landmarks (pinky, index, thumb tips — left then right)
        23/24 = left/right hip
        25/26 = left/right knee
        27/28 = left/right ankle
        29/30 = left/right heel
        31/32 = left/right foot index
    """
    num_nodes = 33

    connections = [
        # Core torso (bilateral symmetry joints)
        (11, 12),   # left shoulder — right shoulder
        (23, 24),   # left hip — right hip
        (11, 23),   # left shoulder — left hip
        (12, 24),   # right shoulder — right hip

        # Left arm chain
        (11, 13),   # left shoulder — left elbow
        (13, 15),   # left elbow — left wrist
        (15, 17),   # left wrist — left pinky
        (17, 19),   # left pinky — left index
        (15, 21),   # left wrist — left thumb

        # Right arm chain
        (12, 14),   # right shoulder — right elbow
        (14, 16),   # right elbow — right wrist
        (16, 18),   # right wrist — right pinky
        (18, 20),   # right pinky — right index
        (16, 22),   # right wrist — right thumb

        # Head to shoulder (only node 0 retained from face group)
        (0, 11),    # nose — left shoulder
        (0, 12),    # nose — right shoulder

        # Left leg chain
        (23, 25),   # left hip — left knee
        (25, 27),   # left knee — left ankle
        (27, 29),   # left ankle — left heel
        (29, 31),   # left heel — left foot index

        # Right leg chain
        (24, 26),   # right hip — right knee
        (26, 28),   # right knee — right ankle
        (28, 30),   # right ankle — right heel
        (30, 32),   # right heel — right foot index
    ]

    return num_nodes, connections


def build_blazepose_33_adjacency_matrix(self_loops=True):
    """
    Constructs a symmetrically-normalised adjacency matrix for the BlazePose
    33-node skeleton, following the formulation of Kipf & Welling (2017) and
    applied to ST-GCN by Yan et al. (2018):

        Â = D̂^{-1/2} · (A + I) · D̂^{-1/2}

    where A is the binary adjacency matrix, I is the identity (self-loops),
    and D̂ is the diagonal degree matrix of (A + I).

    Symmetric normalisation is used instead of row normalisation (D⁻¹A)
    because it preserves the spectral properties of the graph Laplacian and
    is the formulation cited throughout the methodology.

    Args:
        self_loops (bool): If True, adds self-connections (I) to A before
                           normalisation. Default True.

    Returns:
        norm_adj (np.ndarray): Symmetrically-normalised adjacency matrix,
                               shape (33, 33), dtype float32.
    """
    num_nodes, connections = get_blazepose_33_topology()

    # Build binary symmetric adjacency matrix
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for (u, v) in connections:
        if u < num_nodes and v < num_nodes:
            adj[u, v] = 1.0
            adj[v, u] = 1.0  # undirected graph

    # Add self-loops: Ã = A + I
    if self_loops:
        adj += np.eye(num_nodes, dtype=np.float32)

    # Compute degree matrix D̂ from Ã
    degree = adj.sum(axis=1)  # shape (num_nodes,)

    # D̂^{-1/2}: guard against zero-degree isolated nodes
    d_inv_sqrt = np.power(degree + 1e-9, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)

    # Symmetric normalisation: Â = D̂^{-1/2} · Ã · D̂^{-1/2}
    norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt

    return norm_adj.astype(np.float32)