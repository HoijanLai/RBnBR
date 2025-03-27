
import numpy as np
import networkx as nx

def compute_Ci(h_i: float, J_ik: np.ndarray, beta: float, gamma: float) -> float:
    """Compute the expectation value <C_i>_kappa."""
    sin_2beta = np.sin(2 * beta)
    sin_2gamma_hi = np.sin(2 * gamma * h_i)
    prod_cos = np.prod(np.cos(2 * gamma * J_ik))
    
    return h_i * sin_2beta * sin_2gamma_hi * prod_cos

def compute_Cij(J_ij: float, h_i: float, h_j: float, J_ik: np.ndarray, J_jk: np.ndarray, beta: float, gamma: float) -> float:
    """Compute the expectation value <C_ij>_kappa."""
    sin_4beta = np.sin(4 * beta)
    sin_2beta_sq = (np.sin(2 * beta)) ** 2
    sin_2gamma_Jij = np.sin(2 * gamma * J_ij)
    cos_2gamma_hi = np.cos(2 * gamma * h_i)
    cos_2gamma_hj = np.cos(2 * gamma * h_j)
    cos_2gamma_hi_hj = np.cos(2 * gamma * (h_i + h_j))
    cos_2gamma_hi_minus_hj = np.cos(2 * gamma * (h_i - h_j))
    prod_cos_ik = np.prod(np.cos(2 * gamma * J_ik))
    prod_cos_jk = np.prod(np.cos(2 * gamma * J_jk))
    prod_cos_ik_jk = np.prod(np.cos(2 * gamma * (J_ik + J_jk)))
    prod_cos_ik_minus_jk = np.prod(np.cos(2 * gamma * (J_ik - J_jk)))
    
    term1 = (J_ij * sin_4beta / 2) * sin_2gamma_Jij * (cos_2gamma_hi * prod_cos_jk + cos_2gamma_hj * prod_cos_ik)
    term2 = -(J_ij / 2) * sin_2beta_sq * cos_2gamma_hi_hj * prod_cos_ik_jk
    term3 = (J_ij / 2) * sin_2beta_sq * cos_2gamma_hi_minus_hj * prod_cos_ik_minus_jk
    
    return term1 + term2 - term3






def p1_Z_maxcut(graph, beta, gamma): 
    """
    not compatible with the graph with negative edges
    """
    n_nodes = len(graph.nodes)
    Z = np.zeros((n_nodes, n_nodes))
    # Convert graph to adjacency matrix using NetworkX's built-in function

    J = nx.to_numpy_array(graph)
    uw_J = np.where(J > 0, 1, 0)  # Create unweighted adjacency matrix 
    
    # For MaxCut, all h_i are 0
    h = np.zeros(n_nodes)
    
    order_idx = np.arange(n_nodes).reshape(1, n_nodes)
    
    # Compute two-qubit expectation values
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if J[i, j] != 0:  # Only compute for connected nodes
                # Get all connections to nodes i and j, excluding the i-j connection
                uw_J_ii_jj = uw_J[[i, j],:] * (order_idx != i) * (order_idx != j)
                uw_J_ik = uw_J_ii_jj[0, :]
                uw_J_jk = uw_J_ii_jj[1, :]
                
                z_ij =compute_Cij(J[i, j], 0.0, 0.0, uw_J_ik, uw_J_jk, beta, gamma)
                Z[i, j] = z_ij
                Z[j, i] = z_ij
    
    # Compute the total expectation value
    return Z


def p1_expval_maxcut(graph, beta, gamma):
    w = nx.to_numpy_array(graph)
    Z = p1_Z_maxcut(graph, beta, gamma)
    return np.sum(Z * w)
    
    

    