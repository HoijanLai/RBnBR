from qiskit_optimization.applications import Tsp
import networkx as nx

def TSP(n):
    tsp = Tsp.create_random_instance(n, seed=123)
    adj_matrix = nx.to_numpy_array(tsp.graph)
    
    return tsp, adj_matrix