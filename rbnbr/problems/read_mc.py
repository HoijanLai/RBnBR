from pathlib import Path
import networkx as nx
from rbnbr.problems.max_cut import MaxCutProblem
import numpy as np

import pandas as pd

from qiskit_ibm_runtime import QiskitRuntimeService
import random
import re




def read_mc(filename, solution_value, keep_prob=1.0, name=None):
    
    if keep_prob < 1.0: 
        return MaxCutProblem(read_mc_graph(filename, keep_prob), solve=False, name=name)
    else: 
        return MaxCutProblem(read_mc_graph(filename, keep_prob), solve=False, solution_value=solution_value, name=name)

def read_mc_graph(filename, keep_prob=0.1):
    G = nx.Graph()
    
    with open(filename, 'r') as f:
        # Read first line containing number of nodes and edges
        n_nodes, n_edges = map(int, f.readline().split())
        
        # Read all edges
        for _ in range(n_edges):
            line = f.readline()
            u, v, w = map(int, line.split())
            if random.random() < keep_prob:
                G.add_nodes_from([u-1, v-1])
                G.add_edge(u-1, v-1, weight=w)
    
    # Keep only the largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    
    # rename nodes to be 0, 1, 2, ...
    mapping = {old: new for new, old in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    return G
 

def coupling_map(n):
    graph_100 = nx.Graph()
    graph_100.add_nodes_from(np.arange(0, n, 1))
    elist = []

    service = QiskitRuntimeService()
    real_backend = service.least_busy(min_num_qubits=n)

    for edge in real_backend.coupling_map:
        if edge[0] < n and edge[1] < n:
            elist.append((edge[0], edge[1]))
    graph_100.add_edges_from(elist)
    return MaxCutProblem(graph_100, solve=False)
    
    
class MCProblemRetriever:
    def __init__(self, target_location='./biqmac/mac/rudy'):
        self.df = pd.read_csv('./problems/bq/problem_info.csv')
        self.df['Instance'] = self.df['Instance'].apply(remove_suffix)
        self.target_location = Path(target_location)
        
    def get_solution_value(self, problem_name):
        return self.df[self.df['Instance'] == problem_name]['OSV'].values[0]
    
    
    def available_problems(self, max_V=None, max_E=False, prefix=None):
        # get all problem that satisfies the condition
        if max_V:
            df = self.df[self.df['|V|'] <= max_V]
        if max_E:
            df = df[df['|E|'] <= max_E]
        if prefix:
            df = df[df['Instance'].str.startswith(prefix)]
        return df['Instance'].unique().tolist()
    
    def get_problem(self, problem_name=None):
        solution_value = self.get_solution_value(problem_name)
        return read_mc(self.target_location / problem_name, solution_value, name=problem_name)
        
        
    def get_random_problem(self, max_V=None, max_E=None, prefix=None):
        ok_problem_names = self.available_problems(max_V, max_E, prefix)
        problem_name = random.choice(ok_problem_names)
        print(problem_name)
        return self.get_problem(problem_name)


def remove_suffix(s):
    return re.sub(r' \(.mc, .bq\)', '', s)

def add_suffix(s):
    return s + ' (.mc, .bq)'


    
    