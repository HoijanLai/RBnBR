
import numpy as np
from rbnbr.solver.classical.brute_force import BruteForceMaxCutSolver
from rbnbr.utils.visualization import visualize_cut
from rbnbr.problems.problem_base import CombProblemBase
import networkx as nx


from rbnbr.utils.const import ArrayLike


import warnings

class MaxCutProblem(CombProblemBase):
    
    
    @property
    def N(self):
        return self.graph.number_of_nodes()
    
    @property
    def E(self):
        return self.graph.number_of_edges()
    
    @property
    def name(self):
        return self._metadata['name']
    
    
    def __init__(self, 
                 graph, 
                 solve=True, 
                 solution_value=None,
                 name=None,
                 exact_solver=BruteForceMaxCutSolver
        ):
        super().__init__(graph=graph, problem_type='maxcut')
        solver = exact_solver()
        
        self.ref_cost = None
        self.ref_solution_arr = None
        
        if solve:
            breadcrumbs = solver.solve(self)
            self.ref_solution_arr = breadcrumbs.solution
            self.ref_cost = breadcrumbs.cost
            self.add_solution("exact", breadcrumbs)
        if solution_value is not None:
            self.ref_cost = solution_value
            
        if name is not None:
            self._metadata['name'] = name
        

        
    @classmethod
    def generate_random_maxcut_problem(cls, n: int, p: float = None, r: float = None, weighted: bool = False, solve: bool = False, method: str = 'erdos_renyi'):
        # generate a random graph with n nodes and p probability of edge
        if method == 'erdos_renyi':
            graph = nx.erdos_renyi_graph(n, p)
        elif method == 'geometric':
            graph = nx.random_geometric_graph(n, r)
            # Ensure the graph is connected by adding extra edges if necessary
            if not nx.is_connected(graph):
                components = list(nx.connected_components(graph))
                for i in range(len(components) - 1):
                    u = list(components[i])[0]
                    v = list(components[i + 1])[0]
                    graph.add_edge(u, v)
            # Add random weights to edges
        
        else:
            raise ValueError(f"Invalid method: {method}")
        
        if weighted:
            for u, v in graph.edges():
                graph[u][v]['weight'] = round(np.random.uniform(-1.0, 1.0), 2)
        else:
            for u, v in graph.edges():
                graph[u][v]['weight'] = 1
        return cls(graph, solve=solve)
   
    def evaluate_solution(self, solution: np.ndarray):
        # Check if solution is -1/1 encoded or 0/1 encoded
        solution = self._convert_solution(solution)
            
        # Evaluate cut value
        cut_value = 0
        n = len(solution)
        for i in range(n):
            for j in range(i+1, n):
                if self.graph.has_edge(i, j):
                    weight = self.graph[i][j].get('weight', 1)
                    cut_value += 0.5 * weight * (1 - solution[i] * solution[j])
                    
        return cut_value
        
        
        
    def approx_ratio(self, solution: ArrayLike):
        if self.ref_cost is None:
            warnings.warn("Reference solution not available - cannot compute approximation ratio, returning None")
            return None
        return self.evaluate_solution(solution) / self.ref_cost
    
    def _convert_solution(self, solution: ArrayLike):
        solution = np.array(solution)
        unique_vals = np.unique(solution)
        if set(unique_vals).issubset({0, 1}):
            # Convert 0/1 encoding to -1/1 encoding
            return 2 * solution - 1
        elif set(unique_vals).issubset({-1, 1}):
            return solution
        else:
            raise ValueError("Solution must be either -1/1 or 0/1 encoded")
    
    
    def display(self, solution: ArrayLike = None, reset_pos: bool = False, vis_opts={'node_size': 100, 'edge_alpha': 0.6}, **kwargs):
        if reset_pos:
            self.vis_pos = nx.spring_layout(self.graph);
        # Create visualization
        
        kwargs.update({'edge_width': kwargs.get('edge_width', 0.1)})
        pos = visualize_cut(
            graph=self.graph,
            solution=solution,
            pos=self.vis_pos,
            opt=vis_opts,
            title=f"{self.name} on {self.graph.number_of_nodes()} nodes",
            **kwargs
        );
        return pos
    
    
