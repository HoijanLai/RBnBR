from itertools import combinations
import networkx as nx
from typing import Tuple, List

import numpy as np

from rbnbr.problems.problem_base import CombProblemBase
from rbnbr.solver.solver_base import ExactSolver
from rbnbr.solver.solution import Solution

class BruteForceMaxCutSolver(ExactSolver):
    """
    brute-force solver for max cut
    """
    def __init__(self):
        super().__init__()
        self.max_N = np.inf
        
        
    def set_N_limit(self, N_limit: int):
        pass
    
    def sanity_check(
        self, 
        problem: CombProblemBase):
        if problem.graph.number_of_nodes() > self.max_N:
            raise ValueError(f"Number of nodes in problem is greater than the maximum number of nodes allowed: {problem.graph.number_of_nodes()} > {self.max_N}")
        
        
    def solve(self, problem):
        self.sanity_check(problem)
        
        breadcrumbs = Solution()
        
        # Get problem data
        graph = problem.graph
        n = graph.number_of_nodes()
        vertices = list(graph.nodes())
        max_cut_value = 0
        best_solution = np.zeros(n)  # Initialize with all 0s
        
        # Try all possible combinations of vertices for partition S
        for size in range(1, n):
            for S in combinations(vertices, size):
                # Create 0/1 encoded solution
                solution = np.zeros(n)
                solution[list(S)] = 1
                
                # Calculate cut value for current partition
                cut_value = sum(
                    graph[u][v].get('weight', 1)
                    for u in range(n)
                    for v in range(n)
                    if graph.has_edge(u, v) and solution[u] != solution[v]
                )
                
                if cut_value > max_cut_value:
                    max_cut_value = cut_value
                    best_solution = solution
                    
        # Store solution in problem
        # Add breadcrumb
        max_cut_value /= 2
        breadcrumbs.add_step(
            solution=best_solution,
            cost=max_cut_value,
            elapsed_time=0  # Could add timing if needed
        )
        
        return breadcrumbs