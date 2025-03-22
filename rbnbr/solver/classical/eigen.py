


import numpy as np
from rbnbr.problems.problem_base import CombProblemBase
from rbnbr.solver.solution import Solution
from rbnbr.solver.solver_base import SolverBase


class EigenSolver(SolverBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def solve(self, problem: CombProblemBase, *args, **kwargs):
        adj_mat = problem.adjacency_matrix
        evals, eig_vecs = np.linalg.eigh(adj_mat)
        
        
        # sign round (0/1) the eigenvector with the largest eigenvalue
        best_cost = float('-inf')
        best_solution = None
        
        # Try all eigenvectors
        for i in range(len(evals)):
            candidate = (1 + np.sign(eig_vecs[:, i])) // 2
            cost = problem.evaluate_solution(candidate)
            if cost > best_cost:
                best_cost = cost
                best_solution = candidate
                
        solution = best_solution
        # evaluate the solution
        cost = problem.evaluate_solution(solution)
        
        return Solution().add_step(
            solution=solution,
            cost=cost,
            approx_ratio=problem.approx_ratio(solution),
            elapsed_time=0.0
        ) 
    