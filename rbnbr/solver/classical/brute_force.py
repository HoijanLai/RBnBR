
import numpy as np

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
    
    def sanity_check(self, problem):
        if problem.graph.number_of_nodes() > self.max_N:
            raise ValueError(f"Number of nodes in problem is greater than the maximum number of nodes allowed: {problem.graph.number_of_nodes()} > {self.max_N}")
        
        
    def solve(self, problem):
        
        best_assignment, best_cut_value = self._solve(problem)
        # Store solution in problem
        # Add breadcrumb
        breadcrumbs = Solution()
        breadcrumbs.add_step(
            solution=best_assignment[0],
            cost=best_cut_value[0],
            elapsed_time=0  # Could add timing if needed
        )
        
        return breadcrumbs


    def _solve(self, problem):
        self.sanity_check(problem)
        
 
        
        # Get problem data
        A = problem.adjacency_matrix
        n = A.shape[0]
        # Enumerate all possible 0/1 assignments (2^n assignments)
        assignments = ((np.arange(1 << n)[:, None] & (1 << np.arange(n))) > 0).astype(int)  # shape (2^n, n)

        # Compute all cut values simultaneously
        # For each assignment, calculate the cut-value
        diffs = np.abs(assignments[:, :, None] - assignments[:, None, :])  # shape (2^n, n, n)
        cut_values = 0.5 * np.sum(diffs * A, axis=(1, 2))  # shape (2^n,)

        # Find the best solution
        best_idx = np.argmax(cut_values)
        all_best_idx = np.where(cut_values == cut_values[best_idx])[0]
        best_cut_value = cut_values[all_best_idx]
        best_assignment = assignments[all_best_idx]
        
        return best_assignment, best_cut_value
             