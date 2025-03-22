import numpy as np
from typing import List, Tuple
import cvxpy as cp
from rbnbr.solver.solver_base import ApproximationSolver
from rbnbr.problems.problem_base import CombProblemBase
from rbnbr.solver.solution import Solution


class GWMaxcutSolver(ApproximationSolver):
    
    def __init__(self):
        super().__init__()
        
    def solve(self, problem: CombProblemBase) -> List[int]:
        breadcrumbs = Solution()
        adj_matrix = problem.adjacency_matrix
        
        # Extracted method call
        X_val = GWMaxcutSolver.solve_sdp(adj_matrix)
        
        # Step 2: Cholesky decomposition to get vectors
        vectors = GWMaxcutSolver.decompose_X(X_val)
        
        # Step 3: Random hyperplane rounding
        cut = GWMaxcutSolver.random_hyperplane(vectors)

        # Step 4: Evaluate the solution
        breadcrumbs.add_step(
            solution=cut,
            cost=problem.evaluate_solution(cut),  # Fixed: use cut instead of breadcrumbs.solution
            elapsed_time=0,
            approx_ratio=problem.approx_ratio(cut)  # Fixed: use cut instead of breadcrumbs.solution
        )
        
        return breadcrumbs

    @staticmethod
    def solve_sdp(adj_matrix: np.ndarray) -> np.ndarray:
        n = adj_matrix.shape[0]
        X = cp.Variable((n, n), symmetric=True)
        constraints = [X >> 0]  # PSD constraint
        constraints += [cp.diag(X) == 1]  # Unit vectors constraint
        
        is_weighted = not np.all(np.logical_or(adj_matrix == 0, adj_matrix == 1))
        
        if is_weighted:
            objective = cp.Maximize(0.25 * cp.sum(cp.multiply(adj_matrix, (1 - X))))
        else:
            objective = cp.Maximize(0.25 * cp.sum(cp.multiply((adj_matrix > 0).astype(float), (1 - X))))
        
        constraints[0] = X + 1e-6 * np.eye(n) >> 0

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)
        
        if prob.status != cp.OPTIMAL:
            print(f"Solver status: {prob.status}")
            raise RuntimeError(f"SDP optimization failed with status {prob.status}")
        
        X_val = X.value
        X_val = (X_val + X_val.T) / 2  # Ensure perfect symmetry
        return X_val
    
    @staticmethod
    def decompose_X(X_val: np.ndarray) -> np.ndarray:
        reg = 1e-6
        while True:
            try:
                L = np.linalg.cholesky(X_val + reg * np.eye(X_val.shape[0]))
                break
            except np.linalg.LinAlgError:
                reg *= 10
                if reg > 1e-3:  # Set a maximum regularization threshold
                    raise RuntimeError("Failed to find valid Cholesky decomposition")
                
        vectors = L.T  # Each row is a vector
        return vectors
    
    
    @staticmethod
    def random_hyperplane(vectors: np.ndarray) -> np.ndarray:
        # Step 3: Random hyperplane rounding
        r = np.random.normal(0, 1, size=vectors.shape[1])
        r = r / np.linalg.norm(r)  # Normalize the random vector
        
        # Step 4: Assign vertices based on sign of projection
        cut = np.sign(vectors @ r)
        cut = ((cut + 1) // 2).astype(int) # Convert from {-1,1} to {0,1}
        return cut
    
    