import numpy as np
from typing import List, Tuple
import cvxpy as cp
from rbnbr.solver.solver_base import ApproximationSolver
from rbnbr.problems.problem_base import CombProblemBase
from rbnbr.solver.solution import Solution
import networkx as nx


class GWMaxcutSolver(ApproximationSolver):
    
    def __init__(self):
        super().__init__()
        
    def solve(self, problem: CombProblemBase) -> List[int]:
        breadcrumbs = Solution()
        
        L = np.array(0.25 * nx.laplacian_matrix(problem.graph).todense())
        
        # Extracted method call
        L_u = GWMaxcutSolver.get_L_u(L)
        X_val = GWMaxcutSolver.solve_sdp(L_u)
        
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
    def get_L_u(L) -> np.ndarray:
        N = L.shape[0]
        u = cp.Variable(N)  # Correcting vector
        L_u = L + cp.diag(u)  # D - W + diag(u)
        t = cp.Variable()   # Variable bounding the largest eigenvalue

        # Objective: Minimize the largest eigenvalue (minimize t)
        objective = cp.Minimize(t)
        
        # Constraints:
        # (1) (N/4) * L_u << t * I  (i.e., largest eigenvalue of (N/4)L_u <= t)
        # (2) Sum(u) = 0
        constraints = [
            (N/4) * L_u << t * np.eye(N),
            cp.sum(u) == 0
        ]
        
        # Solve the SDP
        prob = cp.Problem(objective, constraints)   
        prob.solve(solver=cp.CVXOPT)

        return L+np.diag(u.value)

    @staticmethod
    def solve_sdp(L) -> np.ndarray:
        psd_mat = cp.Variable(L.shape, PSD=True)
        obj = cp.Maximize(cp.trace(L @ psd_mat))
        constraints = [cp.diag(psd_mat) == 1]  
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.CVXOPT)

        return psd_mat.value
    
    @staticmethod
    def decompose_X(X_val: np.ndarray) -> np.ndarray:
        reg = 1e-6
        evals, evects = np.linalg.eigh(X_val)
        vectors = evects.T[evals > reg].T
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
    
    