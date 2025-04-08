from dataclasses import InitVar, dataclass, KW_ONLY, field
import logging
import random
from rbnbr.solver.classical.gw import GWMaxcutSolver
from rbnbr.solver.BnB.bnb import BnB
from rbnbr.solver.BnB.branching_rules import branch_hardest, branch_confidence, branch_easiest

from rbnbr.problems.max_cut import MaxCutProblem
import copy

import networkx as nx
import numpy as np

import cvxpy as cp

from queue import PriorityQueue

@dataclass
class GW_BnB_MC(BnB, GWMaxcutSolver):
    """
    Vanilla BnB with GW
    bound with eigenvalue
    branch with correlation or low-rank approximation
    """
    _: KW_ONLY
    branching_strategy: str = field(default="r1")
    search_style: str = field(default='bfs')
    optimize_correction: bool = field(default=True)
    random_pass: float = field(default=0.0)
    
    normalize_corr: bool = field(default=False)
    n_trials: int = field(default=1)


    
    def solve(self, problem: MaxCutProblem):
        bc = self.main_loop(problem)
        return bc


    def solve_subproblem(self, sub_uf, problem, **kwargs):
        if not sub_uf.is_root and problem.ref_solution_arr is not None:
            """
            This is a subproblem with an exact solution
            """
            z_sub = problem.ref_solution_arr
            best_z = sub_uf.get_original_solution(z_sub)
            X = None
        else:
            """
            This is a subproblem without an exact solution
            """
            L = nx.laplacian_matrix(problem.graph).toarray()
            X = GWMaxcutSolver.solve_sdp(L)
            
            best_cut = 0
            best_z = None
            for _ in range(self.n_trials):
                vecs = GWMaxcutSolver.decompose_X(X)
                z_sub = GWMaxcutSolver.random_hyperplane(vecs)
                z = sub_uf.get_original_solution(z_sub)
                cut = self.cache_root_problem.evaluate_solution(z)
                if cut > best_cut:
                    best_cut = cut
                    best_z = z
                              
        sol_info = {
            'X': X
        }
        return best_z, sol_info
        


    def pick_branching_pair(self, sub_UF, X, **kwargs):
        
        if X is None:
            return []

        if self.branching_strategy == 'r1':
            pairs = branch_hardest(X, **kwargs)
        elif self.branching_strategy == 'r2':
            pairs = branch_confidence(X, normalized=self.normalize_corr, **kwargs)
        elif self.branching_strategy == 'r3':
            pairs = branch_easiest(X, **kwargs)
        else:
            raise ValueError(f"Invalid strategy: {self.branching_strategy}")
        
        index_map = sub_UF.index_map
        pairs = [(index_map[qb_u], index_map[qb_v]) for qb_u, qb_v in pairs]

        return pairs
    
    def get_bound(self, _, problem, laplacian, compensation,**k):
        if random.random() < self.random_pass:
            return np.inf, {}
        
        if problem is None:
            return np.inf, {}
        
        if problem.ref_solution_arr is not None:
            return np.inf, {}
        
        
        non_zero_indices = np.where(~np.all(laplacian == 0, axis=0))[0]
        L = laplacian[non_zero_indices][:, non_zero_indices]
        # L = nx.laplacian_matrix(problem.graph).toarray()
        n = L.shape[0]
        if self.optimize_correction:
            u = cp.Variable(n)
            t = cp.Variable()  # This will represent our bound
            
            M = (n/4) * (L + cp.diag(u)) + cp.diag(u)
        
            constraints = [
                cp.sum(u) == 0,
                M << t * np.eye(n)
            ]
            
            prob = cp.Problem(cp.Minimize(t), constraints)
        
            try:
                prob.solve()
                
                u_opt = u.value
                M_opt = (L + np.diag(u_opt))
                
                # Compute largest eigenvector
                evals = np.linalg.eigvals(M_opt)
                return (n/4) * np.max(evals), {}
                
            except Exception as e:
                self.logger.warning(f"Optimization failed: {e}, using simple eigenvalue bound")
                # Fall back to simple eigenvalue bound if optimization fails
              
        # Remove rows and columns that are all zeros

        eigvals = np.linalg.eigvals(L)
        return (n/4) * np.max(eigvals) + compensation, {}



    
    