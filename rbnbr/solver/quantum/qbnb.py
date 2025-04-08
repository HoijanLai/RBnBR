from dataclasses import InitVar, dataclass, KW_ONLY, field
import logging
import random
from rbnbr.solver.quantum.qrr import QRRMaxCutSolver
from rbnbr.solver.BnB.bnb import BnB
from rbnbr.solver.BnB.branching_rules import branch_hardest, branch_confidence, branch_easiest

from rbnbr.problems.max_cut import MaxCutProblem
import copy

import networkx as nx
import numpy as np

import cvxpy as cp

from queue import PriorityQueue



@dataclass
class QRR_BnB_MC(BnB, QRRMaxCutSolver):
    """
    Vanilla BnB with QRR
    bound with eigenvalue
    branch with correlation or low-rank approximation
    """
    _: KW_ONLY
    branching_strategy: str = field(default="r1")
    approx_style: str = field(default='default')
    search_style: str = field(default='bfs')
    optimize_correction: bool = field(default=True)
    qrr_information: tuple = field(default=None)
    optim_params: list = field(default=None)
    random_pass: float = field(default=0.0)
    
    normalize_correlation: bool = field(default=False)
    
    def __post_init__(self, *args, **kwargs): 
        BnB.__post_init__(self, *args, **kwargs)
        QRRMaxCutSolver.__post_init__(self, *args, **kwargs)
        self.qrr_info = self._check_qrr_info(self.qrr_information)
        self.opt_params = self._check_optim_params(self.optim_params)
        self.todo = []

    
    def solve(self, problem: MaxCutProblem):
        bc = self.main_loop(problem)
        return bc


    def _solve_subproblem(self, sub_uf, problem, **kwargs):
        
        """
        RETURN THE SUBPROBLEM SOLUTION
        """
    
        # Check if this is the root problem (no variables fixed yet)
        if self.qrr_info is not None and sub_uf.is_root:
            # If we have pre-computed QRR information for the root problem, use it
            z_sub, X, _ = self.qrr_info
        
        elif not sub_uf.is_root and problem.ref_solution_arr is not None:
            z_sub = problem.ref_solution_arr
            X = None
            
        else:
            # Otherwise, solve the subproblem with QRR
            if sub_uf.is_root:
                # For root problem, use provided optimization parameters
                z_sub, X = QRRMaxCutSolver.qrr(self, problem, self.opt_params, approx_style=self.approx_style)
            else:
                # For subproblems, use default parameters
                z_sub, X = QRRMaxCutSolver.qrr(self, problem, approx_style=self.approx_style, index_map=sub_uf.index_map_inv)
                
            # Convert the reduced solution to the original solution
        
        return z_sub, X


    def solve_subproblem(self, sub_uf, problem, **kwargs):
        self.logger.debug(f"Solving subproblem with size {sub_uf.n_rep}")
        z_sub, X = self._solve_subproblem(sub_uf, problem, **kwargs)
        
        z = sub_uf.get_original_solution(z_sub)
        
        sol_info = {
            'X': X
        }
        return z, sol_info

    def pick_branching_pair(self, sub_UF, X, **kwargs):
        
        if X is None:
            return []

        if self.branching_strategy == 'r1':
            pairs = branch_hardest(X, **kwargs)
        elif self.branching_strategy == 'r2':
            pairs = branch_confidence(X, normalized=self.normalize_correlation, **kwargs)
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
    
    
    def _check_qrr_info(self, qrr_information):
        if qrr_information is not None:
            if self.X_type != qrr_information[2]:
                msg = f"Ignored: The specified QRR information has X_type {qrr_information[2]}, but the current QRR solver has X_type {self.X_type}"
                self.logger.warning(msg)
                qrr_information = None
            else:
                qrr_information = copy.deepcopy(qrr_information)
        return qrr_information



@dataclass
class QRR_BnB_MC_V2(QRR_BnB_MC):
    """
    Variant:
    Bound with solver 
    First solve the subproblem with QRR
    Then bound with the solution + gap
    Can set constant gap or gap scale
    when gap scale is set, gap scale will be divided by the number of variables fixed so far
    """
    _: KW_ONLY
    random_pass : float = 0.1
    constant_gap : int = None
    gap_scale : float = 0.0
    

    def root_problem_info(self, original_problem):
        basic_info = super().root_problem_info(original_problem)
        avg_deg = original_problem.graph.number_of_edges() / original_problem.N
        self.cache_avg_deg = avg_deg
        return basic_info
    
    def get_bound(self, sub_uf, problem, **other_info):
        # Check if this is the root problem (no variables fixed yet)
        if self.qrr_info is not None and sub_uf.is_root:
            # If we have pre-computed QRR information for the root problem, use it
            z, X, _ = self.qrr_info
        
        elif problem.ref_solution_arr is not None:
            z = problem.ref_solution_arr
            X = None
            
        else:
            # Otherwise, solve the subproblem with QRR
            if sub_uf.is_root:
                # For root problem, use provided optimization parameters
                z_sub, X, info = QRRMaxCutSolver.qrr(self, problem, self.opt_params, approx_style=self.approx_style, return_info=True)
            else:
                # For subproblems, use default parameters
                z_sub, X, info = QRRMaxCutSolver.qrr(self, problem, approx_style=self.approx_style, return_info=True)
                
            # Convert the reduced solution to the original solution
            # z_sub_r = info['max_relaxed_solution']
            # cut_sub_r = problem.evaluate_solution(z_sub_r, relaxed=True)
            z = sub_uf.get_original_solution(z_sub)
            
        cut = self.cache_root_problem.evaluate_solution(z)
        bound_info = {
            'X': X, 
            'z': z,
            'cut': cut
        }
        
        
        ### BOUND WITH SOLUTION 
        if self.constant_gap is not None:
            gap = self.constant_gap
        else:
            gap = self.gap_scale / (sub_uf.n_var - sub_uf.n_rep) * self.cache_avg_deg
        free2go = (random.random() < self.random_pass) 
        
        bound = np.inf if free2go else cut + gap
        
        return bound, bound_info
    
    def solve_subproblem(self, sub_uf, z=None, **kwargs):
        self.logger.debug(f"Solving subproblem with size {sub_uf.n_rep}")
        if z is None:
            assert sub_uf.is_root, "SOMETHING IS WRONG with your logic"
            problem = kwargs['problem']
            z, X = QRRMaxCutSolver.qrr(self, problem, self.opt_params, approx_style=self.approx_style)
            
            cut = self.cache_root_problem.evaluate_solution(z)
            bound_info = {
                'X': X, 
                'z': z,
                'cut': cut
            }
            return z, bound_info 
        
        return z, {}
    
    
    
@dataclass
class QRR_BnB_MC_Lazy(QRR_BnB_MC):
    _: KW_ONLY
    solve_every : float = 0.1 # every percentage of depth

    def __post_init__(self):
        super().__post_init__()
        self._cached_best_info = {
            'z': None
        }
        self._cached_best_z = None
        self._cached_X = ...
    
    
    def solve_subproblem(self, sub_uf, index_map, **kwargs):
        solve_flag = (self.n_depth % (sub_uf.n_var * self.solve_every) == 0)
        solve_flag = solve_flag or (self._cached_best_z is None)
            
        if solve_flag:
            z_sub, X = self._solve_subproblem(sub_uf, **kwargs)
            z = sub_uf.get_original_solution(z_sub)
            cut = self.cache_root_problem.evaluate_solution(z)
            if self._cached_best_info['cut'] is None or cut > self._cached_best_info['cut']:
                self._cached_best_info = {
                    'z': z,
                    'X': X,
                    'cut': cut,
                    'index_map': index_map
                }
            return z_sub, {}
        else:
            return self._cached_best_z, {}
            
        
    def pick_branching_pair(self, X, sheduled_pairs, **kwargs):
        """ 
        when X is provided, calculate the next pairs and schedule them,
        when scheduled pairs are exhausted, pick the next pair from the scheduled pairs
        
        QUESTIONS: HOW TO SCHEDULE THE GOOD PAIRS? 
        """
        pass
    
    
    
    
    
@dataclass
class QRR_BnB_MC_Local_Improvement(QRR_BnB_MC):
    
    """
    local improvement method 
    """
    _: KW_ONLY
    n_carve: int = 2
    p = 0.2
    
    
    def solve_subproblem(self, sub_uf, problem, **kwargs):
        z_sub, X = self._solve_subproblem(sub_uf, problem, **kwargs)
        z = sub_uf.get_original_solution(z_sub)
        cut = self.cache_root_problem.evaluate_solution(z)
        
        for _ in range(self.n_carve):
            for idx, bit in enumerate(z_sub):
                z_sub_temp = z_sub[:]
                z_sub_temp[idx] = 1 - z_sub_temp[idx]
                
                new_z = sub_uf.get_original_solution(z_sub_temp)
                
                new_cut = self.cache_root_problem.evaluate_solution(new_z)
                    
                if new_cut > cut:
                    z = new_z
                    cut = new_cut
                    X = self.p * np.outer(z_sub_temp, z_sub_temp) + (1-self.p) * X
        
        
        sol_info = {
            'X': X
        }
        return z, sol_info
    
    
    
    



    
    

@dataclass
class QRR_BnB_MC_A(QRR_BnB_MC_V2):
    """
    Variant:
    Bound with solver 
    First solve the subproblem with QRR
    Then bound with the solution + gap
    Can set constant gap or gap scale
    when gap scale is set, gap scale will be divided by the number of variables fixed so far
    """
    _: KW_ONLY
    random_pass : float = 0.1
    constant_gap : int = None
    

    def root_problem_info(self, original_problem):
        basic_info = super().root_problem_info(original_problem)
        avg_deg = original_problem.graph.number_of_edges() / original_problem.N
        self.cache_avg_deg = avg_deg
        return basic_info
    
    def get_bound(self, sub_uf, problem, compensation, **other_info):
        # Check if this is the root problem (no variables fixed yet)
        if self.qrr_info is not None and sub_uf.is_root:
            # If we have pre-computed QRR information for the root problem, use it
            z, X, _ = self.qrr_info
            
        
        elif problem.ref_solution_arr is not None:
            z = problem.ref_solution_arr
            X = None
            bound = np.inf
            
        else:
            # Otherwise, solve the subproblem with QRR
            if sub_uf.is_root:
                # For root problem, use provided optimization parameters
                z_sub, X = QRRMaxCutSolver.qrr(self, problem, self.opt_params, approx_style=self.approx_style)
            else:
                # For subproblems, use default parameters
                z_sub, X = QRRMaxCutSolver.qrr(self, problem, approx_style=self.approx_style)
                
            # Convert the reduced solution to the original solution
            z = sub_uf.get_original_solution(z_sub)
            
        cut = self.cache_root_problem.evaluate_solution(z)
        bound = cut + compensation

        
        bound_info = {
            'X': X, 
            'z': z,
            'cut': cut
        }
        
        return bound, bound_info
    
    def solve_subproblem(self, sub_uf, z=None, **kwargs):
        self.logger.debug(f"Solving subproblem with size {sub_uf.n_rep}")
        if z is None:
            assert sub_uf.is_root, "SOMETHING IS WRONG with your logic"
            problem = kwargs['problem']
            z, X = QRRMaxCutSolver.qrr(self, problem, self.opt_params, approx_style=self.approx_style)
            
            cut = self.cache_root_problem.evaluate_solution(z)
            bound_info = {
                'X': X, 
                'z': z,
                'cut': cut
            }
            return z, bound_info 
        
        return z, {}