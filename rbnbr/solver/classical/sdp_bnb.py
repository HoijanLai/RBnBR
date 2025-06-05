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

from rbnbr.solver.classical.gw import GWMaxCutSolver

@dataclass
class QRR_BnB_MC(BnB, GWMaxCutSolver):
    _: KW_ONLY
    branching_strategy: str = field(default="r1")
    approx_style: str = field(default='default')
    search_style: str = field(default='bfs')
    optim_params: list = field(default=None)
    
    
    def init_todo(self):
        self.todo = []
    
    def has_next(self):
        return len(self.todo) > 0
    
    def next_node(self):
        if self.search_style == 'dfs':
            return self.todo.pop()
        elif self.search_style == 'bfs':
            return self.todo.pop(0)
        else:
            raise ValueError(f"Invalid search style: {self.search_style}")
    
    def add_node(self, node):
        self.todo.append(node)
        
    def make_node(self, uf, bound, problem_info):
        return (uf, bound, problem_info)
    
    def add_left_right_node(self, left_node, right_node):
        if left_node is not None:
            self.todo.append(left_node)
        if right_node is not None:
            self.todo.append(right_node)
    
    def __post_init__(self): 
        super().__post_init__()
        self.todo = []

    
    def solve(self, problem: MaxCutProblem):
        bc = self.main_loop(problem)
        return bc


    def root_problem_info(self, original_problem):
        return {
            'problem': copy.deepcopy(original_problem),
            'index_map': list(range(original_problem.N))
        }

    def solve_subproblem(self, sub_uf, problem, index_map, **kwargs):
    
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
                z_sub, X = QRRMaxCutSolver.qrr(self, problem, approx_style=self.approx_style)
                
            # Convert the reduced solution to the original solution
        
        z = sub_uf.get_original_solution(z_sub, index_map)
        
        sol_info = {
            'X': X
        }
        return z, sol_info


    def pick_branching_pair(self, X, index_map, **kwargs):
        
        if X is None:
            return []

        if self.branching_strategy == 'r1':
            pairs = branch_hardest(X, **kwargs)
        elif self.branching_strategy == 'r2':
            pairs = branch_confidence(X, **kwargs)
        elif self.branching_strategy == 'r3':
            pairs = branch_easiest(X, **kwargs)
        else:
            raise ValueError(f"Invalid strategy: {self.branching_strategy}")
        
        pairs = [(index_map[qb_u], index_map[qb_v]) for qb_u, qb_v in pairs]

        return pairs
    
    def get_bound(self, sub_uf, problem, **k):
        if problem is None:
            return np.inf, {}
        
        if problem.ref_solution_arr is not None:
            return np.inf, {}
        
        n = problem.graph.number_of_nodes()
        
        if self.optimize_correction:
            
            
            u = cp.Variable(n)
            t = cp.Variable()  # This will represent our bound
            
            L = nx.laplacian_matrix(problem.graph).toarray()
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
                logging.warning(f"Optimization failed: {e}, using simple eigenvalue bound")
                # Fall back to simple eigenvalue bound if optimization fails
                
            
        L = nx.laplacian_matrix(problem.graph).toarray()
        eigvals = np.linalg.eigvals(L)
        return (n/4) * np.max(eigvals), {}
    
    
    def _check_qrr_info(self, qrr_information):
        if qrr_information is not None:
            if self.X_type != qrr_information[2]:
                msg = f"Ignored: The specified QRR information has X_type {qrr_information[2]}, but the current QRR solver has X_type {self.X_type}"
                logging.warning(msg)
                qrr_information = None
            else:
                qrr_information = copy.deepcopy(qrr_information)
        return qrr_information



