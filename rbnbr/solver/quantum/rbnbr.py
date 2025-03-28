from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, KW_ONLY
import logging
import random
import warnings
from qiskit_optimization.algorithms.qrao import (
    QuantumRandomAccessEncoding,
    SemideterministicRounding,
    QuantumRandomAccessOptimizer,
    MagicRounding,
)
from qiskit_optimization.applications import Maxcut
from qiskit.primitives import Sampler, Estimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import RealAmplitudes

from rbnbr.solver.quantum.qaoa import QAOAMaxCutSolver
from rbnbr.solver.solver_base import ApproximationSolver
from rbnbr.solver.solution import Solution
from rbnbr.problems.max_cut import MaxCutProblem
from rbnbr.solver.quantum.qrr import QRRMaxCutSolver

from collections import defaultdict

import cvxpy as cp
import numpy as np
import networkx as nx

from rbnbr.solver.quantum.branching_rules import branch_confidence, branch_hardest, branch_easiest
# ---------------------------------------------------------
# Data structures and placeholder for the bounding method
# ---------------------------------------------------------

@dataclass
class QBB(QRRMaxCutSolver, ABC):
    _: KW_ONLY
    random_pass: float = 0.0
    search_style: str = 'dfs' 
    gap: int = 0
    bound_with_solver = False
    
    def solve(self, problem: MaxCutProblem, optim_params=None, qrr_information=None, *, verbose=False) -> Solution:
        optim_params = self._watch_optim_params(optim_params)
        if qrr_information is not None:
            if self.X_type != qrr_information[2]:
                msg = f"Ignored: The specified QRR information has X_type {qrr_information[2]}, but the current QRR solver has X_type {self.X_type}"
                logging.warning(msg)
                qrr_information = None
            else:
                qrr_information = copy.deepcopy(qrr_information)

        bc = self._main_loop(problem, optim_params, qrr_information, verbose)
        return bc
        
    def wd_qrr_info(self, qrr_information):
        if qrr_information is not None:
            if self.X_type != qrr_information[2]:
                msg = f"Ignored: The specified QRR information has X_type {qrr_information[2]}, but the current QRR solver has X_type {self.X_type}"
                logging.warning(msg)
                qrr_information = None
        return qrr_information
    
    def _main_loop(self, original_problem, optim_params, qrr_information, verbose=False):
        
        gap = self.gap
        avg_deg = original_problem.graph.number_of_edges() / original_problem.graph.number_of_nodes()
        
        problem_info = {
            'avg_deg': avg_deg,
        }
        
        ###################
        #  INITIALIZATION 
        ###################
        ### because the search is exponential, we restrict the number of search
        OP_MAX = 3 * original_problem.graph.number_of_nodes()
        n_op = 0
        ### The best solution and the bound (should always be updated together)
        best_cut = 0
        best_sol = None
        
        ### The root node of the search tree
        root_uf = BUF(original_problem.N)
        
        ### This todo list is either a queue or a stack, depending on the search style
        todo = [(root_uf, None)]  # (UF structure, unassigned edges, partial cut value, partial constraints)
        
        ### This is a container for storing the subproblem history
        my_solution = Solution()
        
        
        ############################
        #  Tree Search 
        ############################
        while todo and n_op < OP_MAX:
            ### Get the next subproblem depending on the search style
            if self.search_style == 'dfs':
                subp_uf, parent_problem = todo.pop()
            elif self.search_style == 'bfs':
                subp_uf, parent_problem = todo.pop(0)
            n_op += 1
            

            sub_p, qb_to_rep = subp_uf.get_subproblem(parent_problem)

            bound = self.bound(subp_uf, sub_p, problem_info)
            if verbose:
                print(f"Bound: {bound}")
            
            if sub_p is None:
                sub_p = original_problem
            
            will_branch = bound > best_cut

            if will_branch: # also means it is worth solving the subproblem
                heuristic, bitstr, cut_val = \
                    self.solve_subproblem(
                        subp_uf, 
                        sub_p, 
                        qb_to_rep,
                        original_problem, 
                        optim_params=optim_params, 
                        qrr_information=qrr_information)
                    
                if cut_val > best_cut:
                    if verbose:
                        print(f"Found a better solution: {cut_val}")
                    best_cut = cut_val
                    best_sol = bitstr
                    
                    my_solution.add_step(
                        solution=bitstr, 
                        cost=cut_val,
                        elapsed_time=0.0,
                        approx_ratio=original_problem.approx_ratio(bitstr),
                    )
                    
                u, v = self.pick_branching_pair(heuristic)
                if verbose:
                    print(f"Branching on {u} and {v}")
                
                uf_same = BUF.clone(subp_uf)  
                uf_diff = BUF.clone(subp_uf)
                if uf_same.union(u, v, 0) and uf_diff.union(u, v, 1):
                    todo.append((uf_same, sub_p))
                    todo.append((uf_diff, sub_p))
                
                    
                    
        
        # add the best step (because the last subproblem step is not necessarily the best)
        my_solution.add_step(
            solution=best_sol,
            cost=best_cut,
            elapsed_time=0.0,
            approx_ratio=original_problem.approx_ratio(best_sol),
        )
            
        return my_solution
    
    @abstractmethod
    def pick_branching_pair(self, heuristic):
        pass
    
    @abstractmethod
    def solve_subproblem(self, subp_uf, original_problem, optim_params, qrr_information):
        pass
    
    @abstractmethod
    def bound(self, subp_uf, problem_info):
        pass

@dataclass
class Edge_QBB_MC(QBB):
    """
    Not solvign the big problem, but just perform branch and bound rounding
    """
    _: KW_ONLY
    branching_strategy: str = "r1"
    approx_style: str = 'default'
    def __post_init__(self):
        super().__post_init__() 

    
    def solve_subproblem(
        self, 
        subp_uf, 
        sub_p, # subproblem 
        qb_to_rep, # mapping the node in sub_p to its qubit index
        original_problem, # original problem for evaluating the solution
        *,
        optim_params=None, 
        qrr_information=None):
        """
        Solves a subproblem in the branch and bound process.
        
        Args:
            subp_uf: Union-find data structure representing the subproblem constraints
            original_problem: The original MaxCut problem instance
            optim_params: Optimization parameters for QRR solver
            qrr_information: Pre-computed QRR solution information (if available)
            
        Returns:
            qb_to_rep: Mapping from qubit indices to representative indices
            X: Correlation matrix from QRR
            sol_full: Solution mapped to the original problem space
            cut_full: Objective value of the solution
        """
        # Get the subproblem materials (problem instance and mappings)
        
        if sub_p is None:
            # Subproblem is infeasible
            return None, None, -1
    
        # Check if this is the root problem (no variables fixed yet)
        is_root_prob = sub_p is None
        
        if qrr_information is not None and is_root_prob:
            # If we have pre-computed QRR information for the root problem, use it
            sol_full, X, _ = qrr_information
            cut_full = original_problem.evaluate_solution(sol_full)
            return qb_to_rep, X, sol_full, cut_full
            
        else:
            # Otherwise, solve the subproblem with QRR
            if is_root_prob:
                # For root problem, use provided optimization parameters
                curr_sol, X = QRRMaxCutSolver.qrr(self, original_problem, optim_params, approx_style=self.approx_style)
            else:
                # For subproblems, use default parameters
                curr_sol, X = QRRMaxCutSolver.qrr(self, sub_p, approx_style=self.approx_style)
                
            # Convert the reduced solution to the original solution
            sol_full = subp_uf.get_original_solution(curr_sol, qb_to_rep)
            cut_full = original_problem.evaluate_solution(sol_full)
            
        heuristic = {
            'qubit_mapping': qb_to_rep,
            'correlation_matrix': X
        }
        return heuristic, sol_full, cut_full


    def pick_branching_pair(self, heursitc):
        
        qb_to_rep = heursitc['qubit_mapping']
        X = heursitc['correlation_matrix']
        

        if self.branching_strategy == 'r1':
            qb_u, qb_v = branch_hardest(X)
        elif self.branching_strategy == 'r2':
            qb_u, qb_v = branch_confidence(X)
        elif self.branching_strategy == 'r3':
            qb_u, qb_v = branch_easiest(X)
        else:
            raise ValueError(f"Invalid strategy: {self.branching_strategy}")
        
        u, v = qb_to_rep[qb_u], qb_to_rep[qb_v]
        
        return u, v
    
    def bound(self, subp_uf, sub_p, problem_info):
        if sub_p is None:
            return np.inf

        n = sub_p.graph.number_of_nodes()
        
        u = cp.Variable(n)
        t = cp.Variable()  # This will represent our bound
        
        L = nx.laplacian_matrix(sub_p.graph).toarray()
        M = (n/4) * (L + cp.diag(u)) + cp.diag(u)
        
        constraints = [
            cp.sum(u) == 0,
            M << t * np.eye(n)
        ]
        
        prob = cp.Problem(cp.Minimize(t), constraints)
        
        try:
            prob.solve()
            
            u_opt = u.value
            M_opt = (n/4) * (L + np.diag(u_opt))
            
            # Compute largest eigenvector
            evals = np.linalg.eigvals(M_opt)
            return np.max(evals)
            
        except Exception as e:
            # Fall back to simple eigenvalue bound if optimization fails
            L = nx.laplacian_matrix(sub_p.graph).toarray()
            eigvals = np.linalg.eigvals(L)
            return 0.25 * np.max(eigvals) * sub_p.N
         

class BUF:
    # Union-Find structure 
    @property
    def n_var(self):
        return len(self.parent)
    
    @property
    def depth(self):
        return len(self.parent) - len(set(self.parent)) + 1
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n # 'depth' of its children
        self.offset = [0]*n  # 0 or 1, parity relative to the root of its set
        

    def find(self, x):
        """
        Find and Update
        Find with path-compression.  Returns the root of x.
        Also updates offset[x] so that it is offset to the new root.
        """
        if self.parent[x] != x:
            root = self.find(self.parent[x])
            # update offset[x] based on parent's offset
            self.offset[x] ^= self.offset[self.parent[x]]
            self.parent[x] = root
        return self.parent[x]

    def union(self, x, y, parity):
        """
        Union the sets containing x and y.
        parity = 0 means x and y should be in same set
        parity = 1 means x and y should be in different sets

        Returns True if merge is successful (no conflict).
        Returns False if there's a contradiction.
        """
        rx = self.find(x)
        ry = self.find(y)
        
        # x and y are ALREADY in the same connected component
        #   * no need to merge
        #   * check if the parity is consistent with the union
        if rx == ry: 
            return ((self.offset[x] ^ self.offset[y]) == parity) 

        # x and y are in different connected components
        #   * merge the two components
        #   * no need to check parity
        else: 
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx  # swap so rx always refers to the higher- or equal-rank root

            self.parent[ry] = rx  # attach the 'ry' tree under 'rx'

            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1 # only increment the rank when two components have the same rank
                    
            # since the parent of ry is now rx, we need to update the offset of ry
            self.offset[ry] = self.offset[x] ^ self.offset[y] ^ parity
            return True
        
    @staticmethod
    def clone(uf):
        new_uf = BUF(len(uf.parent))
        new_uf.parent = uf.parent[:]
        new_uf.rank = uf.rank[:]
        new_uf.offset = uf.offset[:]
        return new_uf
        
        
        
    def get_subproblem(self, parent_problem):
        if parent_problem is None:
            # no parent problem, nothing to do
            return None, list(range(self.n_var))
        
        else: 
            graph = parent_problem.graph
            reduced_nodes = set(self.parent)
            qb_to_rep = sorted(list(reduced_nodes))
            # print(f"Depth: {self.depth}")
            # Step 1: Build the mapping from original node -> group representative.


            # Step 2: Determine the unique groups and create a reduced graph.
            reduced_nodes = set(self.parent)
            qb_to_rep = sorted(list(reduced_nodes))
            
            
            reduced_graph = nx.Graph()
            for site in qb_to_rep: # qubit is from 0 to n-1
                reduced_graph.add_node(site)
            
            # For each edge in the original graph, add an edge between groups (if not same)
            for u, v, data in graph.edges(data=True):
                rep_u = self.parent[u]
                rep_v = self.parent[v]
                if rep_u == rep_v:
                    # Intra-group edge; in a proper Max-Cut these vertices are forced together.
                    continue
                else:
                    total_flip = 1 - 2 * (self.offset[u] ^ self.offset[v]) 
                    new_w = data.get('weight', 1) * total_flip
                    if reduced_graph.has_edge(rep_u, rep_v):
                        reduced_graph[rep_u][rep_v]['weight'] += new_w
                    else:
                        reduced_graph.add_edge(rep_u, rep_v, weight=new_w)
                        
            # We will index the reduced nodes in some fixed order.
            if reduced_graph.number_of_edges() == 0:
                return None
            
        
            mc_subproblem = MaxCutProblem(reduced_graph, solve=False)
            return mc_subproblem, qb_to_rep
    
    
    def get_original_solution(self, reduced_solution, qb_to_rep):
        rep_to_qb = {rep: idx for idx, rep in enumerate(qb_to_rep)}
        sol = np.array([0] * len(self.parent))
        
        for node in range(len(self.parent)):
            rep = self.parent[node]
            offset = self.offset[node]
            group_qb = rep_to_qb[rep]
            group_bit = reduced_solution[group_qb]
            sol[node] = group_bit ^ offset
        
        return sol.astype(int).tolist()
    
    
    
    
  
  
