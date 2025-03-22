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


import numpy as np
import networkx as nx

from rbnbr.solver.quantum.branching_rules import branch_confidence, branch_hardest, branch_easiest
# ---------------------------------------------------------
# Data structures and placeholder for the bounding method
# ---------------------------------------------------------

@dataclass
class QBB(QRRMaxCutSolver, ABC):
    _: KW_ONLY
    random_pass: bool = False
    search_style: str = 'bfs' 
    gap: int = 0
    
    def solve(self, problem: MaxCutProblem, optim_params=None, qrr_information=None) -> Solution:
        optim_params = self._watch_optim_params(optim_params)
        if qrr_information is not None:
            if self.X_type != qrr_information[2]:
                msg = f"Ignored: The specified QRR information has X_type {qrr_information[2]}, but the current QRR solver has X_type {self.X_type}"
                logging.warning(msg)
                qrr_information = None
            else:
                qrr_information = copy.deepcopy(qrr_information)

        bc = self._main_loop(problem, optim_params, qrr_information)
        return bc
        
    def wd_qrr_info(self, qrr_information):
        if qrr_information is not None:
            if self.X_type != qrr_information[2]:
                msg = f"Ignored: The specified QRR information has X_type {qrr_information[2]}, but the current QRR solver has X_type {self.X_type}"
                logging.warning(msg)
                qrr_information = None
        return qrr_information
    
    def _main_loop(self, original_problem, optim_params, qrr_information):
        
        ###################
        #  INITIALIZATION 
        ###################
        ### because the search is exponential, we restrict the number of search
        OP_MAX = original_problem.graph.number_of_nodes()
        n_op = 0
        ### The best solution and the bound (should always be updated together)
        cut_bound = 0
        best_sol = None
        
        ### The root node of the search tree
        root_uf = BUF(original_problem.N)
        
        ### This todo list is either a queue or a stack, depending on the search style
        todo = [(root_uf, {})]  # (UF structure, unassigned edges, partial cut value, partial constraints)
        
        ### This is a container for storing the subproblem history
        my_solution = Solution()
        
        
        ############################
        #  Tree Search 
        ############################
        while todo and n_op < OP_MAX:
            ### Get the next subproblem depending on the search style
            if self.search_style == 'bfs':
                subp_uf, _ = todo.pop()
            elif self.search_style == 'dfs':
                subp_uf, _ = todo.pop(0)
            n_op += 1
            
            #####################################
            #  Solve the subproblem 
            #####################################
            heuristic, bitstr, cut_val = self._solve_subproblem(subp_uf, original_problem, optim_params, qrr_information)
           
            ################################
            #  Post-processing 
            ################################
            prune_flag = (cut_val < cut_bound - self.gap) and (random.random() < 0.5)

            ### update the best solution is done differently then pruning
            update_flag = (cut_val > cut_bound)
            
            ### update the best solution
            if update_flag:
                cut_bound = cut_val
                best_sol = bitstr
                my_solution.add_step(
                    solution=bitstr, 
                    cost=cut_val,
                    elapsed_time=0.0,
                    approx_ratio=original_problem.approx_ratio(bitstr),
                )

            #########################################################
            # Schedule the subproblems with todo
            #########################################################
            if prune_flag:
                continue
            else:
                u, v = self.pick_branching_pair(heuristic)
                
                uf_same = BUF.clone(subp_uf)  
                if uf_same.union(u, v, 0):
                    todo.append((uf_same, {}))

                uf_diff = BUF.clone(subp_uf)
                if uf_diff.union(u, v, 1):
                    todo.append((uf_diff, {}))
        
        # add the best step (because the last subproblem step is not necessarily the best)
        my_solution.add_step(
            solution=best_sol,
            cost=cut_bound,
            elapsed_time=0.0,
            approx_ratio=original_problem.approx_ratio(best_sol),
        )
            
        return my_solution
    
    @abstractmethod
    def pick_branching_pair(self, heuristic):
        pass
    
    @abstractmethod
    def _solve_subproblem(self, subp_uf, original_problem, optim_params, qrr_information):
        pass

@dataclass
class Edge_QBB_MC(QBB):
    """
    Not solvign the big problem, but just perform branch and bound rounding
    """
    _: KW_ONLY
    branching_strategy: str = "r1"
    def __post_init__(self):
        super().__post_init__() 

    
    def _solve_subproblem(self, subp_uf, original_problem, optim_params, qrr_information):
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
        subprob_materials = subp_uf.get_subproblem(original_problem)
        if subprob_materials is None:
            # Subproblem is infeasible
            return None, None, None, -1
        
        sub_p, qb_to_rep, rep_to_qb = subprob_materials
    
        # Check if this is the root problem (no variables fixed yet)
        is_root_prob = (sub_p.N == original_problem.N)
        
        if qrr_information is not None and is_root_prob:
            # If we have pre-computed QRR information for the root problem, use it
            sol_full, X, _ = qrr_information
            cut_full = original_problem.evaluate_solution(sol_full)
            return qb_to_rep, X, sol_full, cut_full
            
        else:
            # Otherwise, solve the subproblem with QRR
            if is_root_prob:
                # For root problem, use provided optimization parameters
                curr_sol, X = QRRMaxCutSolver.qrr(self, sub_p, optim_params)
            else:
                # For subproblems, use default parameters
                curr_sol, X = QRRMaxCutSolver.qrr(self, sub_p)
                
            # Convert the reduced solution to the original solution
            sol_full = subp_uf.get_original_solution(curr_sol, rep_to_qb)
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
         

class BUF:
    # Union-Find structure 
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
        
        
        
    def get_subproblem(self, root_problem):
        """
        Constructs a Max-Cut cost Hamiltonian for the reduced graph.
        The reduction is based on the union-find structure.
        
        Assumes graph is a networkx.Graph with nodes labeled 0...n-1.
        Edge weights are taken from the 'weight' attribute (default=1).
        
        The cost Hamiltonian for Max-Cut on the reduced graph is:
            H = sum_{(i,j) in E_reduced} w_{ij}/2 * (I - Z_i Z_j)
        where i,j index the reduced nodes (i.e., unique union-find groups).
        
        Returns:
            A Qiskit operator (SummedOp) representing the Hamiltonian.
        """
        graph = root_problem.graph
        assert graph.number_of_nodes() == len(self.parent), "Graph and UF structure must have the same number of nodes"
        # Step 1: Build the mapping from original node -> group representative.


        # Step 2: Determine the unique groups and create a reduced graph.
        reduced_nodes = set(self.parent)
        qb_to_rep = sorted(list(reduced_nodes))
        rep_to_qb = {rep: idx for idx, rep in enumerate(qb_to_rep)}
        
        if len(reduced_nodes) == graph.number_of_nodes():
            return root_problem, qb_to_rep, rep_to_qb
        
        else:
            reduced_graph = nx.Graph()
            for site in qb_to_rep:
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
                    dw = data.get('weight', 1) * total_flip
                    if reduced_graph.has_edge(rep_u, rep_v):
                        reduced_graph[rep_u][rep_v]['weight'] += dw
                    else:
                        reduced_graph.add_edge(rep_u, rep_v, weight=dw)
                        
            # We will index the reduced nodes in some fixed order.
            if reduced_graph.number_of_edges() == 0:
                return None
            
        
            mc_subproblem = MaxCutProblem(reduced_graph, solve=False)
            return mc_subproblem, qb_to_rep, rep_to_qb
    
    
    def get_original_solution(self, reduced_solution, rep_to_qb):
        sol = np.array([0] * len(self.parent))
        
        for node in range(len(self.parent)):
            rep = self.parent[node]
            offset = self.offset[node]
            group_qb = rep_to_qb[rep]
            group_bit = reduced_solution[group_qb]
            sol[node] = group_bit ^ offset
        
        return sol.astype(int).tolist()
    
    
    
    
  
  
