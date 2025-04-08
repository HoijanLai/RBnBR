"""
Quantum Branch and Bound (QBB) implementation for combinatorial optimization problems.

Author: Haixin Li (Peter Lai)
Email: laihoijan@gmail.com
Organization: Technical University of Munich
License: MIT
Date: 2023-03-29

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.   
"""

from abc import ABC, abstractmethod
import copy
from dataclasses import KW_ONLY, dataclass, field

from rbnbr.solver.solution import Solution
from rbnbr.problems.max_cut import MaxCutProblem

import numpy as np
import networkx as nx



import logging
from rbnbr.solver.BnB.branching_rules import branch_confidence, branch_hardest, branch_easiest
# ---------------------------------------------------------
# Data structures and placeholder for the bounding method
# ---------------------------------------------------------




def get_subproblem(problem_info, rep_u, rep_v, sub_UF, parity):
    """
    Perform variable elimination on the parent graph
    Exactly one of them is in the removed graph
    """
    parent_graph = problem_info['problem'].graph
    compensation = problem_info['compensation']
    laplacian = problem_info['laplacian']
    new_laplacian = get_new_sub_L(laplacian, rep_u, rep_v, sub_UF, parity)
    
    node_set = set(sub_UF.parent)

    ### identify the node kept and removed by the Union-Find operation
    kept_node, rm_node = None, None
    if rep_u in node_set:
        kept_node, rm_node = rep_u, rep_v
    else: 
        kept_node, rm_node = rep_v, rep_u
        
    # assert parity == 1 - 2*sub_UF.offset[rm_node], "SOMETHING is WRONG!!"
    
    reduced_graph = parent_graph.subgraph(node_set).copy()
    # if rm_node in reduced_graph.nodes():
    #     assert rm_node not in reduced_graph.nodes(), 'rm_node should not be in the reduced graph'
    
    ### Create the reduced problem by variable elimination
    for rep_a, rep_b, w in parent_graph.edges(rm_node, 'weight'):
        if rep_a == rep_b or rep_b == kept_node: 
            continue
        # assert rep_a == rm_node, 'networkx doesn\'t guarantee the order of the edges'
        if reduced_graph.has_edge(rep_b, kept_node):
            reduced_graph[rep_b][kept_node]['weight'] += w*parity
        else:
            reduced_graph.add_edge(rep_b, kept_node, weight=w*parity)
    
    
    mc_subproblem = MaxCutProblem(reduced_graph, solve=sub_UF.will_brute_force)
        
    # assert mc_subproblem.N == len(sub_UF.index_map), "SOMETHING is WRONG!!"
    return {
        'problem': mc_subproblem,
        'laplacian': new_laplacian,
        'compensation': compensation + 0.25 * laplacian[rm_node, rm_node]
    }
    
       
def get_new_sub_L(L, rep_u, rep_v, sub_UF, parity):
    new_L = L.copy()
    kept_node, rm_node = None, None
    node_set = set(sub_UF.parent)
    if rep_u in node_set:
        kept_node, rm_node = rep_u, rep_v
    else: 
        kept_node, rm_node = rep_v, rep_u
        
    new_L[kept_node, :] += parity*new_L[rm_node, :]
    new_L[:, kept_node] += parity*new_L[:, rm_node]
    new_L[rm_node, :] = 0
    new_L[:, rm_node] = 0
    
    return new_L
    
            
            
@dataclass
class BnB(ABC):
    _: KW_ONLY
    bf_threshold: int = 18
    top_pair_only: bool = field(default=False, metadata={'help': 'Whether to only branch on the top pair'})
    logger: logging.Logger = field(default=None, metadata={'help': 'Logger'})
    
    def __post_init__(self, *args, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.ERROR)
        self.init_todo()
    
    def main_loop(self, original_problem, OP_MAX=float('inf')):
        self.n_pruned = 0
        root_problem_info = self.root_problem_info(original_problem)
        ###################
        # 👽 INITIALIZATION 
        ###################
        ### because the search is exponential, we restrict the number of search
        n_op = 0
        ### The best solution and the bound (should always be updated together)
        best_cut = 0
        best_sol = None
        
        ### The root node of the search tree
        root_UF = BUF(original_problem.N, self.bf_threshold)
    
        ### This is a container for storing the subproblem history
        my_solution = Solution()
        
        ### Initialize the tree search
        self.init_todo()
        self.add_node((root_UF, np.inf, root_problem_info))
        
        
        ############################
        # 🌳 START : Tree Search 
        ############################
        best_sub_UF = None
        while self.has_next() and n_op < OP_MAX:
            
            #########################################################
            # 🔍 Process the current subproblem
            #########################################################
            sub_UF, _, problem_info = self.next_node()
            

            # solve the subproblem
            z, solution_info = self.solve_subproblem(sub_UF, **problem_info)
            cut_val = root_problem_info['problem'].evaluate_solution(z)
            problem_info.update(solution_info)
            n_op += 1
            
            # pruning decision
            if cut_val > best_cut:
                best_cut, best_sol, best_sub_UF = cut_val, z, sub_UF
                self.logger.debug("BETTER CUT: %s", best_cut)
                self.logger.debug("New best cut has path \n  %s", ('\n'+2*' ').join([f"{x} {y} {parity}" for x, y, parity in sub_UF.history]))
                my_solution.add_step(
                    solution=best_sol,
                    cost=best_cut,
                    elapsed_time=0.0,
                    approx_ratio=root_problem_info['problem'].approx_ratio(best_sol),
                    n_steps=n_op,
                    tree_path=best_sub_UF.history,
                    n_pruned=self.n_pruned
                )
        
            
            pair_rank = self.pick_branching_pair(sub_UF, **problem_info, top_one=self.top_pair_only)
            

            for rep_u, rep_v in pair_rank:
                # if not root_problem_info['problem'].graph.has_edge(rep_u, rep_v):
                #     continue
                left_node, right_node = self._create_branch_nodes(sub_UF, rep_u, rep_v, problem_info, best_cut)
                        
                if left_node is not None or right_node is not None:
                    self.add_left_right_node(left_node, right_node)
                    break  # 🐈 # IMPORTANT: Stop after finding the first valid branching pair
                    
                
                
        # add the best step (because the last subproblem step is not necessarily the best)
        my_solution.add_step(
            solution=best_sol,
            cost=best_cut,
            elapsed_time=0.0,
            approx_ratio=root_problem_info['problem'].approx_ratio(best_sol),
            n_steps=n_op,
            tree_path=best_sub_UF.history,
            n_pruned=self.n_pruned
        )
            
        return my_solution
    
    
    def _create_branch_nodes(self, sub_UF, rep_u, rep_v, problem_info, best_cut):
        #############################################################################
        # 📝 TRY TO 
        # Schedule the subproblems 
        # with todo if the bound is better than the best cut
        
        # INPUT:
        # sub_UF:       the current Union-Find structure
        # rep_u, rep_v: the selected two nodes to branch on
        # problem_info: the problem information matching sub_UF
        #                   this is the information off which the branch problems is based on 
        # best_cut:     the best cut found so far
        
        # OUTPUT:
        # left_node, right_node: the two new nodes to add to the tree
        #############################################################################

        uf_same = BUF.clone(sub_UF)  
        uf_diff = BUF.clone(sub_UF)
        left_node = None
        right_node = None
        # A branch will survive if both union operations are successful
        # and at least one of them has better bound than the best cut
        
        if uf_same.union(rep_u, rep_v, 0) and uf_diff.union(rep_u, rep_v, 1):
            self.logger.debug("Branching on edge %s %s", rep_u, rep_v)
            self.logger.debug("Problem reduced to %d variables", uf_same.n_rep)
            
            
            # left node
            problem_info_same = get_subproblem(problem_info, rep_u, rep_v, uf_same, 1)
            bound_same, bound_info_same = self.get_bound(uf_same, **problem_info_same)
            self.logger.debug("Bound same: %s", bound_same)
            if bound_same >= best_cut:
                problem_info_same.update(bound_info_same)
                left_node = self.make_node(uf_same, bound_same, problem_info_same)
            else:
                self.n_pruned += 1
                self.logger.debug("Pruning %s=%s because bound %s <= best_cut %s", rep_u, rep_v, bound_same, best_cut)
                pass
            
            # right node
            problem_info_diff = get_subproblem(problem_info, rep_u, rep_v, uf_diff, -1)
            bound_diff, bound_info_diff = self.get_bound(uf_diff, **problem_info_diff)
            self.logger.debug("Bound diff: %s", bound_diff)
            if bound_diff >= best_cut:
                problem_info_diff.update(bound_info_diff)
                right_node = self.make_node(uf_diff, bound_diff, problem_info_diff)
            else:
                self.n_pruned += 1
                self.logger.debug("Pruning %s!=%s because bound %s <= best_cut %s", rep_u, rep_v, bound_diff, best_cut)
                pass
            
        return left_node, right_node
    
    
    def root_problem_info(self, original_problem):
        self.cache_root_problem = copy.deepcopy(original_problem)
        return {
            'problem': original_problem,
            'compensation': 0,
            'laplacian': nx.laplacian_matrix(original_problem.graph).toarray()
        }
    
    
    @abstractmethod
    def pick_branching_pair(self, sub_UF, **problem_info):
        pass
    
    @abstractmethod
    def solve_subproblem(self, subp_UF, **problem_info):
        pass
    
    @abstractmethod
    def get_bound(self, subp_UF, **problem_info):
        pass
    
    
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


         

class BUF:
    
    @property
    def index_map(self):
        """
        a fixed and reliable rules to map qubits to representatives
        """
        return sorted(list(set(self.parent)))
    
    @property
    def index_map_inv(self):
        """
        the inverse of the index map
        """
        return {v: k for k, v in enumerate(self.index_map)}
    
    
    @property
    def is_root(self):
        return self.n_var == self.n_rep
    
    @property
    def n_var(self):
        return len(self.parent)
    
    # Union-Find structure 
    @property
    def depth(self):
        return len(self.parent) - len(set(self.parent)) + 1
    
    @property
    def n_rep(self):
        return len(set(self.parent))
    
    @property
    def will_brute_force(self):
        return self.n_rep <= self.brute_force_threshold
    
    def __init__(self, n, brute_force_threshold=18):
        self.parent = list(range(n))
        self.rank = [0]*n # 'depth' of its children
        self.offset = [0]*n  # 0 or 1, parity relative to the root of its set
        self.brute_force_threshold = brute_force_threshold
        self.history = []
        
        
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
            success = ((self.offset[x] ^ self.offset[y]) == parity) 
            if success:
                self.history.append((x, y, parity))
                return True
            else:
                return False

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
            self.path_compression()
            self.history.append((x, y, parity))
            return True
        
    @staticmethod
    def clone(uf):
        new_UF = BUF(len(uf.parent))
        new_UF.parent = uf.parent[:]
        new_UF.rank = uf.rank[:]
        new_UF.offset = uf.offset[:]
        new_UF.brute_force_threshold = uf.brute_force_threshold
        new_UF.history = uf.history[:]
        return new_UF
    
    
    def path_compression(self):
        for i in range(len(self.parent)):
            self.find(i)
    
    def get_original_solution(self, z_sub, **kwargs):
        index_map = self.index_map

        if len(z_sub) == self.n_var:
            return z_sub
        
        sol = np.zeros(self.n_var, dtype=int)
        
        for original_idx in range(len(self.parent)):
            rep = self.parent[original_idx]
            offset = self.offset[original_idx]
            
            idx = index_map.index(rep)
            group_bit = z_sub[idx]
            
            sol[original_idx] = group_bit^offset
        
        return sol.astype(int).tolist()
    
    
    
    
  
  
