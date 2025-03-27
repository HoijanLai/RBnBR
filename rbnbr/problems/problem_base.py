# Standard library imports
import copy
import pickle
from typing import List
import warnings

# Third party imports
import networkx as nx
import numpy as np
import pandas as pd

# Local imports
from rbnbr.solver.solution import Solution

class CombProblemBase:
    
    def solution_summary(self):
        
        data = []
        for solution_type, breadcrumbs in self._solutions.items():
            if breadcrumbs is not None:
                row = {
                    'Solution Type': solution_type,
                    'Cut Value': breadcrumbs.cost,
                    'Approximation Ratio': breadcrumbs.approx_ratio,
                    'Number of Steps': len(breadcrumbs),
                    # 'Time (s)': f"{breadcrumbs.elapsed_time:.4f}"
                }
                # if hasattr(breadcrumbs, 'approx_ratio') and breadcrumbs.approx_ratio is not None:
                #     row['Approximation Ratio'] = f"{breadcrumbs.approx_ratio:.4f}"
                data.append(row)
                
        if not data:
            print(f"No solutions found for {self.name}")
            return None
            
        df = pd.DataFrame(data)
        df = df.sort_values('Cut Value', ascending=False)
        return df
    
    def __init__(self, graph, problem_type: str):
        self.problem_type = problem_type
        
        self._graph = copy.deepcopy(graph)
        self._metadata = {
            'name': graph.name,
        }
        
        self.vis_pos = nx.spring_layout(graph)
        self._solutions = {
            'exact': None
        }

    @property
    def graph(self):
        return self._graph
    
    @property
    def adjacency_matrix(self):
        return nx.to_numpy_array(self.graph)
        
    def solutions(self, solution_type: str = None):
        if solution_type is not None:
            return self._solutions[solution_type]
        else:
            return self._solutions
    
    def add_solution(self, solution_type: str, solution: Solution):
        self._solutions[solution_type] = solution
        
    


     
class ProblemSet:
    def __init__(self, problems: List[CombProblemBase]=None):
        if problems is None:
            self._problems = []
            self._problem_names = []
        else:
            self._problems = problems
            self._problem_names = [problem.name for problem in problems]
        
    def __getitem__(self, index):
        return self._problems[index]
    
    def add_problem(self, problem: CombProblemBase):
        if problem.name in self._problem_names:
            warnings.warn(f'Problem {problem.name} already exists')
        else:
            self._problems.append(problem)
            self._problem_names.append(problem.name)
    
    def __len__(self):
        return len(self._problems)
        
    def sort_by_V(self):
        self._problems.sort(key=lambda x: x.graph.number_of_nodes())
        
    def sort_by_E(self):
        self._problems.sort(key=lambda x: x.graph.number_of_edges())
        
    def sort_by_density(self):
        self._problems.sort(key=lambda x: x.graph.density())
        
    def sort_by_max_degree(self):
        self._problems.sort(key=lambda x: max(x.graph.degree()))
        
    def sort_by_min_degree(self):
        self._problems.sort(key=lambda x: min(x.graph.degree()))
        
    def method_performance(self, method_name: str):
        data = []
        for problem in self._problems:
            performance = problem.solutions(method_name).data
            performance['problem_name'] = problem.name
            data.append(performance)
        return pd.DataFrame(data)[['problem_name', 'cost', 'approx_ratio', 'n_steps']]

        

    def save_problems(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_problems(cls, file_path: str):
        with open(file_path, 'rb') as f:
            return pickle.load(f)