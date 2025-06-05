import copy
from typing import Callable

import numpy as np

from rbnbr.utils.const import ArrayLike

class Solution:
    """
    A single step in the solution process
    """
    
    @property
    def data(self):
        # get the last step as dictionary
        self._best_step['first_seen'] = self.first_sight_optimal()
        self._best_step['total_steps'] = self.total_step()
        result = copy.deepcopy(self._best_step)
        return result
    
    @property
    def cost(self):
        return self._best_step['cost']
    
    @property
    def z(self):
        return self._best_step['solution']
    
    @property
    def approx_ratio(self):
        return self._best_step['approx_ratio']
    
    
    def __init__(
        self, 
        cost_fn: Callable = None,
    ):
        self._cost_fn = cost_fn
        self._step_template = {
            'solution': None,
            'cost': None,
            'time': None,
            'approx_ratio': None
        }
        self._breadcrumbs = []
        
        self._best_step = {
            'cost': None,
            'solution': None,
            'time': None,
            'approx_ratio': None,
            'n_steps': None,
            'first_seen': None
        }
        
        
    def add_step(
        self, 
        solution: ArrayLike, 
        cost: float, 
        elapsed_time: float,
        approx_ratio: float = None,
        **kwargs
    ):
        step = self._step_template.copy()
        step['solution'] = solution
        step['cost'] = cost
        step['time'] = elapsed_time
        step['approx_ratio'] = approx_ratio
        for k, v in kwargs.items():
            step[k] = v
        self._breadcrumbs.append(step)
        
        if self._best_step['cost'] is None or cost >= self._best_step['cost']:
            self._best_step['cost'] = cost
            self._best_step['solution'] = step['solution']
            self._best_step['time'] = elapsed_time
            self._best_step['approx_ratio'] = approx_ratio
            for k, v in kwargs.items():
                self._best_step[k] = v
        
        
        return self
    
    def total_step(self):  
        n_step = 0
        for step in self._breadcrumbs:
            if 'n_steps' in step and step['n_steps'] > n_step:
                n_step = step['n_steps']
        return n_step
    
    def first_sight_optimal(self):
        n_step = np.inf
        for step in self._breadcrumbs:
            if 'n_steps' in step:
                if step['cost'] == self._best_step['cost'] and step['n_steps'] < n_step:
                    n_step = step['n_steps']
        return n_step
        
    
    def __getitem__(self, index):
        return self._breadcrumbs[index]
    
    
    def __str__(self):
        return f"Solution: {self._bitstring(self.z)} \nSolution Value: {self.cost} \nApproximation Ratio: {self.approx_ratio}"
    
    def __len__(self):
        return len(self._breadcrumbs)
    
    def get_history(self, *keys):
        import pandas as pd
        
        history = {}
        for key in keys:
            history[key] = [step[key] for step in self._breadcrumbs]
            
        return pd.DataFrame(history)
    
    
    def _convert_solution(self, solution: ArrayLike):
        solution = np.array(solution)
        unique_vals = np.unique(solution)
        if set(unique_vals).issubset({0, 1}):
            # Convert 0/1 encoding to -1/1 encoding
            return 2 * solution - 1
        elif set(unique_vals).issubset({-1, 1}):
            return solution
        else:
            raise ValueError("Solution must be either -1/1 or 0/1 encoded")


    def _bitstring(self, solution: ArrayLike):
        bs = ''.join(str(int(x > 0)) for x in solution)
        return bs