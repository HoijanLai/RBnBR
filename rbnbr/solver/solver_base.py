from typing import Callable
import numpy as np
from abc import ABC, abstractmethod
from rbnbr.problems.problem_base import CombProblemBase

class SolverBase(ABC):
    """
    Simply designed for collecting breadcrumbs
    """
    
    def __init__(self):
        self._solution_quality = None
        
        
    def solve(
        self, 
        problem: CombProblemBase
    ):
        breadcrumbs = None
        return breadcrumbs
    
    @property
    def solution_quality(self):
        return self._solution_quality
        
        
class ExactSolver(SolverBase):
    def __init__(self):
        super().__init__()
        self._solution_quality = 'exact'
        
    

class ApproximationSolver(SolverBase):
    def __init__(self):
        super().__init__()
        self._solution_quality = 'approximate'
        

