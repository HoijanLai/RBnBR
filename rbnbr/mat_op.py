
import numpy as np
from .utils import evaluate_solution


# Relax-and-Round
def relax_and_round(C):
    A = ...
    eigvals, eigvecs = np.linalg.eigh(C)
    best_solution = None
    best_value = float('inf')
    for vec in eigvecs.T:
        rounded_solution = np.sign(vec)
        value = evaluate_solution(rounded_solution, A)
        if value < best_value:
            best_value = value
            best_solution = rounded_solution
    return best_solution