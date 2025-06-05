import os
import sys

# Resolve the path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the project root to the Python path
sys.path.append(project_root)

# Resolve the path to the third_party directory
third_party_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'third_party'))

# Add the third_party directory to the Python path
if third_party_path not in sys.path:
    sys.path.append(third_party_path)

from rbnbr.problems.max_cut import MaxCutProblem
from rbnbr.solver.solver_base import ApproximationSolver

from rqrao.rqaoa import get_rqaoa_result

from rbnbr.solver.solution import Solution


class RQAOAMaxCutSolver(ApproximationSolver):
    def __init__(self, brute_force_threshold=10, *args, **kwargs):
        self.brute_force_threshold = brute_force_threshold
        super().__init__(*args, **kwargs)
        
    def solve(self, problem: MaxCutProblem, *args, **kwargs):
        edges = []
        edge_weights = []
        
        for edge in problem.graph.edges(data=True):
            edges.append((edge[0], edge[1]))
            edge_weights.append(edge[2].get('weight', 1))
            
        bits, cut, times = get_rqaoa_result(
            edges=edges,
            edge_weights=edge_weights,
            brute_force_threshold=self.brute_force_threshold,
            edge_noise=0.0,
            nb_div=5,
            div_level=3,
            fninp='graphdata_scal.txt',
            fnhyp='hyps_scal.txt',
            fnout='result_scal.txt',
            dir=third_party_path+'/rqrao/'
        )
        solution = [int(bit) for bit in bits]
        cost = problem.evaluate_solution(solution)
        approx_ratio = problem.approx_ratio(solution)
        
        bc = Solution()
        bc.add_step(
            solution=solution,
            cost=cost,
            elapsed_time=0,
            approx_ratio=approx_ratio
        )
        return bc
    
    