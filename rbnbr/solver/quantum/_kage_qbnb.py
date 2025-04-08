from dataclasses import dataclass, field, KW_ONLY
import numpy as np
from rbnbr.solver.quantum.qrr import QRRMaxCutSolver
from rbnbr.solver.quantum.qbnb import QRR_BnB_MC_V2

@dataclass
class QRR_BnB_MC_V4(QRR_BnB_MC_V2):
    
    """
    This is a bound by maximum eigenvalue method
    """
    _: KW_ONLY

    def get_bound(self, sub_uf, problem, index_map, **other_info):
        
        z = None
        X = None
        
        bound = np.inf
        
        bound_info = {}
        
        # Check if this is the root problem (no variables fixed yet)
        if self.qrr_info is not None and sub_uf.is_root:
            # If we have pre-computed QRR information for the root problem, use it
            z, X, _ = self.qrr_info

        elif problem.ref_solution_arr is not None:
            z = problem.ref_solution_arr
            X = None
            bound = problem.ref_solution_cost
            
        else:
            # Otherwise, solve the subproblem with QRR
            if sub_uf.is_root:
                # For root problem, use provided optimization parameters
                z_sub, X, max_eval = \
                    QRRMaxCutSolver.qrr(self, problem, self.opt_params, approx_style=self.approx_style, return_max_eval=True)
            else:
                # For subproblems, use default parameters
                z_sub, X, max_eval = \
                    QRRMaxCutSolver.qrr(self, problem, approx_style=self.approx_style, return_max_eval=True)
                       
            bound = max_eval * 0.25 * sub_uf.n_rep
            print(f"bound: {max_eval}")
            z = sub_uf.get_original_solution(z_sub)
            
        cut = self.cache_root_problem.evaluate_solution(z)
        bound_info = {
            'X': X, 
            'z': z,
            'cut': cut
        }
        
        return bound, bound_info
    

    def solve_subproblem(self, sub_uf, z=None, **kwargs):
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