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

from rbnbr.solver.solver_base import ApproximationSolver
from rbnbr.solver.solution import Solution
from rbnbr.problems.max_cut import MaxCutProblem


class QRBnBMaxCutSolver(ApproximationSolver):
    def __init__(self, rounding_scheme='semideterministic'):
        super().__init__()
        self.rounding_scheme = rounding_scheme

    def solve(self, problem: MaxCutProblem, *args, **kwargs):
        x_opt, fval = self._run_qr_bnb(problem.graph)
        
        breadcrumbs = Solution()
        breadcrumbs.add_step(
            solution=x_opt,
            cost=fval,
            elapsed_time=0,  # TODO: Add timing
            approx_ratio=problem.approx_ratio(x_opt)
        )
        return breadcrumbs

    def _solve_with_relaxation(self, problem):
        # Create an encoding object with a maximum of 3 variables per qubit
        encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=3)
        encoding.encode(problem)
        
        ansatz = RealAmplitudes(2)
        vqe = VQE(
            ansatz=ansatz,
            optimizer=COBYLA(),
            estimator=Estimator(), 
        )
        
        if self.rounding_scheme == 'semideterministic':
            rounding = SemideterministicRounding()
        elif self.rounding_scheme == 'magical_rounding':
            sampler = Sampler(options={"shots": 10000, "seed": 10086})
            rounding = MagicRounding(sampler=sampler)
        else:
            raise ValueError(f"Invalid rounding scheme: {self.rounding_scheme}")

        qrao = QuantumRandomAccessOptimizer(min_eigen_solver=vqe, rounding_scheme=rounding)
        results = qrao.solve(problem)
        return results.x, results.fval

    def _branch(self, prob, j):
        """Branch by fixing variable j to 0 or 1 and updating the objective accordingly"""
        prob_0 = prob.substitute_variables({f'x_{j}': 0})
        prob_1 = prob.substitute_variables({f'x_{j}': 1})
        return [prob_0, prob_1]

    def _select_variable(self, graph, history_idx):
        """Select variable with highest degree that hasn't been branched on yet"""
        degrees = dict(graph.degree())
        available_nodes = [node for node in degrees.keys() if node not in history_idx]
        
        if not available_nodes:
            return None
            
        return max(available_nodes, key=lambda x: degrees[x])

    def _run_qr_bnb(self, graph):
        maxcut = Maxcut(graph)
        prob = maxcut.to_quadratic_program()
        
        prob_set = [prob]
        fval_up = float('inf')
        history_idx = []
        x_opt = None
        
        while prob_set:
            prob = prob_set.pop()
            x, fval = self._solve_with_relaxation(prob)
            
            if fval < fval_up:
                fval_up = fval
                x_opt = x
            else:
                continue
                
            idx = self._select_variable(graph, history_idx)
            history_idx.append(idx)
            
            maxcut = Maxcut(graph)
            prob = maxcut.to_quadratic_program()
            prob_0, prob_1 = self._branch(prob, idx)
            prob_set.append(prob_0)
            prob_set.append(prob_1)
            
        return x_opt, fval_up
