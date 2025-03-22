from dataclasses import KW_ONLY, dataclass, field

import networkx as nx
import numpy as np
from scipy.optimize import minimize

from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA


from qiskit_ibm_runtime import Session
from rbnbr.problems.max_cut import MaxCutProblem
from rbnbr.solver.quantum.exp_p1 import p1_expval_maxcut
from rbnbr.solver.solution import Solution
from rbnbr.solver.solver_base import ApproximationSolver
from rbnbr.solver.quantum.resources import QuantumResources

from qiskit_aer import AerSimulator

from qiskit_aer.primitives import SamplerV2, EstimatorV2

from qiskit import transpile


@dataclass
class QAOAMaxCutSolver(ApproximationSolver):
    # Define all parameters with default values
    reps: int = 1
    maxiter: int = 100
    tol: float = 1e-2
    
    # Additional instance variables that aren't constructor parameters
    cache_optim_history: list = field(default_factory=list)
    _: KW_ONLY
    n_shots: int = 1024
    q_dev: object = None
    device_only: bool = False
    
    def __post_init__(self):
        super().__init__()  # Initialize the base class
        
        # Initialize any computed fields
        self.cache_optim_history = []
        if self.q_dev is None:
            self.backend = AerSimulator(method='matrix_product_state')
        else:
            self.backend = self.q_dev.backend

        if self.q_dev is None:
            self.sampler = SamplerV2()
            self.estimator = EstimatorV2()
        else:
            self.sampler = self.q_dev.sampler()
            self.estimator = self.q_dev.estimator()
    
    
    def get_expval(self, problem, optim_params=None):
        if self.reps == 1 and not self.device_only:
            if optim_params is None:
                self.cache_optim_history = []
                optim_params = self._p1_optimize_params(problem.graph)
                
            return float(p1_expval_maxcut(problem.graph, optim_params[0], optim_params[1])), list(optim_params)
        
        else:
            optim_cc, optim_params = self.optim_circuit(problem.graph, optim_params)
            Hc = QAOAMaxCutSolver._maxcut_Hc(problem.graph)
            return float(self._expval(optim_cc, Hc)), list(optim_params)


    def solve(self, problem: MaxCutProblem, optim_params=None):
        
        optim_cc, optim_params = self.optim_circuit(problem.graph, optim_params)
        meas_result = self._measure(optim_cc)
        best_bitstring = self._qaoa_rounding(meas_result)


        breadcrumbs = Solution()
        breadcrumbs.add_step(
            solution=best_bitstring,
            cost=problem.evaluate_solution(best_bitstring),
            elapsed_time=0, # TODO: add timing
            optim_params=optim_params,
            reps=self.reps
        )
        return breadcrumbs

            
            
    def _qaoa_rounding(self, meas_result):
        n_qbits = meas_result.shape[0]
        counts_int = meas_result.get_int_counts()
        shots = sum(counts_int.values())
        final_distribution_int = {key: val/shots for key, val in counts_int.items()}
        
        # print(final_distribution_int)
        # counts_bin = job.result()[0].data.meas.get_counts()
        # final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}
        
        
        keys = list(final_distribution_int.keys())
        values = list(final_distribution_int.values())
        
        
        most_likely = keys[np.argmax(np.abs(values))]
        most_likely_bitstring = QAOAMaxCutSolver.to_bitstring(most_likely, n_qbits)
        most_likely_bitstring.reverse()
        return most_likely_bitstring
    

    def optim_circuit(self, graph, optim_params=None):
        qc, _ = self._qaoa_ansatz(graph)
        # Perform the optimization
        if optim_params is None:
            self.cache_optim_history = []
            optim_params = self._get_optim_params(graph)

                
        # Get the optimized circuit 
        optim_qc = qc.assign_parameters(optim_params)
        
        # qc.draw('mpl')
        # qc.draw('mpl', fold=False, idle_wires=False)
        # optim_qc.draw('mpl', fold=False, idle_wires=False)
        
        return optim_qc, optim_params

    def _qaoa_ansatz(self, graph):
        H_c = QAOAMaxCutSolver._maxcut_Hc(graph)
        ansatz = QAOAAnsatz(cost_operator=H_c, reps=self.reps)
        ansatz.measure_all()
        pm = generate_preset_pass_manager(optimization_level=3, backend=self.backend)
        templ_qc = pm.run(ansatz)
        return templ_qc, H_c
    
    
    
    def _get_optim_params(self, graph):
        if self.reps == 1 and not self.device_only:
            return self._p1_optimize_params(graph)
        else:
            return self._dev_optimize_params(graph)
    
    def _dev_optimize_params(self, graph):
        qc, Hc = self._qaoa_ansatz(graph)
            
        init_params = [np.pi, np.pi / 2] * self.reps
            
        isa_Hc = Hc.apply_layout(qc.layout)
        
        def _cost_func_closure(params, qc, hamiltonian):
            pub = (qc, hamiltonian, params)
            job = self.estimator.run([pub])

            results = job.result()[0]
            cost = results.data.evs

            self.cache_optim_history.append(cost)

            return cost
        
        
        result = minimize(
            _cost_func_closure,
            init_params,
            args=(qc, isa_Hc),
            method="COBYLA",
            tol=self.tol,
            options={'maxiter': self.maxiter}
        )
        
        
        return result.x
    

    def _p1_optimize_params(self, graph):
        init_params = [np.pi, np.pi / 2] * self.reps
        def p1_cost_func_closure(params, graph):
            cost = p1_expval_maxcut(graph, params[0], params[1])
            self.cache_optim_history.append(cost)
            return cost
        
        result = minimize(
            p1_cost_func_closure,
            init_params,
            args=(graph,),
            method="COBYLA",
            tol=self.tol,
            options={'maxiter': self.maxiter}
        )
        
        return result.x

    
        
    def _measure(self, qc):
        pub= (qc, )
        job = self.sampler.run([pub], shots=self.n_shots)
        return job.result()[0].data.meas
    
    def _expval(self, qc, observables):
        pub = (qc, observables)
        job = self.estimator.run([pub])
        return job.result()[0].data.evs
        
    @staticmethod
    def _maxcut_Hc(graph):
        num_nodes = len(graph.nodes)
        pauli_list = []
        for i, j, data in graph.edges(data=True):
            zz_paulis = QAOAMaxCutSolver._pair_paulis_str(i, j, num_nodes)
            weight = data.get('weight', 1)
            pauli_list.append((zz_paulis, weight))
        return SparsePauliOp.from_list(pauli_list)
    
    @staticmethod
    def _pair_paulis_str(i, j, n_nodes):
        paulis = ["I"] * n_nodes
        paulis[i] = "Z"
        paulis[j] = "Z"
        return "".join(paulis)[::-1]
    
    @staticmethod   
    def to_bitstring(integer, num_bits):
        result = np.binary_repr(integer, width=num_bits)
        return [int(digit) for digit in result]