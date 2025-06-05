import copy
from dataclasses import dataclass, KW_ONLY
import logging
import numpy as np
from rbnbr.solver.quantum.exp_p1 import p1_Z_maxcut
from rbnbr.solver.quantum.qaoa import QAOAMaxCutSolver 
from rbnbr.solver.solution import Solution


from qiskit.quantum_info import SparsePauliOp
    
from rbnbr.solver.classical import GWMaxcutSolver as GWMaxCutSolver


@dataclass
class QRRMaxCutSolver(QAOAMaxCutSolver):
    _: KW_ONLY
    topk: int = None
    method: str = 'analytic' # 'sampler' and 'analytic' are also available
    X_type: str = 'corr'
    
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
    
    def solve(
        self, 
        problem,
        optim_params=None, 
        select_style='best', # or 'eigenmax' or 'eigenmin'
        index_map=None,
        **kwargs
        ):
        if index_map is None:
            index_map = {node: i for i, node in enumerate(problem.graph.nodes)}
        
        optim_params = self._check_optim_params(optim_params)
        
        best_bitstring, X = self.qrr(problem, optim_params, select_style, index_map)
        
        best_cost = problem.evaluate_solution(best_bitstring)
        breadcrumbs = Solution()
        breadcrumbs.add_step(
            solution=best_bitstring,
            cost=best_cost,
            approx_ratio=problem.approx_ratio(best_bitstring),
            elapsed_time=0, # TODO: add timing
            optim_params=optim_params,
            qrr_information=(best_bitstring, X, self.X_type),
            # qrr_cache=self.cache_info
        )
        return breadcrumbs


    @staticmethod
    def _corr_observables(n_nodes):
        idx_list = []
        observables = []
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                 zz_op = QAOAMaxCutSolver._pair_paulis_str(i, j, n_nodes)
                 idx_list.append((i, j))
                 observables.append(SparsePauliOp(zz_op))
        return idx_list, observables
    
    
    def _corr_estim(self, optim_cc):
        n_qubits = optim_cc.num_qubits
        idx_list, observables = QRRMaxCutSolver._corr_observables(n_qubits)
        results = self._expval(optim_cc, observables)
        
        Z = np.zeros((n_qubits, n_qubits))
        for (i, j), value in zip(idx_list, results):
            Z[i, j] = value
            Z[j, i] = value
        return Z[::-1, ::-1]
    
    
    def _corr_sampler(self, optim_cc):
        
        n_qubits = optim_cc.num_qubits
        meas_result = self._measure(optim_cc)
        counts_bin = meas_result.get_counts()
        
        final_distribution_bin = {key: val/self.n_shots for key, val in counts_bin.items()}
        
        Z = np.zeros((n_qubits, n_qubits))
        for bitstring, prob in final_distribution_bin.items():
            state = [int(bit) for bit in bitstring[::-1]]
            for i in range(n_qubits):
                for j in range(i, n_qubits):
                    zi = 1 - 2 * state[i]  # Convert 0/1 to +1/-1
                    zj = 1 - 2 * state[j]
                    Z[i, j] += ((i==j) - 1) * prob * zi * zj
                    if i != j:
                        Z[j, i] = Z[i, j]
        return Z[::-1, ::-1]
    
    
    def get_corr_analytic(self, graph, optim_params=None):
        if optim_params is None:
            self.cache_optim_history = []
            optim_params = self._p1_optimize_params(graph)
        beta, gamma = optim_params
        return p1_Z_maxcut(graph, beta, gamma)
        
    
    def get_corr_dev(self, optim_cc, method='estimator'):
        if method == 'estimator':
            Z = self._corr_estim(optim_cc)
        elif method == 'sampler':
            Z = self._corr_sampler(optim_cc)

        else:
            raise ValueError(f"Invalid method: {method}")
        
        return Z
    
    
    def qrr(
        self, 
        problem, 
        optim_params=None, 
        select_style='best', 
        approx_style='default',
        return_info=False,
        index_map=None,
        
        **kwargs
        ):
        if index_map is None:
            index_map = {node: i for i, node in enumerate(problem.graph.nodes)}
        
        ## 1. COMPUTE THE CORRELATION MATRIX
        if self.method == 'analytic':
            Z = self.get_corr_analytic(problem.graph, optim_params)
        else: 
            optim_cc, optim_params = super().optim_circuit(problem.graph, optim_params, index_map)
            try:
                Z = self.get_corr_dev(optim_cc, self.method)
            except Exception as e:
                print(f"From QRR: method = `{self.method}` requires a valid device for the calculation of the correlation matrix. ")
                raise e
        
        ## 2. COMPUTE THE EIGENVALUES AND EIGENVECTORS
        evals, evecs = np.linalg.eigh(Z)
        # print(np.sum(eigenvectors - eigenvectors.T))
        evecs = evecs.T
        sign_vecs = (1 + np.sign(evecs)) // 2 # 0 / 1 solution
        
        max_eval = np.max(evals)
        
        ## 3. SELECT THE BEST SOLUTION
        if select_style == 'best':
            # Evaluate all candidates and sort by cost
            candidates = []
            for i in range(sign_vecs.shape[0]):
                sol = sign_vecs[i].astype(int).tolist()
                cost = problem.evaluate_solution(sol)
                relaxed_cost = problem.evaluate_solution(evecs[i, :], relaxed=True)
                candidates.append((cost, sol, evecs[i, :], evals[i], i, relaxed_cost))
                # candidates.append((cost, sol, evecs[i, :], np.abs(evals[i]), i))
            
            # Sort in descending order by cost
            candidates.sort(reverse=True, key=lambda x: x[0])
            # relaxed_candidates = sorted(candidates, key=lambda x: x[5], reverse=True)
            # self.cache_info = {
            #     'abs_rank': np.argsort(np.abs(evals)),
            #     'quality_rank': [x[4] for x in candidates]
            # }
            
            # Get the best solution and correlation matrix
            best_solution = candidates[0][1]
            # best_relaxed_solution = relaxed_candidates[0][3]
            # best_relaxed_cost = relaxed_candidates[0][5]
            solution_tuple = (None, None)
            if self.X_type == 'corr':
                solution_tuple = (best_solution, Z)
            
            elif self.X_type == 'relaxation':
                
                if self.topk is not None:
                    k = max(candidates[0][4], self.topk)
                else:
                    k = candidates[0][4] 
                    
                k += 1 # +1 because it will be used in slicing 

                if approx_style == 'default':
                    # the candidates are sorted by the cost
                    # k is the eigen index of the best solution
                    # so the following select the top k solutions
                    # this is completely empirical 
                    # and this is NOT low-rank approximation
                    topk_items = candidates[:k]
                    topk_evs = np.array([x[2] for x in topk_items])
                    topk_evals = np.array([x[3] for x in topk_items])
                    X_k = topk_evs.T @ np.diag(topk_evals) @ topk_evs
                    
                elif approx_style == 'low-rank':
                    # low-rank approximation
                    idx = np.argsort(evals)[::-1] 
                    evals = evals[idx][:k]                 # sorted eigenvalues
                    evecs = evecs[:, idx][:, :k]
                    X_k = evecs @ np.diag(evals) @ evecs.T
                
                else:
                    raise ValueError(f"Invalid approx_style: {approx_style}")
                
                solution_tuple = (best_solution, X_k)
            
            else: # other X_type 
                raise ValueError(f"Invalid X_type: {self.X_type}")
        
        elif select_style == 'eigenmax':
            # select the sign round solution with the largest absolute eigenvalue
            max_eigenvalue_index = np.argmax(np.abs(evals))
            best_solution = sign_vecs[max_eigenvalue_index].astype(int).tolist()
            solution_tuple = (best_solution, Z)
        
        elif select_style == 'eigenmin':
            # select the sign round solution with the smallest absolute eigenvalue
            min_eigenvalue_index = np.argmin(np.abs(evals))
            best_solution = sign_vecs[min_eigenvalue_index].astype(int).tolist()
            solution_tuple = (best_solution, Z)
                
        else: # other select_style
            raise ValueError(f"Invalid select_style: {select_style}")
        
        
        
        if return_info:
            W = problem.adjacency_matrix
            return solution_tuple[0], solution_tuple[1], {
                'max_eval': max_eval,
                'relaxed_cut': 0.5*np.sum((np.ones_like(Z) - Z ) * W),
                # 'max_relaxed_solution': best_relaxed_solution,
                # 'max_relaxed_cost': best_relaxed_cost
            }
        else:
            return solution_tuple

    
    
    

@dataclass
class QRRnGWMaxCutSolver(QRRMaxCutSolver, GWMaxCutSolver):
    _: KW_ONLY
    X_type: str = 'corr'
    
    def __init__(self, *args, **kwargs):
        QRRMaxCutSolver.__init__(self, *args, **kwargs)
        
    def solve(self, problem, qrr_information=None, *args, **kwargs):
        # RUN QRR
        qrr_information = self._check_qrr_info(qrr_information)
        
        if not qrr_information:
            cut, Z = self.qrr(problem, *args, **kwargs)
        else:
            cut, Z = qrr_information
        
        D = np.diag(np.sum(problem.adjacency_matrix, axis=1))
        L = D - Z
        L_u = GWMaxCutSolver.get_L_u(L)
        X_val = GWMaxCutSolver.solve_sdp(L_u)
        vectors = GWMaxCutSolver.decompose_X(X_val)
        cut = GWMaxCutSolver.random_hyperplane(vectors)
        # collect results
        bc = Solution()
        bc.add_step(
            solution=cut,
            cost=problem.evaluate_solution(cut),  # Fixed: use cut instead of breadcrumbs.solution
            elapsed_time=0,
            approx_ratio=problem.approx_ratio(cut)
        )
        
        return bc
        
    def _check_qrr_info(self, qrr_information):
        if qrr_information is not None:
            if self.X_type != qrr_information[2]:
                msg = f"Ignored: The specified QRR information has X_type {qrr_information[2]}, but the current QRR solver has X_type {self.X_type}"
                logging.warning(msg)
                qrr_information = None
        return qrr_information
            
            
        
        
from rbnbr.solver.quantum.exp_p1 import p1_expval_maxcut 
from scipy.optimize import minimize
            
            
def qrr_on_laplacian(laplacian, X_type='corr', select_style='best', topk=None, approx_style='default', return_info=False):
    beta, gamma = qrr_optimize_params(laplacian)
    Z = p1_Z_maxcut(laplacian, beta, gamma)
        ## 2. COMPUTE THE EIGENVALUES AND EIGENVECTORS
    evals, evecs = np.linalg.eigh(Z)
    # print(np.sum(eigenvectors - eigenvectors.T))
    evecs = evecs.T
    sign_vecs = np.sign(evecs).astype(int)
    bin_vecs = ((1 + np.sign(evecs)) // 2).astype(int) # 0 / 1 solution
    
    max_eval = np.max(evals)
    
## 3. SELECT THE BEST SOLUTION
    # Evaluate all candidates and sort by cost
    candidates = []
    for i in range(sign_vecs.shape[0]):
        sol = sign_vecs[i]
        cost = 0.25 * sol.T@laplacian@sol
        relaxed_cost = 0.25*evecs[i, :].T@laplacian@evecs[i, :]
        candidates.append((cost, bin_vecs[i].tolist(), evecs[i, :], evals[i], i, relaxed_cost))
        # candidates.append((cost, sol, evecs[i, :], np.abs(evals[i]), i))
    
    # Sort in descending order by cost
    candidates.sort(reverse=True, key=lambda x: x[0])
    # relaxed_candidates = sorted(candidates, key=lambda x: x[5], reverse=True)
    # self.cache_info = {
    #     'abs_rank': np.argsort(np.abs(evals)),
    #     'quality_rank': [x[4] for x in candidates]
    # }
    
    # Get the best solution and correlation matrix
    best_solution = candidates[0][1]
    # best_relaxed_solution = relaxed_candidates[0][3]
    # best_relaxed_cost = relaxed_candidates[0][5]
    solution_tuple = (None, None)
    if X_type == 'corr':
        solution_tuple = (best_solution, Z)
    
    elif X_type == 'relaxation':
        
        if topk is not None:
            k = max(candidates[0][4], topk)
        else:
            k = candidates[0][4] 
            
        k += 1 # +1 because it will be used in slicing 

        if approx_style == 'default':
            # the candidates are sorted by the cost
            # k is the eigen index of the best solution
            # so the following select the top k solutions
            # this is completely empirical 
            # and this is NOT low-rank approximation
            topk_items = candidates[:k]
            topk_evs = np.array([x[2] for x in topk_items])
            topk_evals = np.array([x[3] for x in topk_items])
            X_k = topk_evs.T @ np.diag(topk_evals) @ topk_evs
            
        elif approx_style == 'low-rank':
            # low-rank approximation
            idx = np.argsort(evals)[::-1] 
            evals = evals[idx][:k]                 # sorted eigenvalues
            evecs = evecs[:, idx][:, :k]
            X_k = evecs @ np.diag(evals) @ evecs.T
        
        else:
            raise ValueError(f"Invalid approx_style: {approx_style}")
        
        solution_tuple = (best_solution, X_k)
    
    else: # other X_type 
        raise ValueError(f"Invalid X_type: {X_type}")

        
    if return_info:
        W = -laplacian
        np.fill_diagonal(W, 0)
        return solution_tuple[0], solution_tuple[1], {
            'max_eval': max_eval,
            'relaxed_cut': 0.5*np.sum((np.ones_like(Z) - Z) * W),
            'best_cut': candidates[0][0]
            # 'max_relaxed_solution': best_relaxed_solution,
            # 'max_relaxed_cost': best_relaxed_cost
        }
    else:
        return solution_tuple
    
    
    
def qrr_optimize_params(graph, maxiter=100, tol=0.01):
    init_params = [np.pi, np.pi / 2]
    def p1_cost_func_closure(params, graph):
        cost = p1_expval_maxcut(graph, params[0], params[1])
        return cost
    
    result = minimize(
        p1_cost_func_closure,
        init_params,
        args=(graph,),
        method="COBYLA",
        tol=tol,
        options={'maxiter': maxiter}
    )
    
    return result.x