from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo


def qaoa_tsp(tsp, adj_matrix, print_steps=False):
    qp = tsp.to_quadratic_program()

    qp2qubo = QuadraticProgramToQubo()
    qubo = qp2qubo.convert(qp)
    qubitOp, offset = qubo.to_ising()
    
    # exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    # result = exact.solve(qubo)

    ee = NumPyMinimumEigensolver()
    result = ee.compute_minimum_eigenvalue(qubitOp)


    x = tsp.sample_most_likely(result.eigenstate)
    z = tsp.interpret(x)


    if print_steps:
        print(qp.prettyprint())
        print("Offset:", offset)
        print("Ising Hamiltonian:")
        print(str(qubitOp))
        print("energy:", result.eigenvalue.real)
        print("tsp objective:", result.eigenvalue.real + offset)
        print("solution:", z)
        print("feasible:", qubo.is_feasible(x))
        print("solution objective:", tsp.tsp_value(z, adj_matrix))
        
    return z