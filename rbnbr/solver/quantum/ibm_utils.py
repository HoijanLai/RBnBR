from qiskit_ibm_runtime import QiskitRuntimeService

import os

# Read token from me_quantum.txt file
with open(os.path.join(os.path.dirname(__file__), "me_quantum.txt"), "r") as f:
    token = f.read().strip()
try:
    QiskitRuntimeService.save_account(
        token=token,
        channel="ibm_quantum",
    )
except Exception as e:
    print(e)


from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
 
# Create empty circuit
example_circuit = QuantumCircuit(2)
example_circuit.measure_all()
 
# You'll need to specify the credentials when initializing QiskitRuntimeService, if they were not previously saved.
service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)
 
sampler = Sampler(backend)
job = sampler.run([example_circuit])
print(f"job id: {job.job_id()}")
result = job.result()
print(result)