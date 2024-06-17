

# Evaluate Solution
def evaluate_solution(solution, A):
    value = 0
    n = len(A)
    for i in range(n):
        for j in range(i+1, n):
            value += A[i][j] * solution[i] * solution[j]
    return value