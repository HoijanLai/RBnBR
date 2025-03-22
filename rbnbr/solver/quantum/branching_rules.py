import numpy as np

def branch_hardest(correlations):
    # pick the pair with the highest correlation using numpy
    return np.unravel_index(np.argmax(np.abs(correlations)), correlations.shape)


def branch_confidence(correlations):
    
    """
    Pick the pair of variables with the highest confidence (closest to -1 or 1).
    This means selecting the pair that minimizes the sum of (1 - |x_ik|)^2 across all k.
    """
    n = correlations.shape[0]
    min_sum = float('inf')
    min_pair = (0, 0)
    
    # Calculate confidence for each pair (i,j)
    for i in range(n):
        for j in range(i+1, n):  # Only consider unique pairs
            # Calculate the sum of (1 - |x_ik|)^2 for this pair
            confidence_sum = np.sum((1 - np.abs(correlations[i, :]))**2) + np.sum((1 - np.abs(correlations[j, :]))**2)
            
            # Update if this pair has higher confidence (lower sum)
            if confidence_sum < min_sum:
                min_sum = confidence_sum
                min_pair = (i, j)
    
    return min_pair
    

def branch_easiest(correlations):
    # pick the pair with the lowest correlation using numpy
    return np.unravel_index(np.argmin(np.abs(correlations)), correlations.shape)


