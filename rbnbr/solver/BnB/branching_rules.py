import numpy as np
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

def branch_hardest(correlations, top_one=True, **kwargs):
    # pick the pair with the highest correlation using numpy
    if top_one:
        # Find the indices of the maximum absolute correlation value
        flat_idx = np.argmax(np.abs(correlations.flatten()))
        # Return only the top pair with highest absolute correlation
        return [np.unravel_index(flat_idx, correlations.shape)]
    else:
        # Get the indices of all pairs sorted by absolute correlation value (highest to lowest)
        flat_indices = np.argsort(np.abs(correlations.flatten()))[::-1]  # Sort in descending order
        # Return all pairs sorted by absolute correlation (highest to lowest)
        sorted_pairs = [np.unravel_index(idx, correlations.shape) for idx in flat_indices]
        return sorted_pairs

def branch_confidence(correlations, top_one=False, normalized=False, **kwargs):
    
    """
    Pick the pair of variables with the highest confidence (closest to -1 or 1).
    This means selecting the pair that minimizes the sum of (1 - |x_ik|)^2 across all k.
    """
    n = correlations.shape[0]
    
    
    # Normalize the correlation matrix to ensure values are between -1 and 1
    # This handles cases where the correlation values might be outside the expected range
    if normalized:
        logger.debug("Normalizing the correlation matrix")
        abs_max = np.max(np.abs(correlations))
        if abs_max > 0:  # Avoid division by zero
            correlations = correlations / abs_max

    pairs = []
    pair_sums= []
    
    # Calculate confidence for each pair (i,j)
    for i in range(n):
        for j in range(i+1, n):  # Only consider unique pairs
            # Calculate the sum of (1 - |x_ik|)^2 for this pair
            confidence_sum = np.sum((1 - np.abs(correlations[i, :]))**2) + np.sum((1 - np.abs(correlations[j, :]))**2)
            
            # Update if this pair has higher confidence (lower sum)
            pair_sums.append(confidence_sum)
            pairs.append([i, j])
    
    pairs = np.array(pairs)
    
    
    if top_one:
        # Return only the top pair with highest confidence (lowest sum)
        return [pairs[np.argmin(pair_sums)]]
    else:
        # Return all pairs sorted by confidence
        pairs = pairs[np.argsort(pair_sums), :]
        return pairs.tolist()
    
def branch_easiest(correlations, top_one=True, **kwargs):
    # pick the pair with the lowest correlation using numpy
    if top_one:
        # Find the indices of the minimum absolute correlation value
        flat_idx = np.argmin(np.abs(correlations.flatten()))
        # Return only the top pair with highest absolute correlation
        return [np.unravel_index(flat_idx, correlations.shape)]
    else:
        # Get the indices of all pairs sorted by absolute correlation value (highest to lowest)
        flat_indices = np.argsort(np.abs(correlations.flatten()))  # Sort in ascending order
        # Return all pairs sorted by absolute correlation (highest to lowest)
        sorted_pairs = [np.unravel_index(idx, correlations.shape) for idx in flat_indices]
        return sorted_pairs

