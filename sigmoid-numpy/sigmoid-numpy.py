import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    arr = np.asarray(x, dtype=float)
    result = 1 / (1 + np.exp(-arr))
    
    # Return a standard Python float if the input was a scalar
    if np.isscalar(x):
        return float(result)
        
    return result