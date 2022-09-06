import time
import numpy as np
from functools import wraps


def timeit(func):
    """Time cost profiling.
    """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.4f} s')
        return result
    return timeit_wrapper

def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize the vector.

    Args:
        x: a vector
    
    Return:
        norm: normalized vector.
    """

    return x / np.linalg.norm(x)

def ismember(A, B):
    """Find the index of each element of A in B. Assume that the elements are unique in both A and B.
    Steal from:
        https://github.com/erdogant/ismember/blob/277d1e2907abd0bb7dcbcaf4633535bc40bcee6a/ismember/ismember.py#L102
    """

    def is_row_in(a, b):
        # Get the unique row index
        _, rev = np.unique(np.concatenate((b,a)),axis=0,return_inverse=True)
        # Split the index
        a_rev = rev[len(b):]
        b_rev = rev[:len(b)]
        # Return the result:
        return np.isin(a_rev,b_rev)
    
    # Step 1: Find row-wise the elements of a_vec in b_vec
    bool_ind = is_row_in(A, B)
    common = A[bool_ind]

    # Step 2: Find the indices for b_vec
    # In case multiple-similar rows are detected, take only the unique ones
    common_unique, common_inv = np.unique(common, return_inverse=True, axis=0)
    b_unique, b_ind = np.unique(B, return_index=True, axis=0)
    common_ind = b_ind[is_row_in(b_unique, common_unique)]
    
    return bool_ind, common_ind[common_inv]