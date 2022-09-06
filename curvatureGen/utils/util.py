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

def Cartesian2Sphere(coords: np.ndarray) -> np.ndarray:
    """A helper function to transform Caresian coordinates to 
    sphere coordinates.

    Args:
        coords: Caresian coordinates (X, Y, Z)
    
    Return:
        sphere_coords: sphere coordinates (rho, theta, phi)
    """

    sphere_coords = np.zeros_like(coords)

    X = coords[:, 0]
    Y = coords[:, 1]
    Z = coords[:, 2]

    # Cartesian to spherical coordinates convertion
    rho = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    theta = np.arccos(np.divide(Z, rho, out=np.zeros_like(rho), where=(rho != 0)))

    phi = np.arctan2(Y, X)
    phi = np.where(phi > 0, phi, phi + 2*np.pi)

    sphere_coords[:, 0] = rho
    sphere_coords[:, 1] = theta
    sphere_coords[:, 2] = phi
    
    return sphere_coords

def Sphere2Cartesian(sphere_coords: np.ndarray) -> np.ndarray:
    """A helper function to transform sphere coordinates to 
    Caresian coordinates.

    Args:
        sphere_coords: sphere coordinates (rho, theta, phi)
    
    Return:
        coords: Caresian coordinates (X, Y, Z)
    """

    coords = np.zeros_like(sphere_coords)

    rho = sphere_coords[:, 0]
    theta = sphere_coords[:, 1]
    phi = sphere_coords[:, 2] 

    # spherical to Cartesian coordinates convertion
    Z = rho * np.cos(theta)

    r = rho * np.sin(theta)
    X = r * np.cos(phi)
    Y = r * np.sin(phi)

    coords[:, 0] = X
    coords[:, 1] = Y
    coords[:, 2] = Z
    
    return coords

def findOrigin(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """A helper function to find the center of mesh.

    Args:
        V: vertices
        F: faces
    
    Return:
        x0: center of the mesh
    """

    weights = 0
    x0 = np.zeros(3)

    for (i, j, k) in F:
        vi, vj, vk = V[i], V[j], V[k]

        ## volume of the tetrahedron
        vol = np.dot(vi, np.cross(vj, vk)) / 6

        weights += vol
        x0 += vol * (vi + vj + vk) / 4
    
    return x0 / weights

def volume(V: np.ndarray, F: np.ndarray) -> float:
    """A helper function to find volume of the mesh.

    Args:
        V: vertices
        F: faces
    
    Return:
        vol: volume of the mesh
    """

    vol = 0.
    for (i, j, k) in F:
        vi, vj, vk = V[i], V[j], V[k]

        ## volume of the tetrahedron
        vol += np.dot(vi, np.cross(vj, vk)) / 6
    
    return vol

def normalizeSphere(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Normalize the mesh to a unit sphere.

    Args:
        V: vertices
        F: faces
    
    Return:
        V_new: normalized vertices
    """

    ## remove translation
    c = findOrigin(V, F)
    V_new = V - c

    ## project vertices to a unit sphere
    scoords = Cartesian2Sphere(V_new)
    scoords[:, 0] = 1
    V_new = Sphere2Cartesian(scoords)

    return V_new