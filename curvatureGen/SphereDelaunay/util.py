import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple

def stereographic(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Stereographic projection.

    Args:
        x: x-coordinates (3D)
        y: y-coordinates (3D)
        z: z-coordinates (3D)
    
    Return:
        X: x-coordinates (2D)
        Y: y-coordinates (2D)
    """

    X = x / (1-z)
    Y = y / (1-z)

    return X, Y

def tangentSpace(z: np.ndarray)-> np.ndarray:
    """Solve the tangent space transform matrix.
    
    Args:
        z: z vector
    
    Return:
        R: transform matrix
    """

    x = np.array([1, 0, 0])
    y = np.cross(z, x)
    x = np.cross(y, z)

    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    z /= np.linalg.norm(z)

    R = np.array([x, y, z])
    R = R / np.linalg.det(R)

    return R

def VFAdj(V: np.ndarray, F: np.ndarray) -> csr_matrix:
    """Build vertex-to-face adjacent matrix.

    Args:
        V: vertices
        F: faces
    
    Return:
        Adj: vertex-to-face adjacent matrix
    """

    # rows = []
    # cols = []

    # for i in range(len(F)):
    #     for v in F[i]:
    #         rows.append(i)
    #         cols.append(v)
    
    # rows = np.array(rows)
    # cols = np.array(cols)

    rows = np.tile(np.arange(len(F)), 3).flatten("F")
    cols = F.flatten("F")

    Adj = csr_matrix((np.ones_like(rows), (rows, cols)), (len(F), len(V)))

    return Adj

def VFAdj_density(V: np.ndarray, F: np.ndarray, density: np.ndarray) -> csr_matrix:
    """Build weighted vertex-to-face adjacent matrix.

    Args:
        V: vertices
        F: faces
        density: vertex density
    
    Return:
        Adj: weighted vertex-to-face adjacent matrix
    """

    ## rows
    rows = np.arange(len(F)).reshape((-1,1))
    rows = np.tile(rows, (1, 3))

    ## cols
    cols = F.copy()

    ## data
    data = density[F].sum(axis=1, keepdims=True)
    data = np.tile(data, (1,3))
    data+= density[F]

    ## normalization
    data = data / (4*density[F].sum(axis=1, keepdims=True))

    rows = rows.flatten()
    cols = cols.flatten()
    data = data.flatten()

    Adj = csr_matrix((data, (rows, cols)), (len(F), len(V)))

    return Adj