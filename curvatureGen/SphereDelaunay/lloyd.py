import igl
import numpy as np
from scipy.sparse import diags

from typing import Tuple

from .util import VFAdj, VFAdj_density
from .delaunay import delaunay

from ..utils import normalizeSphere

def lloyd(V :np.ndarray, F: np.ndarray, density: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One step Lloyd's relaxation on the sphere.

    Args:
        V: vertices
        F: faces
        density: vertex density
    
    Return:
        V_new: updated vertices
        F_new: updated faces
    """

    ## weighted area
    Adj = VFAdj(V, F)
    area= 0.5 * igl.doublearea(V, F)

    D  = diags(area)
    WA = 1./3 * D @ Adj @ density
    WA = WA.reshape((-1,1))

    ## centroids
    WAdj = VFAdj_density(V, F, density)
    centroid = WAdj @ V * WA

    ## update vertices
    V_new = (Adj.T @ centroid) / (Adj.T @ WA)

    ## normalize the vertices to a unit sphere
    V_new = normalizeSphere(V_new, F)
    V_new, F_new = delaunay(V_new)

    return V_new, F_new