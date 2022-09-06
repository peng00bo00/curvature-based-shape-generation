from typing import Tuple

import igl
import numpy as np
from scipy.spatial import Delaunay

from .util import stereographic, tangentSpace


def delaunay(V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Delaunay triangulation on a (unit) sphere.

    Args:
        V: vertices on (unit) sphere
    
    Return:
        V: vertices
        F: faces from Delaunay triangulation
    """

    ## pick the last vertex from V and rotate the vertices
    ## so that v is the north pole
    v = V[-1]

    R = tangentSpace(v)
    V = V[:-1]
    V = V @ R.T

    ## Stereographic projection to 2D plane
    X, Y = stereographic(V[:,0], V[:,1], V[:,2])
    pts = np.array([X, Y]).T

    ## Delaunay triangulation on the projection plane
    tri = Delaunay(pts)
    F = tri.simplices

    ## invert orientation of faces
    col = F[:, 1].copy()
    F[:, 1] = F[:, 2]
    F[:, 2] = col

    ## close the boundary
    l = igl.boundary_loop(F)
    l = np.append(l, l[0])

    vv= len(V)
    f = []

    for i in range(len(l)-1):
        v1 = l[i]
        v2 = l[i+1]

        f.append((v1, vv, v2))

    f = np.array(f)

    V = np.vstack([V, np.array([[0, 0, 1]])])
    F = np.vstack([F, f.reshape((-1,3))])

    ## rotate back
    V = V @ R

    return V, F