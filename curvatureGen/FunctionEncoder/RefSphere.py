from typing import List, Tuple
import numpy as np
import igl

from .util import vrrotvec, vrrotvec2mat

from ..utils import ismember, normalize, normalizeSphere, uniform
from ..BVH import buildBVH
from ..SphereDelaunay import delaunay


def subdivision(V: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Edge subdivision of a sphere mesh.

    Args:
        V: vertices
        F: faces

    Returns:
        V_new: vertices of the new mesh
        F_new: faces of the new mesh. Every 4 row belong to the previous face.
    """

    nV = V.shape[0]

    ## find midpoints on each edge
    E = igl.edges(F)
    mids = 0.5*(V[E[:, 0], :] + V[E[:, 1], :])

    V_new = np.concatenate([V, mids], axis=0)

    ## find index of the midpoints
    Eij = np.sort(F[:, [0, 1]], axis=1)
    Ejk = np.sort(F[:, [1, 2]], axis=1)
    Eki = np.sort(F[:, [2, 0]], axis=1)

    Vi = F[:, 0]
    Vj = F[:, 1]
    Vk = F[:, 2]

    _, Vij = ismember(Eij, E)
    _, Vjk = ismember(Ejk, E)
    _, Vki = ismember(Eki, E)

    Vij += nV
    Vjk += nV
    Vki += nV

    ## build 4 new faces on the original face
    F_new = np.concatenate([
                            np.stack([Vij, Vjk, Vki]).T,
                            np.stack([Vi, Vij, Vki]).T,
                            np.stack([Vij, Vj, Vjk]).T,
                            np.stack([Vki, Vjk, Vk]).T,
                            ], axis=1)
    
    F_new = F_new.reshape((-1, 3))

    ## project vertices to unit sphere
    V_new = normalizeSphere(V_new, F_new)

    return V_new, F_new


class RefSphere:
    scale = [0.8, 0.4, 0.24, 0.12, 0.08]
    """Reference sphere implementation.

    Args:
        lod: level of detail
    """

    def __init__(self, lod: int = 3):
        self.LOD = lod
        self.V, self.F = self._build_sphere()

        ## build BVH
        self.bvh = buildBVH(self.V. self.F)
    
    def _build_sphere(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build the sphere.
        """

        ## LOD=0, this sphere has 10 vertices and 16 faces
        V = uniform(10)
        V, F = delaunay(V)

        ## build the sphere recursively
        ## every time it will have 4 times more faces
        for _ in range(self.LOD):
            V, F = subdivision(V, F)
        
        return V, F
    
    @property
    def grid(self) -> np.ndarray:
        """Create a grid (tangent plane coordinate system on each face) with LOD.
        """

        ## the grid size is set to [32, 32]
        xs = self.xs
        ys = self.ys

        meshX, meshY = np.meshgrid(xs, ys)
        grid = np.stack([meshX, meshY, np.zeros_like(meshX)])
        grid = grid.transpose([1, 2, 0])

        return grid
    
    @property
    def xs(self) -> np.ndarray:
        return (np.arange(32) - 16) / 16 * self.scale[self.LOD]
    
    @property
    def ys(self) -> np.ndarray:
        return (np.arange(32) - 16) / 16 * self.scale[self.LOD]

    @property
    def Rs(self) -> List[np.ndarray]:
        """Relative rotation on each face.
        """
        Vsph, Fsph = self.V, self.F

        Vi = Vsph[Fsph[:, 0]]
        Vj = Vsph[Fsph[:, 1]]
        Vk = Vsph[Fsph[:, 2]]

        Ns = np.cross(Vj-Vi, Vk-Vi)
        Ns = np.array([normalize(n) for n in Ns])

        z = np.array([0,0,1])
        Rs= np.array([vrrotvec2mat(vrrotvec(z, n)) for n in Ns])

        return Rs

    @property
    def Ts(self) -> List[np.ndarray]:
        """Relative rotation on each face.
        """
        Vsph, Fsph = self.V, self.F
        barycenter = Vsph[Fsph,:].mean(axis=1)

        return barycenter