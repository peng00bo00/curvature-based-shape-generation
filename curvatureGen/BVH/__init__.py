import numpy as np

from .ray import Ray
from .bvh import BVH
from .primitive import Triangle

def buildBVH(V: np.ndarray, F: np.ndarray, nBuckets: int=12) -> BVH:
    """Build BVH with vertices and faces.

    Args:
        V: vertices array
        F: faces array
        nBuckets: number of buckets in BVH
    
    Return:
        BVH: a bvh tree
    """

    triangles = [Triangle(V[F[i]], i) for i in range(len(F))]
    bvh = BVH(triangles, nBuckets)

    return bvh