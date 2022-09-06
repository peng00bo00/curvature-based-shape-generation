import numpy as np
from typing import Tuple

from .ray import Ray
from .util import normalize
from .bound import Bound3D

class Triangle:
    """Triangle implementation.
    """

    def __init__(self, v: np.ndarray, faceIdx: int):
        self.v       = v
        self.faceIdx = faceIdx
    
    def __repr__(self) -> str:
        v0, v1, v2 = self.v
        return f"v0: {v0}, v1: {v1}, v2: {v2}, faceIdx: {self.faceIdx}"
    
    def bound(self) -> Bound3D:
        """Get bounding box of the triangle in WCS.
        """

        p1 = np.min(self.v, axis=0)
        p2 = np.max(self.v, axis=0)
        
        return Bound3D(p1, p2)
    
    def centroid(self) -> np.ndarray:
        """Get centroid of the triangle in WCS.
        """

        return np.mean(self.v, axis=0)
    
    def intersect(self, ray: Ray) -> Tuple[bool, float]:
        """Intersection with a given ray.

        Args:
            ray: a ray
        
        Return:
            hit: whether the ray hits the bounding box
            tHit: time of hit
        """
        
        ## find triangle plane
        v0, v1, v2 = self.v
        e1 = v1 - v0
        e2 = v2 - v0

        ## normal of the plane
        N = normalize(np.cross(e1, e2))

        ## check if the ray direction is parallel to the normal
        NdotDir = np.dot(N, ray.dir)
        if np.isclose(NdotDir, 0.):
            return False, 0.
        
        ## find the intersection P
        t = np.dot(N, v0-ray.origin) / NdotDir

        ## t must be positive
        if t < 0:
            return False, 0.

        ## outside-inside test
        P = ray(t)
        ## check e0
        e0 = v1 - v0
        vp0= P - v0
        C  = np.cross(e0, vp0)
        if np.dot(N, C) < 0:
            return False, 0.

        ## check e1
        e1 = v2 - v1
        vp1= P - v1
        C  = np.cross(e1, vp1)
        if np.dot(N, C) < 0:
            return False, 0.

        ## check e2
        e2 = v0 - v2
        vp2= P - v2
        C  = np.cross(e2, vp2)
        if np.dot(N, C) < 0:
            return False, 0.

        return True, t