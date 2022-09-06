import numpy as np

from .ray import Ray

class BaseBound:
    """Base bound interface.
    """

    def __repr__(self) -> str:
        return "Base bound"
    
    def diag(self) -> np.ndarray:
        """Diagnoal of the bounding box.
        """
        raise NotImplementedError
    
    def inside(self, p: np.ndarray) -> bool:
        """Whether a point p is inside the box.
        """
        raise NotImplementedError
    
    def maximumExtent(self) -> int:
        """Find the axis with maximum extent.
        """
        raise NotImplementedError
    
    def centroid(self) -> np.ndarray:
        """Compute centroid of the bounding box.
        """
        raise NotImplementedError
    
    def offset(self, p: np.ndarray) -> np.ndarray:
        """Relative offset of p from the box.
        """
        raise NotImplementedError
    
    def surfaceArea(self) -> float:
        """Surface area of the box.
        """
        raise NotImplementedError
    
    def intersect(self, ray: Ray) -> bool:
        """Intersection with a given ray.

        Args:
            ray: a ray
        
        Return:
            hit: whether the ray hits the bounding box
        """
        raise NotImplementedError


class Bound3D(BaseBound):
    """3D bounding box implementation.
    """

    def __init__(self, p1: np.ndarray, p2: np.ndarray):
        self.pMin = np.minimum(p1, p2)
        self.pMax = np.maximum(p1, p2)
    
    def __repr__(self) -> str:
        return f"pMin: {self.pMin}, pMax: {self.pMax}"
    
    def diag(self) -> np.ndarray:
        """Diagnoal of the bounding box.
        """
        return self.pMax - self.pMin
    
    def inside(self, p: np.ndarray) -> bool:
        """Whether a point p is inside the box.
        """
        x, y, z = p
        return self.pMin[0] <= x <= self.pMax[0] \
               and self.pMin[1] <= y <= self.pMax[1] \
               and self.pMin[2] <= z <= self.pMax[2]
    
    def maximumExtent(self) -> int:
        """Find the axis with maximum extent.
        """
        d = self.diag()
        return np.argmax(d)
    
    def centroid(self) -> np.ndarray:
        """Compute centroid of the bounding box.
        """
        return 0.5*(self.pMin+self.pMax)
    
    def offset(self, p: np.ndarray) -> np.ndarray:
        """Relative offset of p from the box.
        """

        o = p - self.pMin

        for i in range(3):
            if self.pMax[i] > self.pMin[i]:
                o[i] /= (self.pMax[i] - self.pMin[i])

        return o
    
    def surfaceArea(self) -> float:
        """Surface area of the box.
        """
        d = self.diag()
        return 2*(d[0]*d[1] + d[1]*d[2] + d[2]*d[0])
    
    def intersect(self, ray: Ray) -> bool:
        """Intersection with a given ray.

        Args:
            ray: a ray
        
        Return:
            hit: whether the ray hits the bounding box
        """
        t0 = 0.
        t1 = ray.tMax

        for i in range(3):
            invRayDir = 1 / (ray.dir[i]+1e-7)

            tNear= (self.pMin[i] - ray.origin[i]) * invRayDir
            tFar = (self.pMax[i] - ray.origin[i]) * invRayDir

            if tNear > tFar:
                tNear, tFar = tFar, tNear

            t0 = max(t0, tNear)
            t1 = min(t1, tFar)
            
            if t0 > t1:
                return False

        return True


class EmptyBound3D(BaseBound):
    """Empty 3D bounding box implementation.
    """

    def __init__(self):
        self.pMin = np.array([np.Infinity, np.Infinity,np.Infinity])
        self.pMax =-np.array([np.Infinity, np.Infinity,np.Infinity])
    
    def __repr__(self) -> str:
        return "Empty bounding box"
    
    def inside(self, p: np.ndarray) -> bool:
        """Whether a point p is inside the box.
        """
        return False
    
    def surfaceArea(self) -> float:
        """Surface area of the box.
        """
        return 0.
    
    def intersect(self, ray: Ray) -> bool:
        """Intersection with a given ray.

        Args:
            ray: a ray
        
        Return:
            hit: whether the ray hits the bounding box
        """
        return False


def unionBound(b1: BaseBound, b2: BaseBound) -> BaseBound:
    """Union of two bounding box.

    Args:
        b1: the first bounding box
        b1: the second bounding box
    
    Return:
        box: the union of b1 and b2
    """

    ## check if b1 or b2 is empty
    if isinstance(b1, EmptyBound3D):
        return b2
    elif isinstance(b2, EmptyBound3D):
        return b1
    
    p1 = np.minimum(b1.pMin, b2.pMin)
    p2 = np.maximum(b1.pMax, b2.pMax)

    box = Bound3D(p1, p2)

    return box

def pointBound(p: np.ndarray) -> Bound3D:
    """Create a bounding box with one point.

    Args:
        p: point coordinate
    
    Return:
        box: bounding box
    """

    return Bound3D(p, p)