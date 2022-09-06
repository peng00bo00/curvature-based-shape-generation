import numpy as np

from .util import normalize

class Ray:
    """Ray implementation.
    """

    def __init__(self, origin:np.ndarray = np.zeros(3), dir:np.ndarray = np.array([0,0,1]),
                 tMax:float = float("inf"), t:float = 0.):
        self.origin = origin
        self.dir = normalize(dir)
        
        self.tMax= tMax
        self.t   = t
    
    def __repr__(self) -> str:
        return f"Origin: {self.origin}, Direction: {self.dir}, tMax: {self.tMax}, t: {self.t}"
    
    def __call__(self, t) -> np.ndarray:
        """Cast ray with given time.
        """
        return self.origin + t*self.dir