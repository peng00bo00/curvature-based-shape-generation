import numpy as np
from scipy import interpolate

from .util import vrrotvec, vrrotvec2mat, WCS2Tangent, Tangent2WCS
from .RefSphere import RefSphere

from ..BVH import buildBVH, Ray, BVH
from ..utils import normalize, normalizeSphere


def createValueFunc(bvh: BVH, rho: np.ndarray):
    """A helper function to wrap ray casting.

    Args:
        bvh: a BVH instance
        rho: value function on faces
    
    Return:
        valueFunc: wrapped query function
    """
    def valueFunc(ray: Ray) -> float:
        _, _, idx = bvh.intersect(ray)
        faceIdx = bvh.triangles[idx].faceIdx
        return rho[faceIdx]
    
    return valueFunc


class FunctionEncoder:
    """Function encoder on the sphere.

    Args:
        lod: level of detail (used in reference sphere)
    """

    def __init__(self, lod=3):
        ## reference sphere
        self.ref = RefSphere(lod)

    def encode(self, V: np.ndarray, F: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Encode a function with reference sphere.

        Args:
            V: vertices
            F: faces
            value: value function defined on the faces
        
        Return:
            valuesRef: value function defined on the reference sphere
        """
        
        assert len(values) == len(F), "Invalid value function!"

        ## create BVH with reference sphere
        V   = normalizeSphere(V, F)
        bvh = buildBVH(V, F)
        valueFunc = createValueFunc(bvh, values)

        ## retrieve from reference sphere
        Vsph, Fsph = self.ref.V, self.ref.F
        Rs = self.ref.Rs
        Ts = self.ref.Ts

        ## ray casting from ref sphere
        grid   = self.ref.grid
        points = grid.reshape((-1, 3))
        valuesRef = np.zeros((grid.shape[0], grid.shape[1], len(Fsph)))

        ## loop all the ref faces
        for i in range(len(Fsph)):
            print(f"Working on {i}th patch...")
            ## recover WCS coordinates
            ## this is the direction of each ray
            pointsWCS = np.sum(points[..., None, :] * Rs[i], axis=-1) + Ts[i]

            ## cast rays to find value vi
            vi = np.array([valueFunc(Ray(np.zeros(3), d)) for d in pointsWCS])
            valuesRef[:,:,i] = vi.reshape((grid.shape[0], grid.shape[1]))
        
        return valuesRef

    def query(self, V: np.ndarray, F: np.ndarray, valuesRef: np.ndarray) -> np.ndarray:
        """Query the value function.

        Args:
            V: vertices
            F: faces
            valuesRef: value function defined on the reference sphere
        
        Return:
            values: value function defined on the faces
        """
        
        assert len(valuesRef) == len(self.ref.F), "Invalid value function!"

        ## wrapping the valuesRef function
        xs, ys = self.ref.xs, self.ref.ys
        funcs = [interpolate.RectBivariateSpline(xs, ys, valuesRef[:,:,i], kx=1, ky=1) \
                 for i in range(valuesRef.shape[2])]

        ## relative rotation and translation
        Rs = self.ref.Rs
        Ts = self.ref.Ts

        ## initialize ray directions
        barycenter = V[F,:].mean(axis=1)
        values = np.zeros(len(F))

        ## ray casting 
        rays = [Ray(np.zeros(3), d) for d in barycenter]
        for i, ray in enumerate(rays):
            _, t, idx = self.ref.bvh.intersect(ray)
            faceIdx = self.ref.bvh.triangles[idx].faceIdx

            ## retrieve tangent plane coordinates
            R = Rs[faceIdx]
            T = Ts[faceIdx]

            tx, ty, tz = WCS2Tangent(R, T, ray(t))

            ## interpolation on tangent plane
            values[i] = funcs[faceIdx](ty, tx)
        
        return values