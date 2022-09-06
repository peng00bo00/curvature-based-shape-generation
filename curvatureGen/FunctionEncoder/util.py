import numpy as np
from scipy.spatial.transform import Rotation

from ..utils import normalize, normalizeSphere

def vrrotvec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Find the relative rotation from vector a to vector b.

    Args:
        a: the first direction vector
        b: the second direction vector
    
    Returns:
        r: axis-angle representation of the rotation
    """

    ## rotation axis
    axis = normalize(np.cross(a, b))

    ## rotation angle
    theta= np.arccos(np.dot(a, b)) / ((np.linalg.norm(a) * np.linalg.norm(b)))

    return axis*theta

def vrrotvec2mat(r):
    """Axis-angle to rotation matrix.

    Args:
        r: axis-angle representation
    
    Return:
        R: rotation matrix
    """

    R = Rotation.from_rotvec(r).as_matrix()
    R = R / np.linalg.det(R)
    return R

def WCS2Tangent(R: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Transform a point p from WCS to tangent plane.

    Args:
        R: rotation matrix
        t: translation
        p: point in WCS
    
    Return:
        pp: point in tangent plane
    """

    pp = R.T @ (p - t)

    return pp

def Tangent2WCS(R: np.ndarray, t: np.ndarray, pp: np.ndarray) -> np.ndarray:
    """Transform a point pp from tangent (parameterized) plane to WCS.

    Args:
        R: rotation matrix
        t: translation
        pp: point in tangent (parameterized) plane
    
    Return:
        p: point in WCS
    """

    p = R @ pp + t

    return p