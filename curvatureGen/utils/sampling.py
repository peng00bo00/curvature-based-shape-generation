import numpy as np


def uniform(n: int = 100) -> np.ndarray:
    """Uniform sampling on a unit sphere with Fibonacci Lattices.

    Args:
        n: number of samples
    
    Return:
        V: vertices
    """
    goldenRatio = (1 + 5**0.5)/2

    i = np.arange(0, n, dtype=float) + 0.5

    theta = np.arccos(1 - 2*i/n)
    phi   = 2 * np.pi * i / goldenRatio

    x, y, z = np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)
    V = np.array([x, y, z]).T

    return V

def uniform_plane(n: int = 100) -> np.ndarray:
    """Uniform sampling on the parametrization plane.
    Note that this will not give uniform samples on the sphere.

    Args:
        n: number of samples
    
    Return:
        V: vertices
    """

    n = int(np.sqrt(n))
    phi   = np.linspace(0., 2*np.pi*(1-1/n), n)
    theta = np.linspace(np.pi/n, np.pi*(1-1/n), n)

    phi, theta = np.meshgrid(phi, theta)
    phi   = phi.flatten()
    theta = theta.flatten()

    x, y, z = np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)
    V = np.array([x, y, z]).T

    return V