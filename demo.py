import os
import numpy as np
import igl

from curvatureGen import Mesh, FunctionEncoder

## load mesh
V, F = igl.read_triangle_mesh("./mesh/spot.obj")
mesh = Mesh(V, F)
rho  = mesh.face_curvature / np.sqrt(mesh.face_area)

## encode mesh
encoder = FunctionEncoder()
V, F = igl.read_triangle_mesh("./mesh/spot_sphere.obj")
values = encoder.encode(V, F, rho)

np.save("spot_rho.npy", values)