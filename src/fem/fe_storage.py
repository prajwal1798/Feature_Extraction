# =============================================================================
# FE STORAGE: TAKE THE MESH AND TEMPERATURE OUT OF NGSOLVE
#
# The networks never see NGSolve objects. This module turns one solved case
# into plain NumPy arrays (the file itself is written by data/generate.py):
#
#   points      (n_vertices, 2)  x, y of every mesh vertex             [m]
#   triangles   (n_triangles, 3) the 3 vertex numbers of each triangle
#   temperature (n_vertices,)    T at every vertex                     [K]
#   shape_code, p1, p2, cx, cy   the geometry that produced this case
#
# The stored temperature is the vertex value of the order-2 FE solution.
# The extra mid-edge unknowns of the quadratic elements are NOT stored, so
# a plot made from this file is the piecewise-linear picture of T_h.
# =============================================================================

# =============================================================================
# VARIABLES (listed once)
#
#   mesh         : NGSolve mesh from geometry.build_mesh
#   temperature  : NGSolve GridFunction from solver.solve_temperature
#   n_vertices   : number of mesh vertices        (mesh.nv)
#   pts          : vertex coordinates, shape (n_vertices, 2)
#   triangles    : vertex numbers per triangle, shape (n_triangles, 3),
#                  numbered from 0 so they index straight into pts
#   T_fe         : temperature at the vertices, shape (n_vertices,)
# =============================================================================

import numpy as np
from ngsolve import VOL


# =============================================================================
# MESH VERTICES AND TRIANGLES AS NUMPY ARRAYS
#
# NGSolve numbers its vertices 0, 1, 2, ...; each triangle lists the numbers
# of its 3 corners in that same numbering.
# =============================================================================
def get_mesh_points_and_triangles(mesh):
    # VERTEX COORDINATES
    pts = []
    for vertex in mesh.vertices:
        x, y = vertex.point
        pts.append([x, y])
    pts = np.array(pts, dtype=float)

    # TRIANGLE CONNECTIVITY
    triangles = []
    for element in mesh.Elements(VOL):
        corner_ids = []
        for corner in element.vertices:
            corner_ids.append(corner.nr)
        triangles.append(corner_ids)
    triangles = np.array(triangles, dtype=int)

    return pts, triangles


# =============================================================================
# TEMPERATURE AT THE MESH VERTICES
#
# In an NGSolve H1 space the first n_vertices unknowns are the values at the
# vertices, in vertex order. Higher-order unknowns (mid-edge) come after them.
# So the vertex temperatures are simply the first n_vertices entries.
# (pts is kept as an argument so the call in sampling.py stays unchanged.)
# =============================================================================
def evaluate_gfu_on_mesh_vertices(mesh, temperature, pts):
    n_vertices = mesh.nv

    all_values = np.array(temperature.vec)
    T_fe = all_values[:n_vertices]

    return T_fe
