# =============================================================================
# SAMPLING: ONE COMPLETE CASE, FROM GEOMETRY TO DATA ON THE LEARNING GRID
#
# The networks work on a fixed Cartesian grid, the same for every case:
#     Nx = 128 points along x,  Ny = 96 points along y
#     spacing  Lx/(Nx-1) = 0.283 mm,   Ly/(Ny-1) = 0.221 mm
#
# For one cavity this module:
#   1. builds the mesh and the SDF on the grid          (geometry.py)
#   2. solves for the temperature on the mesh           (solver.py)
#   3. marks which grid points are solid                (mask)
#   4. reads the FE temperature at every solid grid point
#   5. also keeps the raw mesh data                     (fe_storage.py)
#
# Grid points inside the cavity have no temperature: T = NaN there.
# =============================================================================

# =============================================================================
# VARIABLES (listed once)
#
#   cfg.Nx, cfg.Ny : number of grid points along x and y
#   xgrid, ygrid   : 1D grid coordinates, lengths Nx and Ny          [m]
#   Xg, Yg         : 2D grid coordinates, shape (Ny, Nx)
#                    row j = fixed y,  column i = fixed x
#   grid_points    : all grid points as a list, shape (Ny*Nx, 2), float32
#   shape,p1,p2,p3,cx,cy : cavity description (see geometry.py)
#   mesh, gfu      : NGSolve mesh and FE temperature
#   sdf            : signed distance on the grid, shape (Ny, Nx)     [m]
#   mask           : 1 = solid grid point (sdf >= 0), 0 = cavity
#   T              : FE temperature on the grid, NaN in the cavity   [K]
#   pts, triangles, T_fe : mesh vertices, triangles, vertex temperatures
# =============================================================================

import numpy as np

from .geometry import build_mesh_and_sdf
from .solver import solve_temperature
from .fe_storage import get_mesh_points_and_triangles, evaluate_gfu_on_mesh_vertices


# =============================================================================
# THE LEARNING GRID
#
# Uniform points from 0 to Lx and from 0 to Ly, both ends included.
# meshgrid turns the two 1D lists into 2D arrays:
#     Xg[j, i] = xgrid[i],   Yg[j, i] = ygrid[j]
# grid_points lists the same points row by row (ravel order), which is the
# order used when a (Ny, Nx) field is flattened to length Ny*Nx = 12288.
# =============================================================================
def make_cartesian_grid(cfg):
    xgrid = np.linspace(0.0, cfg.Lx, cfg.Nx)
    ygrid = np.linspace(0.0, cfg.Ly, cfg.Ny)

    Xg, Yg = np.meshgrid(xgrid, ygrid, indexing="xy")

    x_list = Xg.ravel()
    y_list = Yg.ravel()
    grid_points = np.column_stack([x_list, y_list])
    grid_points = grid_points.astype(np.float32)

    return xgrid, ygrid, Xg, Yg, grid_points


# =============================================================================
# ONE CASE: MESH -> SOLVE -> SAMPLE ON THE GRID
#
# The FE solution is evaluated exactly at each solid grid point:
#     mesh(x, y)      finds the triangle containing (x, y)
#     gfu(...)        evaluates the quadratic T_h inside that triangle
# This is the true FE value, not an interpolation between grid points.
# If a point cannot be located in the mesh it is left as NaN; such cases
# are rejected later by data/generate.py.
# =============================================================================
def sample_case(cfg, shape, p1, p2, cx, cy, p3=0.0):
    # STEP 1: GRID, MESH AND SDF
    xgrid, ygrid, Xg, Yg, grid_points = make_cartesian_grid(cfg)
    mesh, sdf = build_mesh_and_sdf(cfg, shape, p1, p2, cx, cy, Xg, Yg, p3=p3)

    # STEP 2: FINITE ELEMENT SOLVE
    gfu = solve_temperature(cfg, mesh)

    # STEP 3: SOLID MASK (WALL POINTS, sdf = 0, COUNT AS SOLID)
    mask = (sdf >= 0.0).astype(np.float32)

    # STEP 4: TEMPERATURE AT EVERY SOLID GRID POINT
    T = np.full((cfg.Ny, cfg.Nx), np.nan, dtype=np.float32)

    for j in range(cfg.Ny):
        for i in range(cfg.Nx):
            if mask[j, i] == 0.0:
                continue

            x = Xg[j, i]
            y = Yg[j, i]

            try:
                T[j, i] = gfu(mesh(x, y))
            except Exception:
                T[j, i] = np.nan

    # STEP 5: RAW MESH DATA (VERTICES, TRIANGLES, VERTEX TEMPERATURES)
    pts, triangles = get_mesh_points_and_triangles(mesh)
    T_fe = evaluate_gfu_on_mesh_vertices(mesh, gfu, pts)

    return mesh, gfu, sdf, mask, T, pts, triangles, T_fe
