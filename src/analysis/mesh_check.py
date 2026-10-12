# =============================================================================
# MESH CHECK: IS THE FE TRUTH CONVERGED, AND DOES IT CONSERVE ENERGY?
#
# For one circle, one rectangle and one rounded rectangle the FE problem is
# solved on three meshes (maxh, maxh/2, maxh/4) and two things are recorded:
#
#   1. heat balance   q L_x  =  int_hole h (T - T_inf) ds
#      (all heat entering through the top leaves through the cavity wall;
#       the other edges are adiabatic, see fem/solver.py)
#   2. mesh sensitivity: RMSE between the grid temperatures of successive
#      meshes, on the solid grid points
#
# Output: <results>/<case_tag>/mesh_convergence/mesh_convergence.csv
# =============================================================================

import csv
from dataclasses import replace

import numpy as np
from ngsolve import Integrate

from ..fem.sampling import sample_case

TEST_SHAPES = [
    ("circle", 0.012, 0.0, 0.0),
    ("rect", 0.012, 0.008, 0.0),
    ("rounded_rect", 0.012, 0.008, 0.0015),
]


def run_mesh_check(cfg):
    folder = cfg.study_dir / "mesh_convergence"
    folder.mkdir(parents=True, exist_ok=True)

    rows = []
    for shape, p1, p2, p3 in TEST_SHAPES:
        previous_T = None

        for level in range(3):
            h = cfg.maxh / 2 ** level
            level_cfg = replace(cfg, maxh=h)

            mesh, gfu, sdf, mask, T, pts, triangles, T_fe = sample_case(
                level_cfg, shape, p1, p2, cfg.preview_cx, cfg.preview_cy, p3=p3)
            solid = mask > 0.5

            heat_in = cfg.q_top * cfg.Lx
            heat_out = float(Integrate(cfg.h_hole * (gfu - cfg.T_inf), mesh,
                                       definedon=mesh.Boundaries("hole")))

            if previous_T is None:
                change = None
            else:
                change = float(np.sqrt(np.mean((T[solid] - previous_T[solid]) ** 2)))

            row = {
                "shape": shape,
                "level": level,
                "maxh_m": h,
                "order": cfg.order,
                "elements": mesh.ne,
                "mean_solid_K": float(T[solid].mean()),
                "max_solid_K": float(T[solid].max()),
                "heat_in_W_per_m": heat_in,
                "heat_out_W_per_m": heat_out,
                "heat_balance_relative_residual": abs(heat_out - heat_in) / heat_in,
                "successive_grid_rmse_K": change,
            }
            rows.append(row)
            previous_T = T.copy()
            print(row, flush=True)

    with open(folder / "mesh_convergence.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
