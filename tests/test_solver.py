# =============================================================================
# TEST: FE SOLVER  (fem/solver.py)
#   steady state  ->  heat in through the top = heat out through the cavity
#                     q Lx = int_hole h (T - T_inf) ds
#   maximum principle -> every solid temperature lies above the coolant T_inf
# =============================================================================

import types

import numpy as np
from ngsolve import Integrate

from src.fem.geometry import build_mesh
from src.fem.solver import solve_temperature


def make_cfg():
    return types.SimpleNamespace(Lx=0.036, Ly=0.021, maxh=0.0012, rounded_corner_segments=12,
                                 kappa=28.3, h_hole=4000.0, T_inf=300.0, q_top=0.5e6, order=2)


def test_energy_balance_and_minimum_temperature():
    cfg = make_cfg()
    for shape, p1, p2 in [("circle", 0.008, 0.0), ("rect", 0.012, 0.008)]:
        mesh = build_mesh(cfg, shape, p1, p2, 0.0, 0.018, 0.009)
        T = solve_temperature(cfg, mesh)

        heat_in = cfg.q_top * cfg.Lx
        heat_out = Integrate(cfg.h_hole * (T - cfg.T_inf), mesh, definedon=mesh.Boundaries("hole"))
        assert abs(heat_out - heat_in) / heat_in < 1.0e-8, (shape, heat_out)

        vertex_values = np.array(T.vec)[:mesh.nv]
        assert vertex_values.min() > cfg.T_inf
