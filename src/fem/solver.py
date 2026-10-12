# =============================================================================
# SOLVER: STEADY HEAT CONDUCTION BY THE FINITE ELEMENT METHOD (NGSolve)
#
# Strong form, in the solid Omega:
#     - div( k grad T ) = 0
#
# Boundary conditions (n = outward normal of the solid):
#     top    :  k dT/dn = q                      (heat flux enters)
#     hole   : -k dT/dn = h (T - T_inf)          (Robin: convection to coolant)
#     bottom, left, right :  k dT/dn = 0         (adiabatic)
#
# Weak form: multiply by a test function v, integrate over Omega, integrate
# by parts and insert the boundary conditions:
#
#     int_Omega k grad T . grad v  +  int_hole h T v
#   = int_top q v                  +  int_hole h T_inf v
#
#     \________ left side a(T, v) ________/   \____ right side f(v) ____/
#
# There is no fixed-temperature (Dirichlet) boundary, so every unknown is
# free. The Robin term is what pins the temperature level; without it the
# problem would only be defined up to a constant.
# =============================================================================

# =============================================================================
# VARIABLES (listed once)
#
#   cfg.kappa  : thermal conductivity k          [W / (m K)]
#   cfg.h_hole : heat transfer coefficient h     [W / (m^2 K)]
#   cfg.T_inf  : coolant temperature T_inf       [K]
#   cfg.q_top  : heat flux on the top edge q     [W / m^2]
#   cfg.order  : polynomial order of the elements (2 = quadratic)
#   cfg.outdir : folder for output files
#   mesh       : triangular mesh from geometry.build_mesh
#   space      : finite element space (all continuous order-p functions)
#   T, v       : unknown temperature (trial) and test function
#   a          : left-side matrix   K   (stiffness + Robin)
#   f          : right-side vector  F   (top flux + Robin coolant term)
#   temperature: solution T_h, stored as a GridFunction
# =============================================================================

import os

from ngsolve import H1, BilinearForm, LinearForm, GridFunction
from ngsolve import grad, dx, ds, TaskManager, VTKOutput


# =============================================================================
# SOLVE  K T = F  FOR ONE MESH
# =============================================================================
def solve_temperature(cfg, mesh):
    k = cfg.kappa
    h = cfg.h_hole
    T_inf = cfg.T_inf
    q = cfg.q_top

    # FINITE ELEMENT SPACE: CONTINUOUS PIECEWISE POLYNOMIALS OF ORDER p
    space = H1(mesh, order=cfg.order)
    T = space.TrialFunction()
    v = space.TestFunction()

    # LEFT SIDE a(T, v): CONDUCTION OVER THE SOLID + ROBIN ON THE CAVITY WALL
    # dx = integral over the area, ds("name") = integral along that boundary
    a = BilinearForm(space, symmetric=True)
    a += k * grad(T) * grad(v) * dx
    a += h * T * v * ds("hole")

    # RIGHT SIDE f(v): HEAT IN THROUGH THE TOP + COOLANT TEMPERATURE AT THE HOLE
    # adiabatic edges contribute nothing, so they do not appear
    f = LinearForm(space)
    f += q * v * ds("top")
    f += h * T_inf * v * ds("hole")

    temperature = GridFunction(space)

    # ASSEMBLE THE MATRIX AND VECTOR, THEN SOLVE WITH A SPARSE DIRECT SOLVER
    # TaskManager lets NGSolve use several CPU threads
    with TaskManager():
        a.Assemble()
        f.Assemble()

        K_inverse = a.mat.Inverse(freedofs=space.FreeDofs())
        temperature.vec.data = K_inverse * f.vec

    return temperature


# =============================================================================
# WRITE THE TEMPERATURE FIELD TO A VTK FILE (for ParaView)
#
# subdivision = 2 splits every triangle into smaller ones on output, so the
# quadratic solution is drawn smoothly instead of as flat triangles.
# =============================================================================
def export_vtk(cfg, mesh, temperature, name):
    filename = os.path.join(cfg.outdir, name)

    vtk = VTKOutput(
        mesh,
        coefs=[temperature],
        names=["temperature"],
        filename=filename,
        subdivision=2,
    )
    vtk.Do()
