import argparse
import csv
import time
from pathlib import Path

from mpi4py import MPI

import basix.ufl
import dolfinx
import dolfinx.common
import dolfinx.fem.petsc
import numpy as np
import ufl

from rfbnics import global_component_range, global_function_range

start_time = time.perf_counter()


def solve_flow_fttf(
    mesh_filename: str,
    Lp: float,
    df: float,
    Kc: float,
    rho: float,
    eps: float,
    mu: float,
    u_in: float,
):

    mesh_data = dolfinx.io.gmsh.read_from_msh(mesh_filename, comm=MPI.COMM_WORLD, rank=0)
    mesh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags
    tag_map = mesh_data.physical_groups

    scalar_type = dolfinx.default_scalar_type

    # Ergun relation:
    # beta = 1.75 * (1 - epsilon) / (L_p * epsilon^3)
    beta = 1.75 * (1.0 - eps) / (Lp * eps**3)

    # Kozeny--Carman permeability for the fibrous porous electrode:
    # K = d_f^2 * epsilon^3 / [C_KC * (1 - epsilon)^2]
    permeability = df**2 * eps**3 / (Kc * (1.0 - eps) ** 2)

    # -----------------------------
    # Measures and subdomains
    # -----------------------------
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)

    fluid_tags = (tag_map["InletChannel"].tag, tag_map["OutletChannel"].tag)
    dxF = dx(fluid_tags)

    electrode_tags = (tag_map["Electrode"].tag,)
    dxE = dx(electrode_tags)

    # -----------------------------
    # Function spaces
    # -----------------------------
    v_el = basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
    p_el = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)

    V = dolfinx.fem.functionspace(mesh, v_el)
    Q_space = dolfinx.fem.functionspace(mesh, p_el)
    W = ufl.MixedFunctionSpace(V, Q_space)

    vs = dolfinx.fem.Function(V)
    ps = dolfinx.fem.Function(Q_space)
    wh = [vs, ps]

    # -----------------------------
    # Typed DOLFINx constants
    # -----------------------------
    epsilon = dolfinx.fem.Constant(mesh, scalar_type(eps))
    rho_c = dolfinx.fem.Constant(mesh, scalar_type(rho))
    mu_c = dolfinx.fem.Constant(mesh, scalar_type(mu))
    K = dolfinx.fem.Constant(mesh, scalar_type(permeability))
    beta_c = dolfinx.fem.Constant(mesh, scalar_type(beta))

    # Dimensional body-force/source term [N/m^3]
    f_dim = dolfinx.fem.Constant(mesh, np.zeros(mesh.geometry.dim, dtype=scalar_type))

    # -----------------------------
    # Boundary conditions
    # -----------------------------
    inlet_tag = tag_map["Inlet"].tag
    assert facet_tags is not None
    inlet_dofs = dolfinx.fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(inlet_tag))

    def u_inlet(x):
        output = np.zeros((mesh.geometry.dim, x.shape[1]), dtype=scalar_type)
        output[0] = u_in
        return output

    u_bc = dolfinx.fem.Function(V)
    u_bc.interpolate(u_inlet)
    bc_inlet = dolfinx.fem.dirichletbc(u_bc, inlet_dofs)

    wall_dofs = dolfinx.fem.locate_dofs_topological(
        V, facet_tags.dim, facet_tags.find(tag_map["Walls"].tag)
    )

    u_walls = dolfinx.fem.Function(V)
    u_walls.interpolate(lambda x: np.zeros((mesh.geometry.dim, x.shape[1]), dtype=scalar_type))
    bc_walls = dolfinx.fem.dirichletbc(u_walls, wall_dofs)

    cc_dofs = dolfinx.fem.locate_dofs_topological(
        V, facet_tags.dim, facet_tags.find(tag_map["CurrentCollector"].tag)
    )
    u_cc = dolfinx.fem.Function(V)
    u_cc.interpolate(lambda x: np.zeros((mesh.geometry.dim, x.shape[1]), dtype=scalar_type))
    bc_cc = dolfinx.fem.dirichletbc(u_cc, cc_dofs)

    membrane_dofs = dolfinx.fem.locate_dofs_topological(
        V, facet_tags.dim, facet_tags.find(tag_map["Membrane"].tag)
    )
    u_membrane = dolfinx.fem.Function(V)
    u_membrane.interpolate(lambda x: np.zeros((mesh.geometry.dim, x.shape[1]), dtype=scalar_type))
    bc_membrane = dolfinx.fem.dirichletbc(u_membrane, membrane_dofs)

    bcs = [bc_inlet, bc_walls, bc_membrane, bc_cc]

    # -----------------------------
    # Initial linear Stokes-Brinkman solve
    # -----------------------------
    v_test, p_test = ufl.TestFunctions(W)

    # Dimensional initial problem:
    # We keep the linear terms in the equations.
    #
    # Electrode:
    #   -div(v) = 0
    #   -grad(p) + mu/eps Delta(v) - mu/K v = 0
    #
    # Channels:
    #   -div(v) = 0
    #   -grad(p) + mu Delta(v) = 0
    #
    F_init = (
        -ufl.div(vs) * p_test * (dxF + dxE)
        + (mu_c / epsilon) * ufl.inner(ufl.grad(vs), ufl.grad(v_test)) * dxE
        + mu_c * ufl.inner(ufl.grad(vs), ufl.grad(v_test)) * dxF
        + (mu_c / K) * ufl.dot(vs, v_test) * dxE
        - ps * ufl.div(v_test) * (dxF + dxE)
    )

    F_init -= ufl.dot(f_dim, v_test) * (dxF + dxE)

    # Explicit zero RHS for the continuity block.
    # Required to preserve the two-block nested RHS structure.
    F_init += dolfinx.fem.Constant(mesh, scalar_type(0.0)) * p_test * dx(929332)

    w = ufl.TrialFunctions(W)
    a_init, L_init = ufl.system(ufl.replace(F_init, {vs: w[0], ps: w[1]}))

    initial_problem = dolfinx.fem.petsc.LinearProblem(
        ufl.extract_blocks(a_init),
        ufl.extract_blocks(L_init),
        u=wh,
        bcs=bcs,
        kind="nest",
        petsc_options_prefix="DarcyBrinkMan_",
        petsc_options={
            "ksp_monitor": None,
            "ksp_max_it": 500,
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_type": "preonly",
        },
    )

    initial_problem.solve()

    if mesh.comm.rank == 0:
        print("Initial solve complete. Starting nonlinear problem.", flush=True)

    # -----------------------------
    # Nonlinear dimensional residual
    # -----------------------------
    #
    # Electrode dimensional equation:
    #   rho/eps^2 (v . grad) v = -grad(p) + mu/eps Delta(v) - mu/K v - rho beta |v| v
    #
    #
    # Channels:
    #   rho (v . grad) v = -grad(p) + mu Delta(v)

    tol_speed_sq = dolfinx.fem.Constant(mesh, scalar_type((1e-12) ** 2))
    speed = ufl.sqrt(ufl.dot(vs, vs) + tol_speed_sq)


    F = (rho_c / epsilon**2) * ufl.dot(ufl.dot(ufl.grad(vs), vs), v_test) * dxE


    F += rho_c * ufl.dot(ufl.dot(ufl.grad(vs), vs), v_test) * dxF

    F += -ufl.div(vs) * p_test * (dxF + dxE) - ps * ufl.div(v_test) * (dxF + dxE)

    F += (mu_c / epsilon) * ufl.inner(ufl.grad(vs), ufl.grad(v_test)) * dxE

    F += mu_c * ufl.inner(ufl.grad(vs), ufl.grad(v_test)) * dxF

    F += (mu_c / K) * ufl.dot(vs, v_test) * dxE

    F += rho_c * beta_c * ufl.dot(speed * vs, v_test) * dxE

    F -= ufl.dot(f_dim, v_test) * (dxF + dxE)

    J = ufl.derivative(F, wh, w)

    gamma = dolfinx.fem.Constant(mesh, scalar_type(1.0e-3 * mu))

    J_regularized = ufl.extract_blocks(
        gamma * ufl.inner(ufl.div(w[0]), ufl.div(v_test)) * (dxF + dxE) + J
    )

    nonlinear_problem = dolfinx.fem.petsc.NonlinearProblem(
        ufl.extract_blocks(F),
        wh,
        bcs=bcs,
        J=J_regularized,
        kind="nest",
        petsc_options_prefix="stokes_brinkman_",
        petsc_options={
            "snes_error_if_not_converged": True,
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-8,
            "snes_atol": 1e-8,
            "snes_monitor": None,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )

    nonlinear_problem.solve()

    v_out = dolfinx.fem.Function(V, name="V")
    v_out.x.array[:] = vs.x.array
    v_out.x.scatter_forward()

    p_out = dolfinx.fem.Function(Q_space, name="p [Pa]")

    p_out.x.array[:] = ps.x.array
    p_out.x.scatter_forward()

    return mesh, cell_tags, facet_tags, tag_map, v_out, p_out


def solve_problem(
    sigma: np.float64,
    a: np.float64,
    alpha_a: np.float64,
    alpha_c: np.float64,
    k0: np.float64,
    E_formal: np.float64,
    df: np.float64,
    F: np.float64,
    R: np.float64,
    T: np.float64,
    D_V2: np.float64,
    D_V3: np.float64,
    D_H: np.float64,
    D_HSO4: np.float64,
    D_SO4: np.float64,
    c0_V2: np.float64,
    c0_V3: np.float64,
    c0_H: np.float64,
    c0_HSO4: np.float64,
    z_V2: np.float64,
    z_V3: np.float64,
    z_H: np.float64,
    z_HSO4: np.float64,
    z_SO4: np.float64,
    mu: np.float64,
    rho: np.float64,
    Kc: np.float64,
    epsilon: np.float64,
    u_in: np.float64,
    flow_mesh: str,
    Lp: np.float64,
    time_schedule: list[tuple[list[tuple[int, float]], float, str]],
    degree: int,
    output_every: int,
    output_file: str,
    snes_atol: float,
    snes_rtol: float,
    snes_stol: float,
    outdir: Path,
):

    mesh, cell_tags, facet_tags, tag_map, v_flow, p_flow = solve_flow_fttf(
        mesh_filename=flow_mesh,
        Lp=Lp,
        df=df,
        Kc=Kc,
        rho=rho,
        eps=epsilon,
        mu=mu,
        u_in=u_in,
    )

    """

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)"""

    qdeg = max(4, 2 * degree + 2)
    metadata = {"quadrature_degree": qdeg}

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags, metadata=metadata)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags, metadata=metadata)

    dxE = dx((tag_map["Electrode"].tag,))
    dxF = dx((tag_map["InletChannel"].tag, tag_map["OutletChannel"].tag))
    dxAll = dxE + dxF

    inlet_tag = tag_map["Inlet"].tag
    outlet_tag = tag_map["Outlet"].tag
    cc_tag = tag_map["CurrentCollector"].tag
    mem_tag = tag_map["Membrane"].tag
    n_vec = ufl.FacetNormal(mesh)

    _sigma = dolfinx.fem.Constant(mesh, sigma)
    _a = dolfinx.fem.Constant(mesh, a)
    _alpha_a = dolfinx.fem.Constant(mesh, alpha_a)
    _alpha_c = dolfinx.fem.Constant(mesh, alpha_c)
    _k0 = dolfinx.fem.Constant(mesh, k0)
    _E_formal = dolfinx.fem.Constant(mesh, E_formal)
    _df = dolfinx.fem.Constant(mesh, df)
    _F = dolfinx.fem.Constant(mesh, F)
    _R = dolfinx.fem.Constant(mesh, R)
    _T = dolfinx.fem.Constant(mesh, T)
    _D_V2 = dolfinx.fem.Constant(mesh, D_V2)
    _D_V3 = dolfinx.fem.Constant(mesh, D_V3)
    _D_H = dolfinx.fem.Constant(mesh, D_H)
    _D_HSO4 = dolfinx.fem.Constant(mesh, D_HSO4)
    _D_SO4 = dolfinx.fem.Constant(mesh, D_SO4)
    _c0_V2 = dolfinx.fem.Constant(mesh, c0_V2)
    _c0_V3 = dolfinx.fem.Constant(mesh, c0_V3)
    _c0_H = dolfinx.fem.Constant(mesh, c0_H)
    _c0_HSO4 = dolfinx.fem.Constant(mesh, c0_HSO4)
    _z_V2 = dolfinx.fem.Constant(mesh, z_V2)
    _z_V3 = dolfinx.fem.Constant(mesh, z_V3)
    _z_H = dolfinx.fem.Constant(mesh, z_H)
    _z_HSO4 = dolfinx.fem.Constant(mesh, z_HSO4)
    _z_SO4 = dolfinx.fem.Constant(mesh, z_SO4)
    _mu = dolfinx.fem.Constant(mesh, mu)
    _rho = dolfinx.fem.Constant(mesh, rho)
    _eps = dolfinx.fem.Constant(mesh, epsilon)

    if not time_schedule:
        raise ValueError("time_schedule must contain at least one operation")

    first_step_schedule = time_schedule[0][0]

    dt_c = dolfinx.fem.Constant(mesh, np.float64(first_step_schedule[0][1]))

    # Mixed function space:
    el = basix.ufl.element("Lagrange", mesh.basix_cell(), degree)
    me = basix.ufl.blocked_element(el, shape=(6,))
    V = dolfinx.fem.functionspace(mesh, me)
    # Unknown at new time level n+1
    X = dolfinx.fem.Function(V)

    # Known solution from old time level n
    X_old = dolfinx.fem.Function(V)

    phi_s, phi_l, c_V2, c_V3, c_H, c_HSO4 = ufl.split(X)
    _, _, c_V2_old, c_V3_old, c_H_old, c_HSO4_old = ufl.split(X_old)
    v_s, v_l, v_c_V2, v_c_V3, v_c_H, v_c_HSO4 = ufl.TestFunctions(V)

    Q0 = dolfinx.fem.functionspace(mesh, ("DG", 0))

    mask_E = dolfinx.fem.Function(Q0, name="mask_E")
    mask_E.x.array[:] = 0.0
    mask_E.x.array[cell_tags.find(tag_map["Electrode"].tag)] = 1.0

    mask_F = dolfinx.fem.Function(Q0, name="mask_F")
    mask_F.x.array[:] = 1.0 - mask_E.x.array

    # Storage coefficient for transient concentrations.
    # Electrode: epsilon*c
    # Channels: 1*c
    eps_storage = mask_E * _eps + mask_F * dolfinx.fem.Constant(mesh, np.float64(1.0))

    # Electroneutrality:
    c_SO4 = -(_z_V2 * c_V2 + _z_V3 * c_V3 + _z_H * c_H + _z_HSO4 * c_HSO4) / _z_SO4

    eps_c = dolfinx.fem.Constant(mesh, 1e-14)
    c_V2_eff = 0.5 * (c_V2 + ufl.sqrt(c_V2 * c_V2 + eps_c * eps_c))
    c_V3_eff = 0.5 * (c_V3 + ufl.sqrt(c_V3 * c_V3 + eps_c * eps_c))

    # Overpotential
    Eeq_neg = _E_formal + (_R * _T / _F) * ufl.ln((c_V3_eff + eps_c) / (c_V2_eff + eps_c))
    eta = phi_s - phi_l - Eeq_neg

    # Exchange current density i0
    i0 = _F * _k0 * (c_V2_eff**_alpha_c) * (c_V3_eff**_alpha_a)

    # Fixed Darcy velocity from the independent pressure solve
    eps_speed = dolfinx.fem.Constant(mesh, 1e-16)
    v_mag = ufl.sqrt(ufl.dot(v_flow, v_flow) + eps_speed**2)

    # Reynolds number Re = rho |u| d_f / mu
    Re = _rho * v_mag * _df / _mu

    # Mass-transfer coefficients k_m,j
    k_mR = 7.0 * _D_V2 / _df * (Re**0.4)  # for V2+ (R)
    k_mO = 7.0 * _D_V3 / _df * (Re**0.4)  # for V3+ (O)

    # Coefficients A and B
    A = (
        _k0
        * (c_V2_eff + eps_c) ** (_alpha_c - 1.0)
        * (c_V3_eff + eps_c) ** (_alpha_a)
        * ufl.exp(_alpha_a * _F * eta / (_R * _T))
    )
    B = (
        _k0
        * (c_V2_eff + eps_c) ** (_alpha_c)
        * (c_V3_eff + eps_c) ** (_alpha_a - 1.0)
        * ufl.exp(-_alpha_c * _F * eta / (_R * _T))
    )

    # Surface concentrations from explicit solution
    den = 1.0 + A / (k_mR + eps_c) + B / (k_mO + eps_c)

    c_V3_s = (
        (1.0 + A / (k_mR + eps_c)) * c_V3_eff + (A / (k_mO + eps_c)) * c_V2_eff
    ) / den  # C_O^s
    c_V2_s = (
        (1.0 + B / (k_mO + eps_c)) * c_V2_eff + (B / (k_mR + eps_c)) * c_V3_eff
    ) / den  # C_R^s

    # Clamp to nonnegative
    c_V2_s = 0.5 * (c_V2_s + ufl.sqrt(c_V2_s * c_V2_s + eps_c * eps_c))
    c_V3_s = 0.5 * (c_V3_s + ufl.sqrt(c_V3_s * c_V3_s + eps_c * eps_c))

    # BV
    i_F_raw = i0 * (
        (c_V2_s / (c_V2_eff + eps_c)) * ufl.exp(_alpha_a * _F * eta / (_R * _T))
        - (c_V3_s / (c_V3_eff + eps_c)) * ufl.exp(-_alpha_c * _F * eta / (_R * _T))
    )

    # Faradaic current exists only inside the porous electrode
    i_F = mask_E * i_F_raw

    # Volumetric reaction-current source
    f = _a * i_F

    # Since the membrane and current collector have different areas,
    # their current densities are scaled to give the same total current.
    one = dolfinx.fem.Constant(mesh, np.float64(1.0))

    A_cc_local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(one * ds(cc_tag)))
    A_mem_local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(one * ds(mem_tag)))

    A_cc = mesh.comm.allreduce(A_cc_local, op=MPI.SUM)
    A_mem = mesh.comm.allreduce(A_mem_local, op=MPI.SUM)

    if mesh.comm.rank == 0:
        print("A_cc  =", A_cc)
        print("A_mem =", A_mem)
        print("A_mem / A_cc =", A_mem / A_cc)

    _i_mem = dolfinx.fem.Constant(mesh, np.float64(0.0))
    _i_cc = dolfinx.fem.Constant(mesh, np.float64(0.0))

    # Initial guess for c in the mixed function:
    c_V2_init = dolfinx.fem.Expression(_c0_V2, V.sub(2).element.interpolation_points)
    X.sub(2).interpolate(c_V2_init)

    c_V3_init = dolfinx.fem.Expression(_c0_V3, V.sub(3).element.interpolation_points)
    X.sub(3).interpolate(c_V3_init)

    c_H_init = dolfinx.fem.Expression(_c0_H, V.sub(4).element.interpolation_points)
    X.sub(4).interpolate(c_H_init)

    c_HSO4_init = dolfinx.fem.Expression(_c0_HSO4, V.sub(5).element.interpolation_points)
    X.sub(5).interpolate(c_HSO4_init)

    # Initial guesses for potentials.
    phi_s_init = dolfinx.fem.Expression(
        dolfinx.fem.Constant(mesh, np.float64(0.0)), V.sub(0).element.interpolation_points
    )
    X.sub(0).interpolate(phi_s_init)

    phi_l_init = dolfinx.fem.Expression(
        dolfinx.fem.Constant(mesh, np.float64(-float(E_formal))),
        V.sub(1).element.interpolation_points,
    )
    X.sub(1).interpolate(phi_l_init)

    # Initial condition for transient solve:
    # X_old stores c^n at the beginning of the first time step.
    X.x.scatter_forward()
    X_old.x.array[:] = X.x.array
    X_old.x.scatter_forward()

    # Bruggeman approximation
    D_eff_V2 = mask_E * (_D_V2 * _eps**1.5) + mask_F * _D_V2
    D_eff_V3 = mask_E * (_D_V3 * _eps**1.5) + mask_F * _D_V3
    D_eff_H = mask_E * (_D_H * _eps**1.5) + mask_F * _D_H
    D_eff_HSO4 = mask_E * (_D_HSO4 * _eps**1.5) + mask_F * _D_HSO4
    D_eff_SO4 = mask_E * (_D_SO4 * _eps**1.5) + mask_F * _D_SO4

    kappa_np = (_F**2 / (_R * _T)) * (
        (_z_V2**2) * D_eff_V2 * c_V2_eff
        + (_z_V3**2) * D_eff_V3 * c_V3_eff
        + (_z_H**2) * D_eff_H * c_H
        + (_z_HSO4**2) * D_eff_HSO4 * c_HSO4
        + (_z_SO4**2) * D_eff_SO4 * c_SO4
    )
    i_l = -kappa_np * ufl.grad(phi_l) - _F * (
        _z_V2 * D_eff_V2 * ufl.grad(c_V2)
        + _z_V3 * D_eff_V3 * ufl.grad(c_V3)
        + _z_H * D_eff_H * ufl.grad(c_H)
        + _z_HSO4 * D_eff_HSO4 * ufl.grad(c_HSO4)
        + _z_SO4 * D_eff_SO4 * ufl.grad(c_SO4)
    )

    # Weak form:
    # Solid phase:

    """
    sigma_eff = mask_E * ((1.0 - _eps)**1.5 * _sigma) + mask_F * dolfinx.fem.Constant(mesh, 1e-14)
    F = (
        ufl.inner(sigma_eff * ufl.grad(phi_s), ufl.grad(v_s)) * dxAll
        + f * v_s * dxE
    )
    """
    residual = (
        ufl.inner(((1.0 - _eps) ** 1.5 * _sigma) * ufl.grad(phi_s), ufl.grad(v_s)) * dxE
        + f * v_s * dxE
    )

    # Liquid phase:
    residual += +ufl.inner(-i_l, ufl.grad(v_l)) * dxAll - ufl.inner(f, v_l) * dxE

    # V2+ molar flux
    N_V2 = (
        -D_eff_V2 * ufl.grad(c_V2)
        - D_eff_V2 * _z_V2 * _F * c_V2 / (_R * _T) * ufl.grad(phi_l)
        + v_flow * c_V2
    )

    # V3+ molar flux
    N_V3 = (
        -D_eff_V3 * ufl.grad(c_V3)
        - D_eff_V3 * _z_V3 * _F * c_V3 / (_R * _T) * ufl.grad(phi_l)
        + v_flow * c_V3
    )

    # H+ molar flux
    N_H = (
        -D_eff_H * ufl.grad(c_H)
        - D_eff_H * _z_H * _F * c_H / (_R * _T) * ufl.grad(phi_l)
        + v_flow * c_H
    )

    # HSO4- molar flux
    N_HSO4 = (
        -D_eff_HSO4 * ufl.grad(c_HSO4)
        - D_eff_HSO4 * _z_HSO4 * _F * c_HSO4 / (_R * _T) * ufl.grad(phi_l)
        + v_flow * c_HSO4
    )

    # ---------------------------------------------------------
    # Transient species equations, backward Euler:
    #
    # d(eps*c_j)/dt + div(N_j) = S_j
    #
    # ---------------------------------------------------------

    # V2+: source S_V2 = -f/F
    residual += (
        eps_storage * (c_V2 - c_V2_old) / dt_c * v_c_V2 * dxAll
        - ufl.inner(N_V2, ufl.grad(v_c_V2)) * dxAll
        + (1.0 / _F) * f * v_c_V2 * dxE
    )

    # V3+: source S_V3 = +f/F
    residual += (
        eps_storage * (c_V3 - c_V3_old) / dt_c * v_c_V3 * dxAll
        - ufl.inner(N_V3, ufl.grad(v_c_V3)) * dxAll
        - (1.0 / _F) * f * v_c_V3 * dxE
    )

    residual += (
        eps_storage * (c_H - c_H_old) / dt_c * v_c_H * dxAll
        - ufl.inner(N_H, ufl.grad(v_c_H)) * dxAll
    )

    residual += (
        eps_storage * (c_HSO4 - c_HSO4_old) / dt_c * v_c_HSO4 * dxAll
        - ufl.inner(N_HSO4, ufl.grad(v_c_HSO4)) * dxAll
    )

    # Neumann Boundary condition:
    residual -= _i_cc * v_s * ds(cc_tag)
    residual += _i_mem * v_l * ds(mem_tag)

    # Proton flux through the membrane
    residual += _i_mem / (_z_H * _F) * v_c_H * ds(mem_tag)

    # Outlet species fluxes
    # Convective outflow through outlet boundary
    residual += (
        ufl.dot(v_flow, n_vec) * c_V2 * v_c_V2 * ds(outlet_tag)
        + ufl.dot(v_flow, n_vec) * c_V3 * v_c_V3 * ds(outlet_tag)
        + ufl.dot(v_flow, n_vec) * c_H * v_c_H * ds(outlet_tag)
        + ufl.dot(v_flow, n_vec) * c_HSO4 * v_c_HSO4 * ds(outlet_tag)
    )

    # Dirichlet Boundary condition:
    # Constant inlet concentration:
    inlet_facets = facet_tags.find(inlet_tag).astype(np.int32)
    c_V2_bc_dofs_mem = dolfinx.fem.locate_dofs_topological(
        V.sub(2), mesh.topology.dim - 1, inlet_facets
    )
    c_V3_bc_dofs_mem = dolfinx.fem.locate_dofs_topological(
        V.sub(3), mesh.topology.dim - 1, inlet_facets
    )
    c_H_bc_dofs_mem = dolfinx.fem.locate_dofs_topological(
        V.sub(4), mesh.topology.dim - 1, inlet_facets
    )
    c_HSO4_bc_dofs_mem = dolfinx.fem.locate_dofs_topological(
        V.sub(5), mesh.topology.dim - 1, inlet_facets
    )
    bc_c_V2_in = dolfinx.fem.dirichletbc(_c0_V2, c_V2_bc_dofs_mem, V.sub(2))
    bc_c_V3_in = dolfinx.fem.dirichletbc(_c0_V3, c_V3_bc_dofs_mem, V.sub(3))
    bc_c_H_in = dolfinx.fem.dirichletbc(_c0_H, c_H_bc_dofs_mem, V.sub(4))
    bc_c_HSO4_in = dolfinx.fem.dirichletbc(_c0_HSO4, c_HSO4_bc_dofs_mem, V.sub(5))

    # PETSc / SNES options
    petsc_options = {
        "snes_error_if_not_converged": True,
        "snes_type": "newtonls",
        # "snes_linesearch_type": "none",
        "snes_linesearch_type": "bt",
        "snes_max_it": 100,
        "snes_atol": snes_atol,
        "snes_rtol": snes_rtol,
        "snes_stol": snes_stol,
        "snes_monitor": None,
        "snes_converged_reason": None,
        "ksp_type": "preonly",
        "ksp_error_if_not_converged": True,
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1,
        "mat_mumps_icntl_25": 0,
    }
    cffi_options = ["-Ofast", "-march=native"]
    jit_options = {"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]}
    problem = dolfinx.fem.petsc.NonlinearProblem(
        residual,
        X,
        bcs=[bc_c_V2_in, bc_c_V3_in, bc_c_H_in, bc_c_HSO4_in],
        petsc_options=petsc_options,
        petsc_options_prefix="solver",
        jit_options=jit_options,
    )

    gdim = mesh.geometry.dim
    v_el = basix.ufl.element("Lagrange", mesh.basix_cell(), degree, shape=(gdim,))
    Ve = dolfinx.fem.functionspace(mesh, v_el)

    # ---------------------------------------------------------
    # Output helper for transient ParaView visualization
    # ---------------------------------------------------------
    Qp = dolfinx.fem.functionspace(mesh, basix.ufl.element("Lagrange", mesh.basix_cell(), degree))

    uh_p = dolfinx.fem.Function(Qp, name="p [Pa]")
    p_expr = dolfinx.fem.Expression(p_flow, Qp.element.interpolation_points)
    uh_p.interpolate(p_expr)
    uh_p.x.scatter_forward()

    uh_v = dolfinx.fem.Function(Ve, name="v [m/s]")
    v_expr = dolfinx.fem.Expression(v_flow, Ve.element.interpolation_points)
    uh_v.interpolate(v_expr)
    uh_v.x.scatter_forward()

    # Scalar space used for derived concentration fields
    Vs, _ = V.sub(2).collapse()

    # ---------------------------------------------------------
    # Construct output fields from the current solution X
    # ---------------------------------------------------------
    def make_output_fields():
        uh_s = X.sub(0).collapse()
        uh_s.name = "phi_s [V]"

        uh_l = X.sub(1).collapse()
        uh_l.name = "phi_l [V]"

        uh_c2 = X.sub(2).collapse()
        uh_c2.name = "c_V2 [mol/m^3]"

        uh_c3 = X.sub(3).collapse()
        uh_c3.name = "c_V3 [mol/m^3]"

        uh_cH = X.sub(4).collapse()
        uh_cH.name = "c_H [mol/m^3]"

        uh_cHSO4 = X.sub(5).collapse()
        uh_cHSO4.name = "c_HSO4 [mol/m^3]"

        eta_out = dolfinx.fem.Function(uh_s.function_space)
        eta_out.name = "overpotential [V]"
        eta_expr = dolfinx.fem.Expression(
            eta,
            uh_s.function_space.element.interpolation_points,
        )
        eta_out.interpolate(eta_expr)

        # SO4 concentration
        c_SO4_out = dolfinx.fem.Function(Vs)
        c_SO4_out.name = "c_SO4 [mol/m^3]"
        c_SO4_expr = dolfinx.fem.Expression(
            c_SO4,
            Vs.element.interpolation_points,
        )
        c_SO4_out.interpolate(c_SO4_expr)

        # State of charge
        SOC_out = dolfinx.fem.Function(uh_c2.function_space)
        SOC_out.name = "SOC [-]"
        SOC_expr = dolfinx.fem.Expression(
            c_V2 / (c_V2 + c_V3 + eps_c),
            uh_c2.function_space.element.interpolation_points,
        )
        SOC_out.interpolate(SOC_expr)

        # Local interfacial current density
        i_F_out = dolfinx.fem.Function(uh_c2.function_space)
        i_F_out.name = "i_F [A/m^2]"
        i_F_expr = dolfinx.fem.Expression(
            i_F,
            uh_c2.function_space.element.interpolation_points,
        )
        i_F_out.interpolate(i_F_expr)

        # Volumetric reaction-current source
        f_out = dolfinx.fem.Function(uh_c2.function_space)
        f_out.name = "volumetric_source_f [A/m^3]"
        f_expr = dolfinx.fem.Expression(
            f,
            uh_c2.function_space.element.interpolation_points,
        )
        f_out.interpolate(f_expr)

        # Equilibrium potential
        Eeq_out = dolfinx.fem.Function(uh_c2.function_space)
        Eeq_out.name = "Eeq_neg [V]"
        Eeq_expr = dolfinx.fem.Expression(
            Eeq_neg,
            uh_c2.function_space.element.interpolation_points,
        )
        Eeq_out.interpolate(Eeq_expr)

        # Surface V2 concentration
        c_V2_s_out = dolfinx.fem.Function(uh_c2.function_space)
        c_V2_s_out.name = "c_V2_surface [mol/m^3]"
        c_V2_s_expr = dolfinx.fem.Expression(
            c_V2_s,
            uh_c2.function_space.element.interpolation_points,
        )
        c_V2_s_out.interpolate(c_V2_s_expr)

        # Surface V3 concentration
        c_V3_s_out = dolfinx.fem.Function(uh_c3.function_space)
        c_V3_s_out.name = "c_V3_surface [mol/m^3]"
        c_V3_s_expr = dolfinx.fem.Expression(
            c_V3_s,
            uh_c3.function_space.element.interpolation_points,
        )
        c_V3_s_out.interpolate(c_V3_s_expr)

        # Surface-to-bulk V2 ratio
        c_V2_ratio_out = dolfinx.fem.Function(uh_c2.function_space)
        c_V2_ratio_out.name = "c_V2_surface_over_bulk [-]"
        c_V2_ratio_expr = dolfinx.fem.Expression(
            c_V2_s / (c_V2_eff + eps_c),
            uh_c2.function_space.element.interpolation_points,
        )
        c_V2_ratio_out.interpolate(c_V2_ratio_expr)

        # Surface-to-bulk V3 ratio
        c_V3_ratio_out = dolfinx.fem.Function(uh_c3.function_space)
        c_V3_ratio_out.name = "c_V3_surface_over_bulk [-]"
        c_V3_ratio_expr = dolfinx.fem.Expression(
            c_V3_s / (c_V3_eff + eps_c),
            uh_c3.function_space.element.interpolation_points,
        )
        c_V3_ratio_out.interpolate(c_V3_ratio_expr)

        return [
            uh_s,
            uh_l,
            eta_out,
            uh_c2,
            uh_c3,
            uh_cH,
            uh_cHSO4,
            c_SO4_out,
            SOC_out,
            i_F_out,
            f_out,
            Eeq_out,
            c_V2_s_out,
            c_V3_s_out,
            c_V2_ratio_out,
            c_V3_ratio_out,
            uh_p,
            uh_v,
        ]

    # =========================================================
    # Diagnostics
    # =========================================================
    # Compile timestep diagnostic forms only once
    phi_s_cc_form = dolfinx.fem.form(phi_s * ds(cc_tag))

    phi_l_mem_form = dolfinx.fem.form(phi_l * ds(mem_tag))

    amount_V2_form = dolfinx.fem.form(_eps * c_V2 * dxE)

    amount_V3_form = dolfinx.fem.form(_eps * c_V3 * dxE)

    voltage_history = []

    eta_summary_h = dolfinx.fem.Function(Vs)
    eta_summary_expr = dolfinx.fem.Expression(
        mask_E * eta,
        Vs.element.interpolation_points,
    )

    def print_timestep_summary(
        *,
        time_s,
        time_step_size_s,
        mode,
        current_density,
    ):
        """Calculate and print the timestep summary."""

        # -----------------------------------------------------
        # Negative half-cell voltage
        # -----------------------------------------------------

        phi_s_cc_local = dolfinx.fem.assemble_scalar(phi_s_cc_form)

        phi_s_cc = mesh.comm.allreduce(phi_s_cc_local, op=MPI.SUM) / A_cc

        phi_l_mem_local = dolfinx.fem.assemble_scalar(phi_l_mem_form)

        phi_l_mem = mesh.comm.allreduce(phi_l_mem_local, op=MPI.SUM) / A_mem

        calculated_voltage = phi_s_cc - phi_l_mem

        if mesh.comm.rank == 0:
            voltage_history.append(
                {
                    "time_s": float(time_s),
                    "mode": str(mode),
                    "current_density_A_per_m2": float(current_density),
                    "phi_s_cc_V": float(phi_s_cc),
                    "phi_l_mem_V": float(phi_l_mem),
                    "negative_half_cell_voltage_V": float(calculated_voltage),
                }
            )

        # -----------------------------------------------------
        # State of charge
        # -----------------------------------------------------

        amount_V2_local = dolfinx.fem.assemble_scalar(amount_V2_form)

        amount_V2 = mesh.comm.allreduce(amount_V2_local, op=MPI.SUM)

        amount_V3_local = dolfinx.fem.assemble_scalar(amount_V3_form)

        amount_V3 = mesh.comm.allreduce(amount_V3_local, op=MPI.SUM)

        total_active_vanadium = amount_V2 + amount_V3

        state_of_charge = amount_V2 / max(total_active_vanadium, 1.0e-30)

        # -----------------------------------------------------
        # Maximum electrode overpotential
        # -----------------------------------------------------

        eta_summary_h.interpolate(eta_summary_expr)
        eta_summary_h.x.scatter_forward()

        eta_min, eta_max = global_function_range(eta_summary_h)

        maximum_overpotential = max(
            abs(eta_min),
            abs(eta_max),
        )

        # -----------------------------------------------------
        # Minimum concentrations
        # -----------------------------------------------------

        c_V2_min, _ = global_component_range(X, 2)
        c_V3_min, _ = global_component_range(X, 3)

        # -----------------------------------------------------
        # Print
        # -----------------------------------------------------

        if mesh.comm.rank == 0:
            print(
                "\n"
                "Timestep summary:\n"
                f"  time                         = "
                f"{float(time_s):.8e} s\n"
                f"  time-step size               = "
                f"{float(time_step_size_s):.8e} s\n"
                f"  operating mode               = "
                f"{mode}\n"
                "\n"
                f"  calculated voltage           = "
                f"{calculated_voltage:+.8e} V\n"
                f"  state of charge              = "
                f"{state_of_charge:.8e}\n"
                f"  maximum |overpotential|      = "
                f"{maximum_overpotential:+.8e} V\n"
                f"  minimum V2 concentration     = "
                f"{c_V2_min:.8e} mol/m^3\n"
                f"  minimum V3 concentration     = "
                f"{c_V3_min:.8e} mol/m^3\n",
                flush=True,
            )

    # Sign convention
    mode_sign = {
        "charge": -1.0,
        "discharge": 1.0,
        "rest": 0.0,
    }

    n_steps = sum(count for step_schedule, _, _ in time_schedule for count, _ in step_schedule)

    t = 0.0
    step = 0

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with dolfinx.io.VTKFile(mesh.comm, outdir / f"{output_file}.pvd", "w") as vtk:
        # Write initial condition at t = 0
        vtk.write_mesh(mesh)

        vtk.write_function(make_output_fields(), 0.0)

        for step_schedule, current_magnitude, operation_mode in time_schedule:
            if operation_mode not in mode_sign:
                raise ValueError(f"Unknown operation mode: {operation_mode!r}")

            signed_current = mode_sign[operation_mode] * float(current_magnitude)

            # The schedule specifies membrane current density.
            _i_mem.value = np.array([signed_current], dtype=np.float64)

            # Scale collector current density so that:
            # i_cc*A_cc = i_mem*A_mem
            _i_cc.value = np.array([signed_current * A_mem / A_cc], dtype=np.float64)

            for stage_steps, stage_dt in step_schedule:
                for _ in range(stage_steps):
                    step += 1

                    dt_c.value = np.array([float(stage_dt)], dtype=np.float64)
                    t += float(stage_dt)

                    if mesh.comm.rank == 0:
                        print(
                            "\n============================================================\n",
                            flush=True,
                        )

                    problem.solve()
                    X.x.scatter_forward()

                    print_timestep_summary(
                        time_s=t,
                        time_step_size_s=float(dt_c.value),
                        mode=operation_mode,
                        current_density=signed_current,
                    )

                    if step % output_every == 0 or step == n_steps:
                        vtk.write_function(make_output_fields(), t)

                    X_old.x.array[:] = X.x.array
                    X_old.x.scatter_forward()

    # ---------------------------------------------------------
    # Save voltage history
    # ---------------------------------------------------------
    if mesh.comm.rank == 0:
        csv_filename = outdir / f"{output_file}_voltage_history.csv"

        fieldnames = [
            "time_s",
            "mode",
            "current_density_A_per_m2",
            "phi_s_cc_V",
            "phi_l_mem_V",
            "negative_half_cell_voltage_V",
        ]

        with open(csv_filename, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(voltage_history)

        print(
            f"Voltage data written to {csv_filename}",
            flush=True,
        )


if __name__ == "__main__":
    TIME_SCHEDULE = [
        # ([NxDT] , CURRENT MAGNITUDE, "MODE")
        # MODE is charge, rest, or discharge; CURRENT MAGNITUDE is nonnegative and in A/m^2
        # Each (NxDT) entry specifies N time steps of size DT seconds
        # For example, (10, 1.0) = 10 time steps of size 1 seconds
        ([(20, 0.1), (10, 1.0), (10, 10.0)], 0.5, "charge"),
        ([(5, 0.1), (5, 1.0), (10, 10.0)], 0.0, "rest"),
        ([(20, 0.1), (10, 1.0), (10, 10.0)], 0.5, "discharge"),
    ]

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--degree", type=int, default=1, help="Finite element degree")

    parser.add_argument("--output_every", type=int, default=5, help="Write every N time steps")

    parser.add_argument(
        "--sigma",
        type=np.float64,
        default=3402.1,
        help="Intrinsic solid conductivity before Bruggemann correction (S/m)",
    )

    parser.add_argument(
        "-a", type=np.float64, default=2.3e4, help="Specific interfacial area (m^(-1))"
    )

    parser.add_argument(
        "--alpha_a",
        type=np.float64,
        default=0.5,
        help="Anodic transfer coefficient for V2+/V3+ negative electrode",
    )

    parser.add_argument(
        "--alpha_c",
        type=np.float64,
        default=0.5,
        help="Cathodic transfer coefficient for V2+/V3+ negative electrode",
    )

    parser.add_argument(
        "--k0",
        type=np.float64,
        default=2.5e-7,
        help="Standard rate constant for V2+/V3+ negative electrode (m/s)",
    )

    parser.add_argument(
        "--E_formal",
        type=np.float64,
        default=-0.36,
        help="Formal potential for V2+/V3+ negative electrode (V)",
    )

    parser.add_argument("--df", type=np.float64, default=0.5e-6, help="Carbon fiber diameter (m)")

    parser.add_argument(
        "-F", type=np.float64, default=96485.33212, help="Faraday's constant (C/mol)"
    )

    parser.add_argument(
        "-R", type=np.float64, default=8.314462618, help="Universal gas constant (J/(mol*K))"
    )

    parser.add_argument("-T", type=np.float64, default=298.15, help="Temperature (K)")

    parser.add_argument(
        "--D_V2", type=np.float64, default=8.10e-12, help="Diffusivity of V2+ (m^2/s)"
    )

    parser.add_argument(
        "--D_V3", type=np.float64, default=1.65e-11, help="Diffusivity of V3+ (m^2/s)"
    )

    parser.add_argument(
        "--D_H",
        type=np.float64,
        default=9.312e-9,  # source: 10.1149/2.017209jes
        help="Diffusivity of H (m^2/s)",
    )

    parser.add_argument(
        "--D_HSO4",
        type=np.float64,
        default=1.33e-9,  # source: 10.1149/2.017209jes
        help="Diffusivity of HSO4 (m^2/s)",
    )

    parser.add_argument(
        "--D_SO4",
        type=np.float64,
        default=1.065e-9,  # source: 10.1149/2.017209jes
        help="Diffusivity of SO4^(-2) (m^2/s)",
    )

    parser.add_argument(
        "--c0_V2", type=np.float64, default=300.0, help="Inlet concentration of V2+ (mol/m^3)"
    )

    parser.add_argument(
        "--c0_V3", type=np.float64, default=300.0, help="Inlet concentration of V3+ (mol/m^3)"
    )

    parser.add_argument(
        "--c0_H", type=np.float64, default=1000.0, help="Inlet concentration of H+ (mol/m^3)"
    )

    parser.add_argument(
        "--c0_HSO4", type=np.float64, default=1000.0, help="Inlet concentration of HSO4 (mol/m^3)"
    )

    # Charges
    parser.add_argument("--z_V2", type=np.float64, default=2.0, help="Charge number of V2+")

    parser.add_argument("--z_V3", type=np.float64, default=3.0, help="Charge number of V3+")

    parser.add_argument("--z_H", type=np.float64, default=1.0, help="Charge number of H+")

    parser.add_argument("--z_HSO4", type=np.float64, default=-1.0, help="Charge number of HSO4-")

    parser.add_argument("--z_SO4", type=np.float64, default=-2.0, help="Charge number of SO4^(-2)")

    parser.add_argument(
        "--mu", type=np.float64, default=6.963e-3, help="Electrolyte viscosity (Pa·s)"
    )

    parser.add_argument(
        "--rho", type=np.float64, default=1399.8, help="Electrolyte density (kg/m^3)"
    )

    parser.add_argument("--Kc", type=np.float64, default=5.55, help="Kozeny–Carman constant")

    parser.add_argument(
        "--epsilon", type=np.float64, default=0.94, help="Porosity of the porous electrode (-)"
    )

    parser.add_argument(
        "--u_in", type=np.float64, default=1.0e-7, help="Inlet superficial velocity (m/s)"
    )

    parser.add_argument(
        "--flow_mesh",
        type=str,
        default=str(Path.cwd() / "output" / "3D" / "FTTF.msh"),
        help="Gmsh mesh file used by the 3D flow solver",
    )

    parser.add_argument(
        "--Lp",
        type=np.float64,
        default=100.0e-6,
        help="Characteristic pore size used in the Ergun relation (m)",
    )

    parser.add_argument(
        "--outdir", type=Path, default=Path.cwd() / "output" / "3D", help="Output directory"
    )

    parser.add_argument(
        "--output_file", type=str, default="3D_transient", help="Name of the output files"
    )

    parser.add_argument("--snes_atol", type=float, default=1e-13, help="SNES absolute tolerance")

    parser.add_argument("--snes_rtol", type=float, default=1e-8, help="SNES relative tolerance")

    parser.add_argument("--snes_stol", type=float, default=1e-10, help="SNES incremental tolerance")

    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    args.time_schedule = TIME_SCHEDULE

    MPI.COMM_WORLD.barrier()
    with dolfinx.common.Timer() as t:
        solve_problem(**vars(args))
    MPI.COMM_WORLD.barrier()
    elapsed = t.elapsed()
    try:
        seconds = elapsed.total_seconds()
    except AttributeError:
        seconds = float(elapsed[0])
    seconds = MPI.COMM_WORLD.allreduce(seconds, op=MPI.MAX)
    if MPI.COMM_WORLD.rank == 0:
        print(f"Runtime: {seconds:.6f} s")
