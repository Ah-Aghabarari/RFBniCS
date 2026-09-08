import argparse
import csv
from pathlib import Path

from mpi4py import MPI

import basix.ufl
import dolfinx.common
import dolfinx.fem.petsc
import numpy as np
import ufl

from rfbnics import assemble_global, global_component_range, global_function_range


def solve_problem(
    W: np.float64,
    Nx: int,
    sigma: np.float64,
    a: np.float64,
    alpha_a: np.float64,
    alpha_c: np.float64,
    k0: np.float64,
    E_formal: np.float64,
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
    epsilon: np.float64,
    time_schedule: list[tuple[list[tuple[int, float]], float, str]],
    degree: int,
    output_every: int,
    output_file: str,
    snes_atol: float,
    snes_rtol: float,
    snes_stol: float,
    outdir: Path,
):
    mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, Nx, [0.0, float(W)])

    _sigma = dolfinx.fem.Constant(mesh, sigma)
    _a = dolfinx.fem.Constant(mesh, a)
    _alpha_a = dolfinx.fem.Constant(mesh, alpha_a)
    _alpha_c = dolfinx.fem.Constant(mesh, alpha_c)
    _k0 = dolfinx.fem.Constant(mesh, k0)
    _E_formal = dolfinx.fem.Constant(mesh, E_formal)
    _F = dolfinx.fem.Constant(mesh, F)
    _R = dolfinx.fem.Constant(mesh, R)
    _T = dolfinx.fem.Constant(mesh, T)
    _i_app = dolfinx.fem.Constant(mesh, np.float64(0.0))
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
    _eps = dolfinx.fem.Constant(mesh, epsilon)

    if not time_schedule:
        raise ValueError("time_schedule must contain at least one operation")

    first_step_schedule = time_schedule[0][0]

    dt_c = dolfinx.fem.Constant(mesh, np.float64(first_step_schedule[0][1]))

    # -------------------------------------------------------------------------
    # Boundary markers
    # -------------------------------------------------------------------------
    def current_collector(x):
        return np.isclose(x[0], 0)

    def membrane(x):
        return np.isclose(x[0], W)

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    num_facets_local = (
        mesh.topology.index_map(mesh.topology.dim - 1).size_local
        + mesh.topology.index_map(mesh.topology.dim - 1).num_ghosts
    )

    values = np.full(num_facets_local, 0, dtype=np.int32)
    values[boundary_facets] = 1
    values[
        dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, current_collector)
    ] = 2
    values[dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, membrane)] = 3

    non_zero_entries = np.flatnonzero(values).astype(np.int32)
    ft = dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim - 1,
        non_zero_entries,
        values[non_zero_entries],
    )

    qdeg = max(4, 2 * degree + 2)
    metadata = {"quadrature_degree": qdeg}

    dx = ufl.Measure("dx", domain=mesh, metadata=metadata)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft, metadata=metadata)

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

    c_V2_s = c_V2_eff
    c_V3_s = c_V3_eff

    # BV
    i_F = i0 * (
        (c_V2_s / (c_V2_eff + eps_c)) * ufl.exp(_alpha_a * _F * eta / (_R * _T))
        - (c_V3_s / (c_V3_eff + eps_c)) * ufl.exp(-_alpha_c * _F * eta / (_R * _T))
    )

    f = _a * i_F

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
    D_eff_V2 = _D_V2 * _eps**1.5
    D_eff_V3 = _D_V3 * _eps**1.5
    D_eff_H = _D_H * _eps**1.5
    D_eff_HSO4 = _D_HSO4 * _eps**1.5
    D_eff_SO4 = _D_SO4 * _eps**1.5

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
    _sigma = (1.0 - _eps) ** 1.5 * _sigma
    residual = ufl.inner(_sigma * ufl.grad(phi_s), ufl.grad(v_s)) * dx + ufl.inner(f, v_s) * dx

    # Liquid phase:
    residual += +ufl.inner(-i_l, ufl.grad(v_l)) * dx - ufl.inner(f, v_l) * dx

    # V2+ molar flux
    N_V2 = -D_eff_V2 * ufl.grad(c_V2) - D_eff_V2 * _z_V2 * _F * c_V2 / (_R * _T) * ufl.grad(phi_l)

    # V3+ molar flux
    N_V3 = -D_eff_V3 * ufl.grad(c_V3) - D_eff_V3 * _z_V3 * _F * c_V3 / (_R * _T) * ufl.grad(phi_l)

    # H+ molar flux
    N_H = -D_eff_H * ufl.grad(c_H) - D_eff_H * _z_H * _F * c_H / (_R * _T) * ufl.grad(phi_l)

    # HSO4- molar flux
    N_HSO4 = -D_eff_HSO4 * ufl.grad(c_HSO4) - D_eff_HSO4 * _z_HSO4 * _F * c_HSO4 / (
        _R * _T
    ) * ufl.grad(phi_l)

    # ---------------------------------------------------------
    # Transient species equations, backward Euler:
    #
    # d(eps*c_j)/dt + div(N_j) = S_j
    #
    # ---------------------------------------------------------

    # V2+: source S_V2 = -f/F
    residual += (
        _eps * (c_V2 - c_V2_old) / dt_c * v_c_V2 * dx
        - ufl.inner(N_V2, ufl.grad(v_c_V2)) * dx
        + (1.0 / _F) * f * v_c_V2 * dx
    )

    # V3+: source S_V3 = +f/F
    residual += (
        _eps * (c_V3 - c_V3_old) / dt_c * v_c_V3 * dx
        - ufl.inner(N_V3, ufl.grad(v_c_V3)) * dx
        - (1.0 / _F) * f * v_c_V3 * dx
    )

    residual += _eps * (c_H - c_H_old) / dt_c * v_c_H * dx - ufl.inner(N_H, ufl.grad(v_c_H)) * dx

    residual += (
        _eps * (c_HSO4 - c_HSO4_old) / dt_c * v_c_HSO4 * dx
        - ufl.inner(N_HSO4, ufl.grad(v_c_HSO4)) * dx
    )

    # Neumann Boundary condition:
    residual -= _i_app * v_s * ds(2)
    residual += _i_app * v_l * ds(3)

    # Dirichlet Boundary condition:
    membrane_facets = ft.find(3).astype(np.int32)
    c_V2_bc_dofs_mem = dolfinx.fem.locate_dofs_topological(
        V.sub(2), mesh.topology.dim - 1, membrane_facets
    )
    c_V3_bc_dofs_mem = dolfinx.fem.locate_dofs_topological(
        V.sub(3), mesh.topology.dim - 1, membrane_facets
    )
    c_H_bc_dofs_mem = dolfinx.fem.locate_dofs_topological(
        V.sub(4), mesh.topology.dim - 1, membrane_facets
    )
    c_HSO4_bc_dofs_mem = dolfinx.fem.locate_dofs_topological(
        V.sub(5), mesh.topology.dim - 1, membrane_facets
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

    # ---------------------------------------------------------
    # Output spaces and time-independent output fields
    # ---------------------------------------------------------

    # Scalar space used for derived concentration fields
    Vs, _ = V.sub(2).collapse()

    # ---------------------------------------------------------
    # Construct output fields from the current solution X
    # ---------------------------------------------------------
    def make_output_fields():
        # Primary unknowns
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

        # Overpotential
        eta_out = dolfinx.fem.Function(uh_s.function_space)
        eta_out.name = "overpotential [V]"

        eta_expr = dolfinx.fem.Expression(
            eta,
            uh_s.function_space.element.interpolation_points,
        )
        eta_out.interpolate(eta_expr)
        eta_out.x.scatter_forward()

        # SO4 concentration
        c_SO4_out = dolfinx.fem.Function(Vs)
        c_SO4_out.name = "c_SO4 [mol/m^3]"

        c_SO4_expr = dolfinx.fem.Expression(
            c_SO4,
            Vs.element.interpolation_points,
        )
        c_SO4_out.interpolate(c_SO4_expr)
        c_SO4_out.x.scatter_forward()

        # State of charge
        SOC_out = dolfinx.fem.Function(uh_c2.function_space)
        SOC_out.name = "SOC [-]"

        SOC_expr = dolfinx.fem.Expression(
            c_V2 / (c_V2 + c_V3 + eps_c),
            uh_c2.function_space.element.interpolation_points,
        )
        SOC_out.interpolate(SOC_expr)
        SOC_out.x.scatter_forward()

        # Local interfacial current density
        i_F_out = dolfinx.fem.Function(uh_c2.function_space)
        i_F_out.name = "i_F [A/m^2]"

        i_F_expr = dolfinx.fem.Expression(
            i_F,
            uh_c2.function_space.element.interpolation_points,
        )
        i_F_out.interpolate(i_F_expr)
        i_F_out.x.scatter_forward()

        # Volumetric reaction-current source
        f_out = dolfinx.fem.Function(uh_c2.function_space)
        f_out.name = "volumetric_source_f [A/m^3]"

        f_expr = dolfinx.fem.Expression(
            f,
            uh_c2.function_space.element.interpolation_points,
        )
        f_out.interpolate(f_expr)
        f_out.x.scatter_forward()

        # Equilibrium potential
        Eeq_out = dolfinx.fem.Function(uh_c2.function_space)
        Eeq_out.name = "Eeq_neg [V]"

        Eeq_expr = dolfinx.fem.Expression(
            Eeq_neg,
            uh_c2.function_space.element.interpolation_points,
        )
        Eeq_out.interpolate(Eeq_expr)
        Eeq_out.x.scatter_forward()

        # Surface V2 concentration
        c_V2_s_out = dolfinx.fem.Function(uh_c2.function_space)
        c_V2_s_out.name = "c_V2_surface [mol/m^3]"

        c_V2_s_expr = dolfinx.fem.Expression(
            c_V2_s,
            uh_c2.function_space.element.interpolation_points,
        )
        c_V2_s_out.interpolate(c_V2_s_expr)
        c_V2_s_out.x.scatter_forward()

        # Surface V3 concentration
        c_V3_s_out = dolfinx.fem.Function(uh_c3.function_space)
        c_V3_s_out.name = "c_V3_surface [mol/m^3]"

        c_V3_s_expr = dolfinx.fem.Expression(
            c_V3_s,
            uh_c3.function_space.element.interpolation_points,
        )
        c_V3_s_out.interpolate(c_V3_s_expr)
        c_V3_s_out.x.scatter_forward()

        # Surface-to-bulk V2 ratio
        c_V2_ratio_out = dolfinx.fem.Function(uh_c2.function_space)
        c_V2_ratio_out.name = "c_V2_surface_over_bulk [-]"

        c_V2_ratio_expr = dolfinx.fem.Expression(
            c_V2_s / (c_V2_eff + eps_c),
            uh_c2.function_space.element.interpolation_points,
        )
        c_V2_ratio_out.interpolate(c_V2_ratio_expr)
        c_V2_ratio_out.x.scatter_forward()

        # Surface-to-bulk V3 ratio
        c_V3_ratio_out = dolfinx.fem.Function(uh_c3.function_space)
        c_V3_ratio_out.name = "c_V3_surface_over_bulk [-]"

        c_V3_ratio_expr = dolfinx.fem.Expression(
            c_V3_s / (c_V3_eff + eps_c),
            uh_c3.function_space.element.interpolation_points,
        )
        c_V3_ratio_out.interpolate(c_V3_ratio_expr)
        c_V3_ratio_out.x.scatter_forward()

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
        ]

    # =========================================================
    # Diagnostics
    # =========================================================

    cc_tag = 2
    mem_tag = 3

    one = dolfinx.fem.Constant(mesh, np.float64(1.0))

    L_cc = assemble_global(one * ds(cc_tag))
    L_mem = assemble_global(one * ds(mem_tag))

    # Compile timestep diagnostic forms only once
    phi_s_cc_form = dolfinx.fem.form(phi_s * ds(cc_tag))

    phi_l_mem_form = dolfinx.fem.form(phi_l * ds(mem_tag))

    amount_V2_form = dolfinx.fem.form(_eps * c_V2 * dx)

    amount_V3_form = dolfinx.fem.form(_eps * c_V3 * dx)

    voltage_history = []

    eta_summary_h = dolfinx.fem.Function(Vs)
    eta_summary_expr = dolfinx.fem.Expression(
        eta,
        Vs.element.interpolation_points,
    )

    def print_timestep_summary(
        *,
        time_s,
        time_step_size_s,
        mode,
        current_density,
    ):
        """Calculate and print the requested timestep summary."""

        # -----------------------------------------------------
        # Calculated half-cell voltage
        # -----------------------------------------------------

        phi_s_cc_local = dolfinx.fem.assemble_scalar(phi_s_cc_form)

        phi_s_cc = mesh.comm.allreduce(phi_s_cc_local, op=MPI.SUM) / L_cc

        phi_l_mem_local = dolfinx.fem.assemble_scalar(phi_l_mem_form)

        phi_l_mem = mesh.comm.allreduce(phi_l_mem_local, op=MPI.SUM) / L_mem

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
        # Maximum overpotential
        # -----------------------------------------------------

        eta_summary_h.interpolate(eta_summary_expr)
        eta_summary_h.x.scatter_forward()

        eta_min, eta_max = global_function_range(eta_summary_h)

        maximum_overpotential = max(abs(eta_min), abs(eta_max))

        # -----------------------------------------------------
        # Minimum active-vanadium concentration
        # -----------------------------------------------------

        c_V2_min, _ = global_component_range(X, 2)
        c_V3_min, _ = global_component_range(X, 3)

        # -----------------------------------------------------
        # Print summary on rank zero
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

            _i_app.value = np.array([float(signed_current)], dtype=np.float64)

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
        ([(5, 0.1), (10, 1.0), (20, 2.0)], 0.0, "rest"),
        ([(20, 0.1), (10, 1.0), (10, 10.0)], 0.5, "discharge"),
    ]

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-W", type=np.float64, default=2.5e-3, help="Width of electrode (M)")

    parser.add_argument("-Nx", type=int, default=250, help="Number of elements in x-direction")

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
        "--c0_V2", type=np.float64, default=300.0, help="Concentration of V2+ on membrane (mol/m^3)"
    )

    parser.add_argument(
        "--c0_V3", type=np.float64, default=300.0, help="Concentration of V3+ on membrane (mol/m^3)"
    )

    parser.add_argument(
        "--c0_H", type=np.float64, default=1000.0, help="Concentration of H+ on membrane (mol/m^3)"
    )

    parser.add_argument(
        "--c0_HSO4",
        type=np.float64,
        default=1000.0,
        help="Concentration of HSO4 on membrane (mol/m^3)",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path().cwd() / "output" / "1D", help="Output directory"
    )
    # Charges
    parser.add_argument("--z_V2", type=np.float64, default=2.0, help="Charge number of V2+")

    parser.add_argument("--z_V3", type=np.float64, default=3.0, help="Charge number of V3+")

    parser.add_argument("--z_H", type=np.float64, default=1.0, help="Charge number of H+")

    parser.add_argument("--z_HSO4", type=np.float64, default=-1.0, help="Charge number of HSO4-")

    parser.add_argument("--z_SO4", type=np.float64, default=-2.0, help="Charge number of SO4^(-2)")

    parser.add_argument(
        "--epsilon", type=np.float64, default=0.94, help="Porosity of the porous electrode (-)"
    )

    parser.add_argument(
        "--output_file", type=str, default="1D_transient", help="Name of the output files"
    )

    parser.add_argument("--snes_atol", type=float, default=1e-8, help="SNES absolute tolerance")

    parser.add_argument("--snes_rtol", type=float, default=1e-10, help="SNES relative tolerance")

    parser.add_argument("--snes_stol", type=float, default=1e-10, help="SNES incremental tolerance")

    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    args.time_schedule = TIME_SCHEDULE

    # res = solve_problem(**vars(args))

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
