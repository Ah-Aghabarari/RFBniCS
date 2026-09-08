

import time
start_time = time.perf_counter()
from pathlib import Path
import dolfinx.geometry
import matplotlib.pyplot as plt
import numpy.typing as npt
import argparse
import numpy as np
from mpi4py import MPI
import dolfinx.fem.petsc
import basix.ufl
import ufl
import dolfinx.common



def evaluate_over_line(
    u: dolfinx.fem.Function,
    start_point: npt.NDArray[np.floating],
    end_point: npt.NDArray[np.floating],
    num_points: int,
):
    weights = np.linspace(0, 1, num_points, endpoint=True)
    points = np.outer(1 - weights, start_point) + np.outer(weights, end_point)

    if points.shape[1] < 3:

        _points = points.copy()
        points = np.zeros((points.shape[0], 3), dtype=np.float64)
        points[:, :_points.shape[1]] = _points

    domain = u.function_space.mesh
    bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim, padding=1e-08)
    cells = []
    points_on_proc = []
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, points)

    for i, point in enumerate(points):

        if len(colliding_cells.links(i)) > 0:

            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])


    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = u.eval(points_on_proc, cells)




    return points_on_proc, u_values














def solve_problem(
    W: np.float64,
    Nx: int,
    cell_type: dolfinx.mesh.CellType,
    sigma: np.float64,
    kappa: np.float64,
    s: np.float64,
    alpha_a: np.float64,
    alpha_c: np.float64,
    k0: np.float64,
    Eeq: np.float64,
    F: np.float64,
    R: np.float64,
    T: np.float64,
    j_applied: np.float64,
    D_V2: np.float64,
    D_V3: np.float64,
    c0_V2: np.float64,
    c0_V3: np.float64,
    z: np.float64,
    epsilon: np.float64,
    degree: int,
    snes_atol: float,
    snes_rtol: float,
    snes_stol: float,
):
    
    mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, Nx, points=(0.0, W))


    _sigma = dolfinx.fem.Constant(mesh, sigma)
    _kappa = dolfinx.fem.Constant(mesh, kappa)
    _s = dolfinx.fem.Constant(mesh, s)
    _alpha_a = dolfinx.fem.Constant(mesh, alpha_a)
    _alpha_c = dolfinx.fem.Constant(mesh, alpha_c)
    _k0 = dolfinx.fem.Constant(mesh, k0)
    _Eeq = dolfinx.fem.Constant(mesh, Eeq)
    _F = dolfinx.fem.Constant(mesh, F)
    _R = dolfinx.fem.Constant(mesh, R)
    _T = dolfinx.fem.Constant(mesh, T)
    _ja = dolfinx.fem.Constant(mesh, j_applied)
    _D_V2 = dolfinx.fem.Constant(mesh, D_V2)
    _D_V3 = dolfinx.fem.Constant(mesh, D_V3)
    _c0_V2 = dolfinx.fem.Constant(mesh, c0_V2)
    _c0_V3 = dolfinx.fem.Constant(mesh, c0_V3)
    _z = dolfinx.fem.Constant(mesh, z)
    _eps = dolfinx.fem.Constant(mesh, epsilon)

    #  function space:
    el = basix.ufl.element('Lagrange', mesh.basix_cell(), degree)
    me = basix.ufl.blocked_element(el, shape=(4,))
    V = dolfinx.fem.functionspace(mesh, me)
    X = dolfinx.fem.Function(V)
    phi_e, phi_l, c_V2, c_V3 = ufl.split(X)
    v_e, v_l, v_c_V2, v_c_V3 = ufl.TestFunctions(V)

    # Initial guess:

    X.sub(1).interpolate(lambda x: np.zeros(x.shape[1], dtype=np.float64))
    Eeq_val = float(Eeq)
    X.sub(0).interpolate(lambda x: np.full(x.shape[1], Eeq_val, dtype=np.float64))

    eps_c = dolfinx.fem.Constant(mesh, 1e-14)

    c_V2_eff = 0.5*(c_V2 + ufl.sqrt(c_V2 * c_V2 + eps_c * eps_c))
    c_V3_eff = 0.5*(c_V3 + ufl.sqrt(c_V3 * c_V3 + eps_c * eps_c))

    i0 = _z * _F * _k0 * c_V2_eff ** _alpha_c * c_V3_eff ** _alpha_a
    Eeq_local = _Eeq + _R * _T / _F * ufl.ln((c_V3_eff + eps_c) / (c_V2_eff + eps_c))
    eta = phi_e - phi_l - Eeq_local
    i_loc = i0 * (
        ufl.exp(_alpha_a * _z * _F * eta / (_R * _T))
        - ufl.exp(-_alpha_c * _z * _F * eta / (_R * _T)))

    f = _s * i_loc

    Vc_V2, _ = V.sub(2).collapse()
    c_V2_n = dolfinx.fem.Function(Vc_V2)
    c_V2_n.name = 'c_V2_n'
    c_V2_n.interpolate(lambda x: np.full(x.shape[1], float(_c0_V2)))
    Vc_V3, _ = V.sub(3).collapse()
    c_V3_n = dolfinx.fem.Function(Vc_V3)
    c_V3_n.name = 'c_V3_n'
    c_V3_n.interpolate(lambda x: np.full(x.shape[1], float(_c0_V3)))

    # Initial guess

    c_V2_init = dolfinx.fem.Expression(_c0_V2, V.sub(2).element.interpolation_points)
    X.sub(2).interpolate(c_V2_init)
    c_V3_init = dolfinx.fem.Expression(_c0_V3, V.sub(3).element.interpolation_points)
    X.sub(3).interpolate(c_V3_init)

    #  effectives; 
    D_eff_V2 = _D_V2 * _eps ** 1.5
    D_eff_V3 = _D_V3 * _eps ** 1.5
    _sigma = (1.0 - _eps) ** 1.5 * _sigma


    # Weak form
    F = ufl.inner(_sigma * ufl.grad(phi_e), ufl.grad(v_e)) * ufl.dx + ufl.inner(f, v_e) * ufl.dx

    F += +ufl.inner(_kappa * ufl.grad(phi_l), ufl.grad(v_l)) * ufl.dx - ufl.inner(f, v_l) * ufl.dx

    F += (+ufl.inner(D_eff_V2 * ufl.grad(c_V2), ufl.grad(v_c_V2)) * ufl.dx
        + 1.0 / _F * f * v_c_V2 * ufl.dx)
    
    F += (+ufl.inner(D_eff_V3 * ufl.grad(c_V3), ufl.grad(v_c_V3)) * ufl.dx
        - 1.0 / _F * f * v_c_V3 * ufl.dx)



    def current_collector(x):

        return np.isclose(x[0], 0)

    def membrane(x):
        
        return np.isclose(x[0], W)
    
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    num_facets_local = (
        mesh.topology.index_map(mesh.topology.dim - 1).size_local
        + mesh.topology.index_map(mesh.topology.dim - 1).num_ghosts)

    values = np.full(num_facets_local, 0, dtype=np.int32)
    values[boundary_facets] = 1
    values[
        dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, current_collector)
    ] = 2
    values[dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, membrane)] = 3
    non_zero_entries = np.flatnonzero(values)
    ft = dolfinx.mesh.meshtags(
        mesh, mesh.topology.dim - 1, non_zero_entries, values[non_zero_entries])
    
    ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft)

    # Neumann Boundary:
    F -= _ja * v_e * ds(2)
    F += _ja * v_l * ds(3)


    mem_facets = ft.find(3).astype(np.int32)
    c_V2_bc_dofs_mem = dolfinx.fem.locate_dofs_topological(V.sub(2), mesh.topology.dim - 1, mem_facets)
    
    bc_c_V2_Mem = dolfinx.fem.dirichletbc(_c0_V2, c_V2_bc_dofs_mem, V.sub(2))
    mem_facets = ft.find(3).astype(np.int32)

    c_V3_bc_dofs_mem = dolfinx.fem.locate_dofs_topological(V.sub(3), mesh.topology.dim - 1, mem_facets)
    
    bc_c_V3_Mem = dolfinx.fem.dirichletbc(_c0_V3, c_V3_bc_dofs_mem, V.sub(3))

    petsc_options = {

        'snes_error_if_not_converged': True,
        'snes_type': 'newtonls',
        'snes_linesearch_type': 'none',
        'snes_atol': snes_atol,
        'snes_rtol': snes_rtol,
        'snes_stol': snes_stol,
        'snes_monitor': None,
        'ksp_type': 'preonly',
        'ksp_error_if_not_converged': True,
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
        'mat_mumps_icntl_24': 1,
        'mat_mumps_icntl_25': 0,
    }

    cffi_options = ['-Ofast', '-march=native']
    jit_options = {'cffi_extra_compile_args': cffi_options, 'cffi_libraries': ['m']}

    problem = dolfinx.fem.petsc.NonlinearProblem(
        F, X, bcs=[bc_c_V2_Mem, bc_c_V3_Mem],
        petsc_options=petsc_options, petsc_options_prefix='solver', jit_options=jit_options)
    
    ###############
    problem.solve()
    ##############




    uh_e = X.sub(0).collapse()
    uh_e.name = 'phi_e'
    uh_l = X.sub(1).collapse()
    uh_l.name = 'phi_l'
    uh_c2 = X.sub(2).collapse()
    uh_c2.name = 'c_V2'
    uh_c3 = X.sub(3).collapse()
    uh_c3.name = 'c_V3'

    eta_out = dolfinx.fem.Function(uh_e.function_space)
    eta_expr = dolfinx.fem.Expression(eta, uh_e.function_space.element.interpolation_points)
    eta_out.interpolate(eta_expr)
    eta_out.name = 'eta'  ############

    with dolfinx.io.VTKFile(mesh.comm, 'ue.pvd', 'w') as vtk:
        vtk.write_mesh(mesh)
        vtk.write_function([uh_e, uh_l, eta_out, uh_c2, uh_c3], 0)

    return {'uh_e': uh_e, 'uh_l': uh_l, 'uh_c_V2': uh_c2, 'uh_c_V3': uh_c3, 'eta': eta_out, 'W': W}








if __name__ == '__main__':

    def to_cell(cell: str) -> dolfinx.mesh.CellType:
        return dolfinx.mesh.to_type(cell)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-W', type=np.float64, default=0.0025, help='Width of electrode (M)')

    parser.add_argument('-Nx', type=int, default=2048, help='Number of elements in x-direction')

    parser.add_argument(
        '--cell-type',
        type=to_cell,
        choices=[dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral],
        dest='cell_type',
        default='quadrilateral')
    
    parser.add_argument('--degree', type=int, default=1, help='Finite element degree')

    parser.add_argument(
        '--sigma', type=np.float64, default=3402.1,
        help='Electrical conductivity of porous electrode (S/m)')
    
    parser.add_argument(
        '--kappa', type=np.float64, default=5.9514,
        help='Ionic conductivity of electrolyte (S/m)')
    
    parser.add_argument(
        '-s', type=np.float64, default=23000.0, help='Specific surface area (m^(-1))')
    
    parser.add_argument(
        '--alpha_a', type=np.float64, default=0.5, help='Anodic transfer coefficient (-)')
    
    parser.add_argument(
        '--alpha_c', type=np.float64, default=0.5, help='Cathodic transfer coefficient (-)')

    
    parser.add_argument(
        '--k0', type=np.float64, default=2.5e-07, help='Standard rate constant (m/s)')
    
    parser.add_argument('--Eeq', type=np.float64, default=-0.36, help='Equilibrium potential (V)')

    parser.add_argument(
        '-F', type=np.float64, default=96485.33212, help="Faraday's constant (C/mol)")
    
    parser.add_argument(
        '-R', type=np.float64, default=8.314462618, help='Universal gas constant (J/(mol*K))')
    
    parser.add_argument('-T', type=np.float64, default=298.15, help='Temperature (K)')

    parser.add_argument(
        '--j_applied', type=np.float64, default=-0.5, help='Applied current density (A/m^2)')
    
    parser.add_argument(
        '--D_V2', type=np.float64, default=8.1e-12, help='Diffusivity of V2+ (m^2/s)')
    
    parser.add_argument(
        '--D_V3', type=np.float64, default=1.65e-11, help='Diffusivity of V3+ (m^2/s)')
    
    parser.add_argument(
        '--c0_V2', type=np.float64, default=300.0,
        help='Inlet concentration of V2+ (mol/m^3)')
    
    parser.add_argument(
        '--c0_V3', type=np.float64, default=300.0,
        help='Inlet concentration of V3+ (mol/m^3)')
    
    parser.add_argument(
        '--z', type=np.float64, default=1.0,
        help='Number of electrons transferred in the reaction [-]')
    
    parser.add_argument(
        '--epsilon', type=np.float64, default=0.94, help='Porosity of the porous electrode (-)')
    
    parser.add_argument('--snes_atol', type=float, default=1e-06, help='SNES absolute tolerance')

    parser.add_argument('--snes_rtol', type=float, default=1e-06, help='SNES relative tolerance')

    parser.add_argument('--snes_stol', type=float, default=1e-06, help='SNES incremental tolerance')



    args, unknown = parser.parse_known_args()

    MPI.COMM_WORLD.barrier()
    with dolfinx.common.Timer() as t:
        res = solve_problem(**vars(args))
    MPI.COMM_WORLD.barrier()
    elapsed = t.elapsed()


    try:
        seconds = elapsed.total_seconds()
    except AttributeError:

        seconds = float(elapsed[0])
    seconds = MPI.COMM_WORLD.allreduce(seconds, op=MPI.MAX)


    if MPI.COMM_WORLD.rank == 0:

        print(f'Total runtime: {seconds:.6f} s')
    comm = MPI.COMM_WORLD

    out_dir = Path('exports')
    if comm.rank == 0:
        out_dir.mkdir(exist_ok=True)
    start_point = np.array([0.0], dtype=np.float64)
    end_point = np.array([res['W']], dtype=np.float64)
    num_points = args.Nx + 1

    def export_profile(field_fn, yname: str, filename: str):

        pts_local, y_local = evaluate_over_line(field_fn, start_point, end_point, num_points)
        x_local = pts_local[:, 0]
        y_local = y_local.reshape(-1)
        x_list = comm.gather(x_local, root=0)
        y_list = comm.gather(y_local, root=0)

        if comm.rank == 0:
            x = np.concatenate(x_list)
            y = np.concatenate(y_list)
            order = np.argsort(x)
            x = x[order]
            y = y[order]
            data = np.column_stack([x, y])
            np.savetxt(out_dir / filename, data, delimiter=',', header=f'x,{yname}', comments='')
            return x, y


        
        return None, None

    
    x_eta, eta_vals = export_profile(res['eta'], 'eta', f'Fenics_2ndsimplifed_Nx{args.Nx}_eta_vs_x.csv')

    export_profile(res['uh_c_V2'], 'c_V2', f'Fenics_2ndsimplifed_Nx{args.Nx}_cV2_vs_x.csv')
    export_profile(res['uh_c_V3'], 'c_V3', f'Fenics_2ndsimplifed_Nx{args.Nx}_cV3_vs_x.csv')


    if comm.rank == 0:
        plt.figure()
        plt.plot(x_eta, eta_vals, linewidth=2, label='$\\eta$')
        plt.grid(True)
        plt.legend()
        plt.xlabel('x [m]')
        plt.ylabel('Overpotential [V]')
        plt.tight_layout()
        plt.savefig(out_dir / 'eta_vs_x.png', dpi=300)
        plt.show()

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f'TOTAL Elapsed time: {elapsed_time:.4f} seconds')
