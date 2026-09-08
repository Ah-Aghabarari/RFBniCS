# Benchmark from https://doi.org/10.48550/arXiv.2507.22818

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
    H: np.float64,
    N: int,
    M: int,
    cell_type: dolfinx.mesh.CellType,
    sigma: np.float64,
    kappa: np.float64,
    s: np.float64,
    j0: np.float64,
    Eeq: np.float64,
    F: np.float64,
    R: np.float64,
    T: np.float64,
    j_applied: np.float64,
    degree: int,
    snes_atol: float,
    snes_rtol: float,
    snes_stol: float,
):
    mesh = dolfinx.mesh.create_rectangle( MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([W, H])], [N, M],cell_type=cell_type)
    
    _sigma = dolfinx.fem.Constant(mesh , sigma)
    _kappa = dolfinx.fem.Constant(mesh , kappa)
    _s = dolfinx.fem.Constant(mesh , s)
    _j0 = dolfinx.fem.Constant(mesh , j0)
    _Eeq = dolfinx.fem.Constant(mesh , Eeq)
    _F = dolfinx.fem.Constant(mesh , F)
    _R = dolfinx.fem.Constant(mesh , R)
    _T = dolfinx.fem.Constant(mesh , T)
    _ja = dolfinx.fem.Constant(mesh  j_applied)


    el = basix.ufl.element('Lagrange', mesh.basix_cell() , degree)
    me = basix.ufl.blocked_element(el, shape=(2, ))
    V = dolfinx.fem.functionspace(mesh , me)
    potentials = dolfinx.fem.Function(V)
    phi_e, phi_l = ufl.split(potentials)



    a = 2 * _s * _j0
    b = 0.5 * _F / (_R * _T)
    eta = phi_e - phi_l - _Eeq
    f = a * ufl.sinh(b * eta)
    v_e, v_l = ufl.TestFunctions(V)

    F = ufl.inner(_sigma * ufl.grad(phi_e), ufl.grad(v_e)) * ufl.dx + ufl.inner(f, v_e) * ufl.dx
    F += ufl.inner(_kappa * ufl.grad(phi_l), ufl.grad(v_l)) * ufl.dx - ufl.inner(f, v_l) * ufl.dx

    def current_collector(x):
        return np.isclose(x[0], 0)

    def membrane(x):
        return np.isclose(x[0], W)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

    num_facets_local = (
        mesh.topology.index_map(mesh.topology.dim - 1).size_local
        + mesh.topology.index_map(mesh.topology.dim - 1).num_ghosts)
    
    boundary_value = 1
    cc_value = 2
    membrane_value = 3

    values = np.full(num_facets_local, 0, dtype=np.int32)
    values[boundary_facets] = boundary_value
    values[
        dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, current_collector)
    ] = cc_value
    values[
        dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, membrane)
    ] = membrane_value

    non_zero_entries = np.flatnonzero(values)

    ft = dolfinx.mesh.meshtags(
        mesh, mesh.topology.dim - 1 , non_zero_entries , values[non_zero_entries])
    
    ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft)
    F -= _ja * v_e * ds(cc_value)
    F += _ja * v_l * ds(membrane_value)

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
        F,
        potentials,
        petsc_options=petsc_options,
        petsc_options_prefix='solver',
        jit_options=jit_options)

    problem.solve()

    uh_e = potentials.sub(0).collapse()
    uh_l = potentials.sub(1).collapse()
    eta_out = dolfinx.fem.Function(uh_e.function_space, name='eta')
    eta_expr = dolfinx.fem.Expression(eta, uh_e.function_space.element.interpolation_points)
    uh_e.name = 'phi_e'
    uh_l.name = 'phi_l'
    eta_out.interpolate(eta_expr)


    return (uh_e, uh_l, eta_out)








if __name__ == '__main__':

    def to_cell(cell: str) -> dolfinx.mesh.CellType:
        return dolfinx.mesh.to_type(cell)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-W', type=np.float64, default=0.005, help='Width of electrode (M)')

    parser.add_argument('-H', type=np.float64, default=0.1, help='Height of electrode (M)')

    parser.add_argument('-N', type=int, default=2048, help='Number of elements in x-direction')

    parser.add_argument('-M', type=int, default=10, help='Number of elements in y-direction')

    parser.add_argument(
        '--cell-type',
        type=to_cell,
        choices=[dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral],
        dest='cell_type',
        default=dolfinx.mesh.CellType.quadrilateral)
    
    parser.add_argument('--degree', type=int, default=1, help='Finite element degree')

    parser.add_argument(
        '--sigma', type=np.float64, default=103.1891,
        help='Electrical conductivity of porous electrode (S/m)')
    
    parser.add_argument(
        '--kappa', type=np.float64, default=5.9514,
        help='Ionic conductivity of electrolyte (S/m)')
    
    parser.add_argument(
        '-s', type=np.float64, default=16400.0, help='Specific surface area (m^(-1))')
    
    parser.add_argument(
        '--j0', type=np.float64, default=2.7657, help='Exchange current density (A/m^2)')
    
    parser.add_argument(
        '--Eeq', type=np.float64, default=-0.1609, help='Equilibrium potential (V)')
    
    parser.add_argument(
        '-F', type=np.float64, default=96485.33212, help="Faraday's constant (C/mol)")
    
    parser.add_argument(
        '-R', type=np.float64, default=8.314462618,
        help='Universal gas constant (J/(mol*K))')
    
    parser.add_argument('-T', type=np.float64, default=298.15, help='Temperature (K)')

    parser.add_argument(
        '--j_applied', type=np.float64, default=-400.0,
        help='Applied current density (A/m^2)')
    
    parser.add_argument('--snes_atol', type=float, default=1e-06, help='SNES absolute tolerance')

    parser.add_argument('--snes_rtol', type=float, default=1e-06, help='SNES relative tolerance')

    parser.add_argument('--snes_stol', type=float, default=1e-06, help='SNES incremental tolerance')

    args, unknown = parser.parse_known_args()

    with dolfinx.common.Timer() as t:
        uh_e, uh_l, eta = solve_problem(**vars(args))
    elapsed = t.elapsed().total_seconds()

    if MPI.COMM_WORLD.rank == 0:
        print(f'Wall time for this run: {elapsed:.6e} s')

    with dolfinx.io.VTKFile(uh_e.function_space.mesh.comm, 'ue.pvd', 'w') as vtk:
        vtk.write_mesh(uh_e.function_space.mesh)
        vtk.write_function([uh_e, uh_l, eta], 0.0)

    start_point = np.array([0.0, args.H / 2])
    end_point = np.array([args.W, args.H / 2])
    points, u_values = evaluate_over_line(eta, start_point, end_point, args.N + 1)
    fig = plt.figure()
    plt.plot(points[:, 0], u_values, 'k', linewidth=2, label='$\\eta$ (FEniCS)')
    plt.grid(True)
    plt.legend()
    plt.xlabel('x[m]')
    plt.ylabel('Overpotential[V]')
    plt.show()


from pathlib import Path

export_dir = Path.cwd() / 'exports'
export_dir.mkdir(parents=True, exist_ok=True)
eta_csv = export_dir / f'fenics_eta_1st_simplified_model_Nx{args.N}_M{args.M}.csv'
np.savetxt(
    eta_csv,
    np.column_stack([points[:, 0], u_values]),
    delimiter=',',
    header='x_m,eta_V',
    comments='',
    fmt='%.10e')


print(f'[export] wrote {eta_csv}.')
