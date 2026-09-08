import numpy as np
import pybamm
from pathlib import Path
import time
BASE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = BASE_DIR / 'exports'
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

nn = 2048
solver_tol = 1e-06

def eval_profile(proc_var, x_array, t):

    try:
        vals = proc_var(x=x_array, t=t)
    except TypeError:
        vals = proc_var(t, x_array)

    return np.asarray(vals, dtype=float).reshape(-1)

def solve_pybamm(nn_solve):

    start_time = time.perf_counter()
    model = pybamm.BaseModel(name='1D dimensionless psi-phi-cV2-cV3')
    xi = pybamm.SpatialVariable('xi', domain=['electrode'], coord_sys='cartesian')
    psi_hat = pybamm.Variable('Dimensionless solid potential', domain='electrode')
    phi_hat = pybamm.Variable('Dimensionless electrolyte potential', domain='electrode')
    cV2_hat = pybamm.Variable('Dimensionless concentration V2+', domain='electrode', bounds=(0, np.inf))
    cV3_hat = pybamm.Variable('Dimensionless concentration V3+', domain='electrode', bounds=(0, np.inf))
    sigma = pybamm.Parameter('Solid conductivity')
    kappa = pybamm.Parameter('Electrolyte  conductivity ')
    D_V2 = pybamm.Parameter('Diffusivity V2+ [m2.s-1]')
    D_V3 = pybamm.Parameter('Diffusivity V3+ [m2.s-1]')
    a = pybamm.Parameter('Surface area per unit volume [m-1]')
    c0_V2 = pybamm.Parameter('Inlet concentration of V2+ [mol/m^3]')
    c0_V3 = pybamm.Parameter('Inlet concentration of V3+ [mol/m^3]')
    F = pybamm.Parameter('F constant ')
    j_app = pybamm.Parameter('Applied current density [A.m-2]')
    W = pybamm.Parameter('Electrode thickness [m]')
    k0 = pybamm.Parameter('Standard rate constant [m/s]')
    z = pybamm.Parameter('Number of electrons transferred in the reaction [-]')
    epsilon = pybamm.Parameter('Porosity of the porous electrode [-]')
    R = pybamm.Parameter('gas constant [j.mol-1.k-1]')
    T = pybamm.Parameter('Temperature [k]')
    E_formal = pybamm.Parameter('Formal potential [V]')
    alpha_a = pybamm.Parameter('Anodic transfer coefficient [-]')
    alpha_c = pybamm.Parameter('Cathodic transfer coefficient [-]')



    sigma = sigma * (1 - epsilon) ** 1.5
    D_V2 = D_V2 * epsilon ** 1.5
    D_V3 = D_V3 * epsilon ** 1.5
    V_scale = R * T / F

    # Characteristic concentration 
    C_scale = pybamm.Scalar(300.0)

    I_scale = pybamm.Scalar(0.5)

    j_hat = j_app / I_scale
    
    # dimensional smoothing:

    eps_hat = pybamm.Scalar(1e-14/300.0)
    cV2_eff_hat = 0.5*(cV2_hat + pybamm.sqrt(cV2_hat * cV2_hat + eps_hat * eps_hat))
    cV3_eff_hat = 0.5*(cV3_hat + pybamm.sqrt(cV3_hat * cV3_hat + eps_hat * eps_hat))
    Eeq_hat = pybamm.log((cV3_eff_hat + eps_hat) / (cV2_eff_hat + eps_hat))
    eta_hat = psi_hat - phi_hat - Eeq_hat
    cV2_eff = C_scale * cV2_eff_hat
    cV3_eff = C_scale * cV3_eff_hat




    i0 = z * F * k0 * cV2_eff ** alpha_c * cV3_eff ** alpha_a
    i_loc = i0 * (pybamm.exp(alpha_a * z * eta_hat) - pybamm.exp(-alpha_c * z * eta_hat))
    S = a * i_loc
    S_hat = S * W / I_scale
    sigma_hat = sigma * V_scale/(I_scale * W)
    kappa_hat = kappa * V_scale/(I_scale * W)
    D_V2_hat = z * F * D_V2 * C_scale/(I_scale * W)
    D_V3_hat = z * F * D_V3 * C_scale/(I_scale * W)
    is_hat = -sigma_hat * pybamm.grad(psi_hat)
    ie_hat = -kappa_hat * pybamm.grad(phi_hat)
    Nc2_hat = -D_V2_hat * pybamm.grad(cV2_hat)
    Nc3_hat = -D_V3_hat * pybamm.grad(cV3_hat)

    # governing equations
    model.algebraic = {
        psi_hat: pybamm.div(is_hat) + S_hat,
        phi_hat: pybamm.div(ie_hat) - S_hat,
        cV2_hat: -pybamm.div(Nc2_hat) - S_hat,
        cV3_hat: -pybamm.div(Nc3_hat) + S_hat,
    }
    model.rhs = {}

    # Boundary conditions
    
    model.boundary_conditions = {
        psi_hat: {
            'left': (-j_hat / sigma_hat, 'Neumann'),
            'right': (pybamm.Scalar(0), 'Neumann'),
        },
        phi_hat: {
            'left': (pybamm.Scalar(0), 'Dirichlet'),
            'right': (-j_hat / kappa_hat, 'Neumann'),
        },
        cV2_hat: {
            'left': (pybamm.Scalar(0), 'Neumann'),
            'right': (c0_V2 / C_scale, 'Dirichlet'),
        },
        cV3_hat: {
            'left': (pybamm.Scalar(0), 'Neumann'),
            'right': (c0_V3 / C_scale, 'Dirichlet'),
        },
    }

    # guess:

    cV2_hat_0 = c0_V2/ C_scale
    cV3_hat_0 = c0_V3/ C_scale
    psi_hat_0 = pybamm.Scalar(0)
    phi_hat_0 = pybamm.Scalar(0)

    model.initial_conditions = {
        psi_hat: psi_hat_0,
        phi_hat: phi_hat_0,
        cV2_hat: cV2_hat_0,
        cV3_hat: cV3_hat_0}

    #dimensionl quantities for output:
    psi = E_formal + V_scale * psi_hat
    phi = V_scale * phi_hat
    cV2 = C_scale * cV2_hat
    cV3 = C_scale * cV3_hat
    eta = V_scale * eta_hat
    Eeq = E_formal + V_scale * Eeq_hat

    model.variables = {
        'Solid potential [V]': psi,
        'Electrolyte potential [V]': phi,
        'Concentration V2+ [mol.m-3]': cV2,
        'Concentration V3+ [mol.m-3]': cV3,
        'Overpotential [V]': eta,
        'Source S [A.m-3]': S,
        'cV2+cV3 [mol.m-3]': cV2 + cV3,
        'Voltage [V]': pybamm.boundary_value(psi, 'left'),
        'Equilibrium potential [V]': Eeq,

        'Dimensionless solid potential': psi_hat,
        'Dimensionless electrolyte potential': phi_hat,
        'Dimensionless concentration V2+': cV2_hat,
        'Dimensionless concentration V3+': cV3_hat,
        'Dimensionless overpotential': eta_hat}


    param = pybamm.ParameterValues({
        'Solid conductivity': 3402.1,
        'Electrolyte  conductivity ': 5.9514,
        'Diffusivity V2+ [m2.s-1]':  8.1e-12,
        'Diffusivity V3+ [m2.s-1]': 1.65e-11,
        'Surface area per unit volume [m-1]': 23000.0,
        'Inlet concentration of V2+ [mol/m^3]' : 300.0,
        'Inlet concentration of V3+ [mol/m^3]': 300.0,
        'F constant ': 96485.33212,
        'Applied current density [A.m-2]' :  -0.5,
        'Electrode thickness [m]': 0.0025,
        'Standard rate constant [m/s]': 2.5e-07,
        'Number of electrons transferred in the reaction [-]' : 1.0,
        'Porosity of the porous electrode [-]': 0.94,
        'gas constant [j.mol-1.k-1]': 8.314462618,
        'Temperature [k]': 298.15,
        'Formal potential [V]': -0.36,
        'Anodic transfer coefficient [-]': 0.5,
        'Cathodic transfer coefficient [-]': 0.5,
    })



    geometry = {'electrode': {xi: {'min': pybamm.Scalar(0), 'max': pybamm.Scalar(1)}}}
    param.process_model(model)
    param.process_geometry(geometry)
    submesh_types = {'electrode': pybamm.Uniform1DSubMesh}

    var_pts = {xi: nn_solve}
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
    spatial_methods = {'electrode': pybamm.FiniteVolume()}
    disc = pybamm.Discretisation(mesh, spatial_methods)
    disc.process_model(model)

    # Solve
    t_eval = [0.0, 1.0]
    solver = pybamm.CasadiAlgebraicSolver(
        tol=solver_tol, step_tol=1e-10, extra_options={'max_iter': 2000, 'print_iteration': True})
    solution = solver.solve(model, t_eval)

    elapsed_time = time.perf_counter() - start_time
    print(f'nn = {nn_solve}, elapsed time = {elapsed_time:.4f} s')



    return solution, mesh, t_eval



total_start = time.perf_counter()
solution, mesh, t_eval = solve_pybamm(nn)
print(f'TOTAL elapsed time: {time.perf_counter() - total_start:.4f} s')


out_dir = EXPORT_DIR
xi_eval = np.linspace(0.0, 1.0, nn + 1)
W_value = 0.0025
x_eval = W_value * xi_eval
t_final = float(t_eval[-1])

eta_line = eval_profile(solution['Overpotential [V]'], xi_eval, t_final)
c2_line = eval_profile(solution['Concentration V2+ [mol.m-3]'], xi_eval, t_final)
c3_line = eval_profile(solution['Concentration V3+ [mol.m-3]'], xi_eval, t_final)


np.savetxt(
    out_dir / f'pybamm_2ndsimplifed_Nx{nn}_eta_vs_x.csv',
    np.column_stack([x_eval, eta_line]), delimiter=',', header='x_m,eta_V', comments='', fmt='%.10e')

np.savetxt(
    out_dir / f'pybamm_2ndsimplifed_Nx{nn}_cV2_vs_x_Nx{nn}.csv',
    np.column_stack([x_eval, c2_line]), delimiter=',', header='x_m,cV2_molm3', comments='', fmt='%.10e')


np.savetxt(
    out_dir / f'pybamm_2ndsimplifed_Nx{nn}_cV3_vs_x_Nx{nn}.csv',
    np.column_stack([x_eval, c3_line]), delimiter=',', header='x_m,cV3_molm3', comments='', fmt='%.10e')

pybamm.print_citations()
