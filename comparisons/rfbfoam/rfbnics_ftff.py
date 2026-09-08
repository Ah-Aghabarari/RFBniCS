import gmsh


def create_rounded_channel_x(x_start, x_end, z_center, y_start, depth, width):

    r = width / 2.0


    box = gmsh.model.occ.addBox(
        x_start,
        y_start,
        z_center - r,
        x_end - x_start,
        depth,
        width,
    )

    cyl1 = gmsh.model.occ.addCylinder(
        x_start,
        y_start,
        z_center,
        0.0,
        depth,
        0.0,
        r,
    )
    cyl2 = gmsh.model.occ.addCylinder(
        x_end,
        y_start,
        z_center,
        0.0,
        depth,
        0.0,
        r,
    )

    fused, _ = gmsh.model.occ.fuse([(3, box)], [(3, cyl1), (3, cyl2)])
    return fused[0][1]


def create_fttf_model():
    gmsh.initialize()
    gmsh.model.add("RfbFoam_FTTF_Rounded_negative_z")


    L_e = 15.0   
    W_e = 17.0     
    t_e = 0.42      
    d_ch = 1.0      
    w_ch = 1.0      

    x_ch_start = 1.5
    x_ch_end = 13.5

    z_inlet = -16.0
    z_outlet = -1.0


    electrode = gmsh.model.occ.addBox(
        0.0,
        0.0,
        -W_e,
        L_e,
        t_e,
        W_e,
    )


    inlet_ch = create_rounded_channel_x(
        x_start=x_ch_start,
        x_end=x_ch_end,
        z_center=z_inlet,
        y_start=t_e,
        depth=d_ch,
        width=w_ch,
    )

    outlet_ch = create_rounded_channel_x(
        x_start=x_ch_start,
        x_end=x_ch_end,
        z_center=z_outlet,
        y_start=t_e,
        depth=d_ch,
        width=w_ch,
    )

    gmsh.model.occ.synchronize()


    out, out_map = gmsh.model.occ.fragment(
        [(3, electrode)],
        [(3, inlet_ch), (3, outlet_ch)],
    )
    gmsh.model.occ.synchronize()

    electrode_tags = [tag for dim, tag in out_map[0] if dim == 3]
    inlet_tags = [tag for dim, tag in out_map[1] if dim == 3]
    outlet_tags = [tag for dim, tag in out_map[2] if dim == 3]

    gmsh.model.addPhysicalGroup(3, electrode_tags, name="Electrode")
    gmsh.model.addPhysicalGroup(3, inlet_tags, name="InletChannel")
    gmsh.model.addPhysicalGroup(3, outlet_tags, name="OutletChannel")


    inlet_surfs = []
    outlet_surfs = []
    cc_surfs = []
    wall_surfs = []
    membrane_surfs = []

    eps = 1.0e-4

    for dim, tag in gmsh.model.getEntities(2):
        upward, _ = gmsh.model.getAdjacencies(dim, tag)


        if len(upward) == 2:
            continue

        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        z_center = 0.5 * (zmin + zmax)

        if abs(ymin - (t_e + d_ch)) < eps and abs(ymax - (t_e + d_ch)) < eps:

            if abs(z_center - z_inlet) < abs(z_center - z_outlet):
                inlet_surfs.append(tag)
            else:
                outlet_surfs.append(tag)


        elif abs(ymin - 0.0) < eps and abs(ymax - 0.0) < eps:
            membrane_surfs.append(tag)


        elif abs(ymin - t_e) < eps and abs(ymax - t_e) < eps:
            cc_surfs.append(tag)

        else:
            wall_surfs.append(tag)

    if inlet_surfs:
        gmsh.model.addPhysicalGroup(2, inlet_surfs, name="Inlet")
    if outlet_surfs:
        gmsh.model.addPhysicalGroup(2, outlet_surfs, name="Outlet")
    if cc_surfs:
        gmsh.model.addPhysicalGroup(2, cc_surfs, name="CurrentCollector")
    if wall_surfs:
        gmsh.model.addPhysicalGroup(2, wall_surfs, name="Walls")
    if membrane_surfs:
        gmsh.model.addPhysicalGroup(2, membrane_surfs, name="Membrane")

    print("Physical volume counts:")
    print("  Electrode     ", len(electrode_tags))
    print("  InletChannel  ", len(inlet_tags))
    print("  OutletChannel ", len(outlet_tags))

    print("Physical surface counts:")
    print("  Inlet           ", len(inlet_surfs))
    print("  Outlet          ", len(outlet_surfs))
    print("  CurrentCollector", len(cc_surfs))
    print("  Membrane        ", len(membrane_surfs))
    print("  Walls           ", len(wall_surfs))


    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.002)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)

    gmsh.option.setNumber("Mesh.ScalingFactor", 1.0e-3)

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    gmsh.write("RfbFoam_FTTF_Rounded_new.msh")
    gmsh.finalize()


if __name__ == "__main__":
    create_fttf_model()








import argparse
import time
import csv

from mpi4py import MPI
import basix.ufl
import dolfinx
import dolfinx.common
import dolfinx.fem.petsc
import numpy as np
import ufl

start_time = time.perf_counter()

def solve_flow_fttf():

    mesh_data = dolfinx.io.gmsh.read_from_msh("RfbFoam_FTTF_Rounded_new.msh",comm=MPI.COMM_WORLD,rank=0)
    mesh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags
    tag_map = mesh_data.physical_groups

    scalar_type = dolfinx.default_scalar_type


    L = 100e-6          
    te = 420e-6         
    we = 15e-3          
    Q = 1e-12           
    v_e = 1e-7          
    rho = 1015.0        
    eps = 0.877         
    nu = 1.126e-6       
    mu = rho * nu       
    K = 1.0 / 9.18e10   


    beta = 1.75 * (1.0 - eps) / (L * eps**3)

 
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)

    fluid_tags = (tag_map["InletChannel"].tag, tag_map["OutletChannel"].tag)
    dxF = dx(fluid_tags)

    electrode_tags = (tag_map["Electrode"].tag,)
    dxE = dx(electrode_tags)


    v_el = basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
    p_el = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)

    V = dolfinx.fem.functionspace(mesh, v_el)
    Q_space = dolfinx.fem.functionspace(mesh, p_el)
    W = ufl.MixedFunctionSpace(V, Q_space)

    vs = dolfinx.fem.Function(V)
    ps = dolfinx.fem.Function(Q_space)
    wh = [vs, ps]


    epsilon = dolfinx.fem.Constant(mesh, scalar_type(eps))
    rho_c = dolfinx.fem.Constant(mesh, scalar_type(rho))
    mu_c = dolfinx.fem.Constant(mesh, scalar_type(mu))
    K_c = dolfinx.fem.Constant(mesh, scalar_type(K))
    beta_c = dolfinx.fem.Constant(mesh, scalar_type(beta))


    f_dim = dolfinx.fem.Constant( mesh, np.zeros(mesh.geometry.dim, dtype=scalar_type))


    inlet_tag = tag_map["Inlet"].tag
    inlet_dofs = dolfinx.fem.locate_dofs_topological( V, facet_tags.dim, facet_tags.find(inlet_tag))

    inlet_velocity = v_e  # [m/s]

    flow_component = 1

    def u_inlet(x):
        output = np.zeros((mesh.geometry.dim, x.shape[1]), dtype=scalar_type)
        output[flow_component] = -inlet_velocity
        return output

    u_bc = dolfinx.fem.Function(V)
    u_bc.interpolate(u_inlet)
    bc_inlet = dolfinx.fem.dirichletbc(u_bc, inlet_dofs)

    wall_dofs = dolfinx.fem.locate_dofs_topological(
        V,
        facet_tags.dim,
        facet_tags.find(tag_map["Walls"].tag),
    )

    outlet_dofs = dolfinx.fem.locate_dofs_topological(
        V,
        facet_tags.dim,
        facet_tags.find(tag_map["Outlet"].tag),
    )


    wall_dofs = np.setdiff1d(wall_dofs, inlet_dofs)
    wall_dofs = np.setdiff1d(wall_dofs, outlet_dofs)

    u_walls = dolfinx.fem.Function(V)
    u_walls.interpolate(
        lambda x: np.zeros(
            (mesh.geometry.dim, x.shape[1]),
            dtype=scalar_type,
        )
    )
    bc_walls = dolfinx.fem.dirichletbc(u_walls, wall_dofs)

    cc_dofs = dolfinx.fem.locate_dofs_topological(
        V,
        facet_tags.dim,
        facet_tags.find(tag_map["CurrentCollector"].tag),
    )
    u_cc = dolfinx.fem.Function(V)
    u_cc.interpolate(
        lambda x: np.zeros(
            (mesh.geometry.dim, x.shape[1]),
            dtype=scalar_type,
        )
    )
    bc_cc = dolfinx.fem.dirichletbc(u_cc, cc_dofs)

    membrane_dofs = dolfinx.fem.locate_dofs_topological(
        V,
        facet_tags.dim,
        facet_tags.find(tag_map["Membrane"].tag),
    )
    u_membrane = dolfinx.fem.Function(V)
    u_membrane.interpolate(
        lambda x: np.zeros(
            (mesh.geometry.dim, x.shape[1]),
            dtype=scalar_type,
        )
    )
    bc_membrane = dolfinx.fem.dirichletbc(u_membrane, membrane_dofs)

    bcs = [bc_inlet, bc_walls, bc_membrane, bc_cc]


    v_test, p_test = ufl.TestFunctions(W)


    F_init = (
        -ufl.div(vs) * p_test * (dxF + dxE)
        + (mu_c / epsilon) * ufl.inner(ufl.grad(vs), ufl.grad(v_test)) * dxE
        + mu_c * ufl.inner(ufl.grad(vs), ufl.grad(v_test)) * dxF
        + (mu_c / K_c) * ufl.dot(vs, v_test) * dxE
        - ps * ufl.div(v_test) * (dxF + dxE)
    )

    F_init -= ufl.dot(f_dim, v_test) * (dxF + dxE)


    F_init += dolfinx.fem.Constant(mesh, scalar_type(0.0)) * p_test * dx(929332)

    w = ufl.TrialFunctions(W)
    a_init, L_init = ufl.system(
        ufl.replace(F_init, {vs: w[0], ps: w[1]})
    )

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

    tol_speed_sq = dolfinx.fem.Constant(
        mesh,
        scalar_type((1e-12)**2),
    )
    speed = ufl.sqrt(ufl.dot(vs, vs) + tol_speed_sq)


    F = (rho_c / epsilon**2) * ufl.dot(ufl.dot(ufl.grad(vs), vs), v_test) * dxE


    F += rho_c * ufl.dot(ufl.dot(ufl.grad(vs), vs), v_test) * dxF






    F += (
        -ufl.div(vs) * p_test * (dxF + dxE)
        - ps * ufl.div(v_test) * (dxF + dxE)
    )

    F += (
        (mu_c / epsilon)
        * ufl.inner(ufl.grad(vs), ufl.grad(v_test))
        * dxE
    )

    F += (
        mu_c
        * ufl.inner(ufl.grad(vs), ufl.grad(v_test))
        * dxF
    )

    F += (
        (mu_c / K_c)
        * ufl.dot(vs, v_test)
        * dxE
    )

    F += (
        rho_c
        * beta_c
        * ufl.dot(speed * vs, v_test)
        * dxE
    )

    F -= ufl.dot(f_dim, v_test) * (dxF + dxE)

    J = ufl.derivative(F, wh, w)

    gamma = dolfinx.fem.Constant(
        mesh,
        scalar_type(1.0e-3 * mu),
    )

    J_regularized = ufl.extract_blocks(
        gamma
        * ufl.inner(ufl.div(w[0]), ufl.div(v_test))
        * (dxF + dxE)
        + J
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


    return mesh, cell_tags, facet_tags, tag_map, v_out

















def solve_problem(
    sigma: np.float64,
    kappa: np.float64,
    s: np.float64,
    alpha_A: np.float64, 
    alpha_C: np.float64,
    kmR: np.float64,
    kmO: np.float64,
    j0: np.float64,
    Eeq: np.float64,
    F: np.float64,
    R: np.float64,
    T: np.float64,
    j_applied: np.float64,
    D_V2: np.float64,
    D_V3: np.float64,
    c0_V2: np.float64,
    c0_V3: np.float64,
    c_Ref: np.float64,
    z_V2: np.float64,
    z_V3: np.float64,
    z: np.float64,
    nu: np.float64,
    rho: np.float64,
    epsilon: np.float64,
    tau: np.float64,
    degree: int,
    snes_atol: float,
    snes_rtol: float,
    snes_stol: float,
):


    mesh, cell_tags, facet_tags, tag_map, u_flow = solve_flow_fttf()
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
    _kappa = dolfinx.fem.Constant(mesh, kappa)
    _s = dolfinx.fem.Constant(mesh, s)
    _alpha_a = dolfinx.fem.Constant(mesh, alpha_A)
    _alpha_c = dolfinx.fem.Constant(mesh, alpha_C)
    #_kmR = dolfinx.fem.Constant(mesh, kmR)
    #_kmO = dolfinx.fem.Constant(mesh, kmO)
    _j0 = dolfinx.fem.Constant(mesh, j0)
    _Eeq = dolfinx.fem.Constant(mesh, Eeq)
    _F = dolfinx.fem.Constant(mesh, F)
    _R = dolfinx.fem.Constant(mesh, R)
    _T = dolfinx.fem.Constant(mesh, T)
    _ja = dolfinx.fem.Constant(mesh, j_applied)
    _D_V2 = dolfinx.fem.Constant(mesh, D_V2) 
    _D_V3 = dolfinx.fem.Constant(mesh, D_V3)
    _c0_V2 = dolfinx.fem.Constant(mesh, c0_V2)
    _c0_V3 = dolfinx.fem.Constant(mesh, c0_V3)
    _c_Ref = dolfinx.fem.Constant(mesh, c_Ref)
    _z_V2 = dolfinx.fem.Constant(mesh, z_V2)
    _z_V3 = dolfinx.fem.Constant(mesh, z_V3)
    _z = dolfinx.fem.Constant(mesh, z)
    _nu = dolfinx.fem.Constant(mesh, nu)
    _rho = dolfinx.fem.Constant(mesh, rho)
    _eps = dolfinx.fem.Constant(mesh, epsilon)
    _tau = dolfinx.fem.Constant(mesh, tau)



    u_mag = ufl.sqrt(ufl.dot(u_flow, u_flow))

    _kmR = 0.000775 * (u_mag + 1.0e-20) ** 0.9818 + 1.0e-20
    _kmO = 0.000775 * (u_mag + 1.0e-20) ** 0.9818 + 1.0e-20
    


    
    

    el = basix.ufl.element("Lagrange", mesh.basix_cell(), degree)
    me = basix.ufl.blocked_element(el, shape=(4,))
    V = dolfinx.fem.functionspace(mesh, me)
    X = dolfinx.fem.Function(V)
    phi_e, phi_l, c_V2, c_V3 = ufl.split(X)
    v_e,  v_l,  v_c_V2,  v_c_V3= ufl.TestFunctions(V)
    
    Q0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    mask_E = dolfinx.fem.Function(Q0, name="mask_E")
    mask_E.x.array[:] = 0.0
    mask_E.x.array[cell_tags.find(tag_map["Electrode"].tag)] = 1.0

    mask_F = dolfinx.fem.Function(Q0, name="mask_F")
    mask_F.x.array[:] = 1.0 - mask_E.x.array


    sigma_eff = mask_E *  _sigma + mask_F * dolfinx.fem.Constant(mesh, 1e-11) 
    s_eff = mask_E * _s
    kappa_eff = mask_E * (_kappa * _eps / _tau) + mask_F * _kappa
    D_eff_V2 = mask_E * (_D_V2 * _eps / _tau) + mask_F * _D_V2
    D_eff_V3 = mask_E * (_D_V3 * _eps / _tau) + mask_F * _D_V3
    
    """

    sigma_eff = mask_E *  ((1.0 - _eps)**1.5 * _sigma) + mask_F * dolfinx.fem.Constant(mesh, 1e-11) #RfbFoam used 1e-11
    s_eff = mask_E * _s
    kappa_eff = mask_E * (_kappa ) + mask_F * _kappa
    D_eff_V2 = mask_E * ((_eps)**1.5 * _D_V2)  + mask_F * _D_V2
    D_eff_V3 = mask_E * ((_eps)**1.5 * _D_V3)  + mask_F * _D_V3

    #D_eff_V2 = mask_E * (_D_V2 * _eps**1.5) + mask_F * _D_V2
    #D_eff_V3 = mask_E * (_D_V3 * _eps**1.5) + mask_F * _D_V3

    """


    """
    # Concentrations and Butler–Volmer term:

    eps_c = dolfinx.fem.Constant(mesh, 1e-12)
    c_V2_eff = 0.5 * (c_V2 + ufl.sqrt(c_V2 * c_V2 + eps_c * eps_c))
    c_V3_eff = 0.5 * (c_V3 + ufl.sqrt(c_V3 * c_V3 + eps_c * eps_c))
    """

    eps_c = dolfinx.fem.Constant(mesh, 1e-12)

    c_floor = dolfinx.fem.Constant(mesh, 1e-8)

    c_V2_eff = 0.5 * (c_V2 + ufl.sqrt(c_V2 * c_V2 + c_floor * c_floor)) + c_floor
    c_V3_eff = 0.5 * (c_V3 + ufl.sqrt(c_V3 * c_V3 + c_floor * c_floor)) + c_floor



    
    
    i0 = _j0
    b = 0.5 * _F / (_R * _T)



    theta_nernst = dolfinx.fem.Constant(mesh, 0.0)

    Eeq_bulk = (_Eeq+ theta_nernst* (_R * _T / (_z * _F))* ufl.ln((c_V3_eff + eps_c) / (c_V2_eff + eps_c)))

    eta = phi_e - phi_l - Eeq_bulk

    '''
    Eeq_bulk = _Eeq + (_R*_T/(_z*_F)) * ufl.ln((c_V3_eff + eps_c)/(c_V2_eff + eps_c))
    eta = phi_e - phi_l - Eeq_bulk
    '''




    A = ufl.exp(_alpha_a * _F / (_R * _T) * eta)
    C = ufl.exp(-_alpha_c * _F / (_R * _T) * eta)


    num = (_j0 / _c_Ref) * (c_V2_eff * A - c_V3_eff * C)
    den = 1.0 + (_j0 / (_z * _F * _c_Ref)) * (A / _kmR + C / _kmO)



    i_loc = num / den

    lambda_rxn = dolfinx.fem.Constant(mesh, 0.0)

    f = lambda_rxn * s_eff * i_loc

    """

    
    #i0 = _z * _F * _k0 * c_V2_eff**_alpha * (c_V3_eff)**(1.0 - _alpha)
    #i0 = 1.0 * 96485.33 * 1.75e-7 * c_V2_eff**0.5 * (c_V3_eff)**(0.5)
    i0 = _F * dolfinx.fem.Constant(mesh, 1.75e-7) * ufl.sqrt(c_V2_eff) * ufl.sqrt(c_V3_eff)
    b = 0.5 * _F / (_R * _T)
    theta_nernst = dolfinx.fem.Constant(mesh, 0.0)

    Eeq_bulk = (_Eeq+ theta_nernst* (_R * _T / (_z * _F))* ufl.ln((c_V3_eff + eps_c) / (c_V2_eff + eps_c)))

    eta = phi_e - phi_l - Eeq_bulk
    #eta = phi_e - phi_l - _Eeq
    lambda_rxn = dolfinx.fem.Constant(mesh, 0.0)
    f = lambda_rxn * 2.0 * s_eff * i0 * ufl.sinh(b * eta)
    """
    
    #f = 2.0 * s_eff * i0 * ufl.sinh(b * eta)
    """

    # Previous-step concentrations (c_V2^n, c_V3^n):
    Vc_V2, _ = V.sub(2).collapse()
    c_V2_n = dolfinx.fem.Function(Vc_V2)
    c_V2_n.name = "c_V2_n"
    c_V2_n.interpolate(lambda x: np.full(x.shape[1], float(_c0_V2)))

    Vc_V3, _ = V.sub(3).collapse()
    c_V3_n = dolfinx.fem.Function(Vc_V3)
    c_V3_n.name = "c_V3_n"
    c_V3_n.interpolate(lambda x: np.full(x.shape[1], float(_c0_V3)))
    """



    




    c_V2_init = dolfinx.fem.Expression(_c0_V2, V.sub(2).element.interpolation_points)
    X.sub(2).interpolate(c_V2_init)

    c_V3_init = dolfinx.fem.Expression(_c0_V3, V.sub(3).element.interpolation_points)
    X.sub(3).interpolate(c_V3_init)
    
    

    phi_e_init = dolfinx.fem.Expression(dolfinx.fem.Constant(mesh, 0.0),V.sub(0).element.interpolation_points)
    X.sub(0).interpolate(phi_e_init)

    phi_l_init = dolfinx.fem.Expression(dolfinx.fem.Constant(mesh, -0.76),V.sub(1).element.interpolation_points)
    X.sub(1).interpolate(phi_l_init)




    



    F = (
        ufl.inner(sigma_eff * ufl.grad(phi_e), ufl.grad(v_e)) * dx
        + ufl.inner(f, v_e) * dxE
    )

    



    F += (
        ufl.inner(kappa_eff * ufl.grad(phi_l), ufl.grad(v_l)) * dx
        - ufl.inner(f, v_l) * dxE
    )


    '''


    F += (
        +ufl.inner(kappa * ufl.grad(phi_l), ufl.grad(v_l)) * ufl.dx
        - ufl.inner(f, v_l) * ufl.dx
    )

    '''



    F += (
      + ufl.inner(D_eff_V2 * ufl.grad(c_V2), ufl.grad(v_c_V2)) * dx
      - ufl.inner(u_flow * c_V2, ufl.grad(v_c_V2)) * dx
      + (1.0 / (_z * _F)) * f * v_c_V2 * dxE 
    )


    F += (
      + ufl.inner(D_eff_V3 * ufl.grad(c_V3), ufl.grad(v_c_V3)) * dx
      - ufl.inner(u_flow * c_V3, ufl.grad(v_c_V3)) * dx
      - (1.0 / (_z * _F)) * f * v_c_V3 * dxE 
    )

   





    """
    # Neumann boundary terms
    F -= _ja * v_e * ds(cc_tag)
    F += _ja * v_l * ds(mem_tag)
    """



    F += (
        + ufl.dot(u_flow, n_vec) * c_V2   * v_c_V2   * ds(outlet_tag)
        + ufl.dot(u_flow, n_vec) * c_V3   * v_c_V3   * ds(outlet_tag)
    )


    inlet_facets = facet_tags.find(inlet_tag).astype(np.int32)

    c_V2_bc_dofs = dolfinx.fem.locate_dofs_topological(V.sub(2), facet_tags.dim, inlet_facets)
    c_V3_bc_dofs = dolfinx.fem.locate_dofs_topological(V.sub(3), facet_tags.dim, inlet_facets)

    bc_c_V2_in = dolfinx.fem.dirichletbc(_c0_V2, c_V2_bc_dofs, V.sub(2))
    bc_c_V3_in = dolfinx.fem.dirichletbc(_c0_V3, c_V3_bc_dofs, V.sub(3))




    cc_facets = facet_tags.find(cc_tag).astype(np.int32)
    mem_facets = facet_tags.find(mem_tag).astype(np.int32)

    phi_e_cc_dofs = dolfinx.fem.locate_dofs_topological(V.sub(0), facet_tags.dim, cc_facets)
    phi_l_mem_dofs = dolfinx.fem.locate_dofs_topological(V.sub(1), facet_tags.dim, mem_facets)

    phi_e_cc_value = dolfinx.fem.Constant(mesh, 0.0)    # Phi1 = 0 at current collector
    #phi_l_mem_value = dolfinx.fem.Constant(mesh, -1.0)  # Phi2 = -1 at membrane

    phi_l_mem_value = dolfinx.fem.Constant(mesh,np.float64(-1.0))
    

    bc_phi_e_cc = dolfinx.fem.dirichletbc(phi_e_cc_value, phi_e_cc_dofs, V.sub(0))
    bc_phi_l_mem = dolfinx.fem.dirichletbc(phi_l_mem_value, phi_l_mem_dofs, V.sub(1))


    

    

    petsc_options = {
        "snes_error_if_not_converged": True,
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_atol": snes_atol,
        "snes_rtol": snes_rtol,
        "snes_stol": snes_stol,
        "snes_max_it": 100,
        # Line-search diagnostics and controls
        #"snes_linesearch_monitor": None,
        #"snes_linesearch_max_it": 50,
        #"snes_linesearch_minlambda": 1e-16,
        
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
        F,
        X,
        bcs=[bc_c_V2_in, bc_c_V3_in, bc_phi_e_cc, bc_phi_l_mem],
        petsc_options=petsc_options,
        petsc_options_prefix="solver",
        jit_options=jit_options,
    )









    with dolfinx.io.VTKFile(mesh.comm, "electrochem_scalars.pvd", "w") as vtk_s, \
         dolfinx.io.VTKFile(mesh.comm, "flow_velocity.pvd", "w") as vtk_u:
    

        theta_nernst.value = np.float64(0.0)
    
        reaction_steps = np.concatenate((np.array([0.0]),np.geomspace(1e-8, 1.0, 4)))


    
        for lam in reaction_steps:
            lambda_rxn.value = np.float64(lam)
    
            if mesh.comm.rank == 0:
                print(
                    f"Stage 1: lambda = {lam:g}, theta_nernst = 0",
                    flush=True,
                )
    
            problem.solve()


        
    
        lambda_rxn.value = np.float64(1.0)
    
        nernst_steps = np.linspace(0.0, 1.0, 4)
    
        for theta in nernst_steps:
            theta_nernst.value = np.float64(theta)
    
            if mesh.comm.rank == 0:
                print(
                    f"Stage 2: lambda = 1, theta_nernst = {theta:g}",
                    flush=True,
                )
    
            problem.solve() 

        
    
        uh_e = X.sub(0).collapse(); uh_e.name = "phi_e"
        uh_l = X.sub(1).collapse(); uh_l.name = "phi_l"
        uh_c2 = X.sub(2).collapse(); uh_c2.name = "c_V2"
        uh_c3 = X.sub(3).collapse(); uh_c3.name = "c_V3"

    
        eta_out = dolfinx.fem.Function(uh_e.function_space)
        eta_expr = dolfinx.fem.Expression(
            eta,
            uh_e.function_space.element.interpolation_points,
        )
        eta_out.interpolate(eta_expr)
        eta_out.name = "eta"
    
        vtk_s.write_function([uh_e, uh_l, eta_out, uh_c2, uh_c3], 0.0)
        vtk_u.write_function(u_flow, 0.0)

   
    return {
        "uh_e": uh_e,
        "uh_l": uh_l,
        "uh_c_V2": uh_c2,
        "uh_c_V3": uh_c3,
        "eta": eta_out,
        "u": u_flow,
    }









    



if __name__ == "__main__":



    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--degree", type=int, default=1, help="Finite element degree")
    parser.add_argument(
        "--sigma",
        type=np.float64,
        default=275.78, # in RfbFoam
        help="Electrical conductivity of porous electrode (S/m)",
    )

    parser.add_argument(
        "--kappa",
        type=np.float64,
        default=34.84,
        help="Ionic conductivity of electrolyte (S/m)",
    )

    parser.add_argument(
        "-s",
        type=np.float64,
        default=68010.0,
        help="Specific surface area (m^(-1))",
    )
    parser.add_argument(
        "--alpha_A",
        type=np.float64,
        default=0.5,
        help="Anodic charge transfer coefficient (-)",
    )

    parser.add_argument(
        "--alpha_C",
        type=np.float64,
        default=0.5,
        help="Cathodic charge transfer coefficient (-)",
    )

    parser.add_argument(
        "--kmR",
        type=np.float64,
        default=1.0,
        help="Mass transfer coefficients of reductant (m/s)",
    )

    parser.add_argument(
        "--kmO",
        type=np.float64,
        default=1.0,
        help="Mass transfer coefficients of oxidant (m/s)",
    )

    parser.add_argument(
        "--j0", type=np.float64, default=165.0 , help="Exchange current density (A/m^2)"
    )
    parser.add_argument(
        "--Eeq", type=np.float64, default=0.771 , help="Equilibrium potential (V)"
    )
    
    parser.add_argument(
        "-F", type=np.float64, default=96485.33212, help="Faraday's constant (C/mol)"
    )
    parser.add_argument(
        "-R",
        type=np.float64,
        default=8.314462618,
        help="Universal gas constant (J/(mol*K))",
    )
    parser.add_argument(
        "-T", type=np.float64,
        default=293.5, help="Temperature (K)",
        )
    parser.add_argument(
        "--j_applied",
        type=np.float64,
        default=-4.0,
        help="Applied current density (A/m^2)",
    )
    parser.add_argument(
        "--D_V2",
        type=np.float64,
        default=5.7e-10, # 4.8e-10 / 0.877^1.5 = 5.84e-10 in RfbFoam
        help="Diffusivity of V2+ (m^2/s)",
    )
    parser.add_argument(
        "--D_V3",
        type=np.float64,
        default=4.8e-10,  # 0.877^1.5 = 6.94e-10 in RfbFoam
        help="Diffusivity of V3+ (m^2/s)",
    )

    parser.add_argument(
        "--c0_V2",
        type=np.float64,
        default=250.0,
        help="Inlet concentration of V2+ (mol/m^3)",
    )

    parser.add_argument(
        "--c0_V3",
        type=np.float64,
        default=250.0,
        help="Inlet concentration of V3+ (mol/m^3)",
    )

    parser.add_argument(
        "--c_Ref",
        type=np.float64,
        default=250.0,
        help="Reference density, quanitity used to non-dim BV, (mol/m^3)",
    )
    

    # Charges
    parser.add_argument(
        "--z_V2", type=np.float64, default=2.0, help="Charge number of V2+"
    )
    parser.add_argument(
        "--z_V3", type=np.float64, default=3.0, help="Charge number of V3+"
    )

    parser.add_argument(
        "--z", type=np.float64, default=1.0, help="Number of electrons transferred in the reaction [-]"
    )
    parser.add_argument(
        "--nu", type=np.float64, default=1.126e-6, help="Kinematic viscosity (m^2/s)"  #1.143e-3
    )
    parser.add_argument(
        "--rho", type=np.float64, default=1015.0, help="Fludid density (kg/m^3)"  
    )

    parser.add_argument(
        "--epsilon",
        type=np.float64,
        default=0.877,
        help="Porosity of the porous electrode (-)",
    )

    parser.add_argument(
        "--tau",
        type=np.float64,
        default=1.42,
        help="Tortuosity (-)",
    )

    parser.add_argument(
        "--snes_atol", type=float, default=1e-7, help="SNES absolute tolerance"
    )
    parser.add_argument(
        "--snes_rtol", type=float, default=1e-7, help="SNES relative tolerance"
    )
    parser.add_argument(
        "--snes_stol", type=float, default=1e-7, help="SNES incremental tolerance"
    )
    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    res = solve_problem(**vars(args))
    




    eta_fn = res["eta"]
    

end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"TOTAL Elapsed time: {elapsed_time:.4f} seconds")



