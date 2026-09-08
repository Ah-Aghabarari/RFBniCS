from pathlib import Path

import gmsh


def create_rounded_channel_z(x_start, x_depth, y_center, z_start, z_end, width):
    """
    Create one rounded channel in the new orientation.

    Coordinate convention, in mm before Mesh.ScalingFactor:
      x = through-plane/depth direction
      y = inlet-to-outlet direction
      z = channel/electrode length direction

    Channel:
      depth direction:  x
      width direction:  y
      length direction: z

    The inlet/outlet opening is the x = 0 face.
    """
    r = width / 2.0

    # Rectangular middle section
    box = gmsh.model.occ.addBox(
        x_start,
        y_center - r,
        z_start,
        x_depth,
        width,
        z_end - z_start,
    )

    # Rounded end caps.
    # Cylinder axis is in the x direction.
    cyl1 = gmsh.model.occ.addCylinder(
        x_start,
        y_center,
        z_start,
        x_depth,
        0.0,
        0.0,
        r,
    )

    cyl2 = gmsh.model.occ.addCylinder(
        x_start,
        y_center,
        z_end,
        x_depth,
        0.0,
        0.0,
        r,
    )

    fused, _ = gmsh.model.occ.fuse([(3, box)], [(3, cyl1), (3, cyl2)])
    return fused[0][1]


def create_fttf_model():
    gmsh.initialize()
    gmsh.model.add("RfbFoam_FTTF_Rounded_flow_x_spacing_y")

    # ---------------------------------------------------------
    # 1. Dimensions, in mm
    # ---------------------------------------------------------
    # New orientation:
    #   x: 0 ... 1.42 mm       through-plane/depth
    #   y: 0 ... 17 mm         inlet-to-outlet direction
    #   z: 0 ... 15 mm         channel/electrode length
    L_e = 15.0  # electrode/channel length in z
    W_e = 17.0  # electrode width in y
    t_e = 0.42  # electrode thickness in x

    d_ch = 1.0  # channel depth in x
    w_ch = 1.0  # channel width in y

    # Rounded channel end-cap centers in z.
    # With radius 0.5 mm, channels span z = 1.0 ... 14.0 mm.
    z_ch_start = 1.5
    z_ch_end = 13.5

    # Inlet/outlet positions in y.
    # Direction from inlet to outlet is +y.
    y_inlet = 1.0
    y_outlet = 16.0

    # x positions:
    # Channels occupy x = 0 ... d_ch.
    # Electrode occupies x = d_ch ... d_ch + t_e.
    x_channel_start = 0.0
    x_electrode_start = d_ch

    # ---------------------------------------------------------
    # 2. Create geometry using OpenCASCADE kernel
    # ---------------------------------------------------------
    electrode = gmsh.model.occ.addBox(
        x_electrode_start,
        0.0,
        0.0,
        t_e,
        W_e,
        L_e,
    )

    inlet_ch = create_rounded_channel_z(
        x_start=x_channel_start,
        x_depth=d_ch,
        y_center=y_inlet,
        z_start=z_ch_start,
        z_end=z_ch_end,
        width=w_ch,
    )

    outlet_ch = create_rounded_channel_z(
        x_start=x_channel_start,
        x_depth=d_ch,
        y_center=y_outlet,
        z_start=z_ch_start,
        z_end=z_ch_end,
        width=w_ch,
    )

    gmsh.model.occ.synchronize()

    # ---------------------------------------------------------
    # 3. Fragment volumes for conformal meshing
    # ---------------------------------------------------------
    out, out_map = gmsh.model.occ.fragment(
        [(3, electrode)],
        [(3, inlet_ch), (3, outlet_ch)],
    )
    gmsh.model.occ.synchronize()

    electrode_tags = [tag for dim, tag in out_map[0] if dim == 3]
    inlet_tags = [tag for dim, tag in out_map[1] if dim == 3]
    outlet_tags = [tag for dim, tag in out_map[2] if dim == 3]

    # ---------------------------------------------------------
    # 4. Assign physical volume groups
    # ---------------------------------------------------------
    gmsh.model.addPhysicalGroup(3, electrode_tags, name="Electrode")
    gmsh.model.addPhysicalGroup(3, inlet_tags, name="InletChannel")
    gmsh.model.addPhysicalGroup(3, outlet_tags, name="OutletChannel")

    # ---------------------------------------------------------
    # 5. Assign physical surface groups
    # ---------------------------------------------------------
    inlet_surfs = []
    outlet_surfs = []
    cc_surfs = []
    wall_surfs = []
    membrane_surfs = []

    eps = 1.0e-4

    x_inout = 0.0
    x_cc = d_ch
    x_membrane = d_ch + t_e

    for dim, tag in gmsh.model.getEntities(2):
        upward, _ = gmsh.model.getAdjacencies(dim, tag)

        # Internal electrode/channel interfaces are not external BC patches.
        if len(upward) == 2:
            continue

        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        y_center = 0.5 * (ymin + ymax)

        # Inlet/outlet openings are the x = 0 faces of the two channels.
        # Inlet flow should enter in +x.
        # Outlet flow should leave in -x.
        if abs(xmin - x_inout) < eps and abs(xmax - x_inout) < eps:
            if abs(y_center - y_inlet) < abs(y_center - y_outlet):
                inlet_surfs.append(tag)
            else:
                outlet_surfs.append(tag)

        # Current collector / rib-land patch:
        # exposed front face of electrode, x = d_ch.
        elif abs(xmin - x_cc) < eps and abs(xmax - x_cc) < eps:
            cc_surfs.append(tag)

        # Membrane patch:
        # back face of electrode, x = d_ch + t_e.
        elif abs(xmin - x_membrane) < eps and abs(xmax - x_membrane) < eps:
            membrane_surfs.append(tag)

        # Everything else:
        # side walls, rounded channel walls, electrode outer sides.
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

    # ---------------------------------------------------------
    # 6. Meshing
    # ---------------------------------------------------------
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)

    # Geometry is defined in mm; write mesh coordinates in m.
    gmsh.option.setNumber("Mesh.ScalingFactor", 1.0e-3)

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    mesh_file = Path.cwd() / "output" / "3D" / "FTTF.msh"
    mesh_file.parent.mkdir(parents=True, exist_ok=True)

    gmsh.write(str(mesh_file))
    gmsh.finalize()


if __name__ == "__main__":
    create_fttf_model()
