from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import numpy.typing as npt


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
        points[:, : _points.shape[1]] = _points

    domain = u.function_space.mesh
    bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim, padding=1e-8)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points)
    # Choose one of the cells that contains the point
    colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, points)
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    points_on_proc_as_array = np.array(points_on_proc, dtype=np.float64)
    u_values = u.eval(points_on_proc_as_array, np.asarray(cells, dtype=np.int32))

    return points_on_proc, u_values


def global_component_range(X: dolfinx.fem.Function, component_index: int):
    """Return the global min and max of one component of X."""
    component = X.sub(component_index).collapse()
    global_min, global_max = global_function_range(component)
    return global_min, global_max


def global_function_range(function: dolfinx.fem.Function):
    """Return the MPI-global minimum and maximum of a scalar function."""
    mesh = function.function_space.mesh
    index_map = function.function_space.dofmap.index_map
    block_size = function.function_space.dofmap.index_map_bs
    if block_size > 1:
        raise ValueError(
            "global_function_range only supports scalar functions. "
            f"Function has block size {block_size}."
        )
    num_owned = index_map.size_local * block_size
    is_p_space = function.function_space.element.basix_element.family == basix.ElementFamily.P
    is_constant_or_linear = function.function_space.element.basix_element.degree <= 1
    if not (is_p_space and is_constant_or_linear):
        raise ValueError(
            "global_function_range only supports scalar functions with P1 or P0 elements."
        )
    values = np.asarray(function.x.array[:num_owned]).real

    if values.size > 0:
        local_min = float(np.min(values))
        local_max = float(np.max(values))
    else:
        local_min = np.inf
        local_max = -np.inf

    global_min = mesh.comm.allreduce(
        local_min,
        op=MPI.MIN,
    )

    global_max = mesh.comm.allreduce(
        local_max,
        op=MPI.MAX,
    )

    return global_min, global_max
