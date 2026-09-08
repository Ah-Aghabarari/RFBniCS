from mpi4py import MPI

import dolfinx
import ufl


def assemble_global(expression: ufl.Form) -> float:
    """Assemble a scalar UFL expression across all MPI ranks."""
    compiled_form = dolfinx.fem.form(expression)
    local_value = dolfinx.fem.assemble_scalar(compiled_form)

    return compiled_form.mesh.comm.allreduce(
        local_value,
        op=MPI.SUM,
    )
