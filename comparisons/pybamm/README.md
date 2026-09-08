# PyBaMM comparison

This directory contains the scripts used for the one-dimensional excess-supporting-electrolyte comparison reported in the manuscript.

## Files

### `reference_bvp.py`

Computes a high-accuracy reference solution using
`scipy.integrate.solve_bvp`.

It exports the overpotential and V2+/V3+ concentration profiles used as the benchmark.

### `pybamm_ese_1d.py`

Implements the same reduced 1D problem using a custom `pybamm.BaseModel` and PyBaMM's finite-volume discretization.

### `rfbnics_ese_1d.py`

Implements the same problem using RFBniCS/FEniCSx.

## Running the comparison

Run the three scripts separately and compare the generated profiles in the `exports/` directory.
