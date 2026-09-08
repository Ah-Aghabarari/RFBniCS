# RFBniCS

RFBniCS is an open-source finite-element framework implemented in [FEniCSx](https://fenicsproject.org/) for simulating redox flow battery (RFB) half-cells.

RFBniCS solves a macro-homogeneous porous-electrode model including multicomponent species transport, ionic and electronic charge conservation, interfacial Faradaic charge transfer, and electrolyte flow.

The framework includes one-, two-, and three-dimensional formulations and provides an extensible basis for numerical studies of RFB transport and electrochemistry.

## Installation

Running any demo requires:

- FEniCSx (>= 0.10.0): See [DOLFINx binary installation](https://github.com/FEniCS/dolfinx#binary-packages) for installation options.

Generating the 3D mesh additionally requires:

- Gmsh (Python API): See [Gmsh download instructions](https://gmsh.info/#Download) for installation options.

Clone the repository with:

```bash
git clone https://github.com/Ah-Aghabarari/RFBniCS.git
cd RFBniCS
```

The easiest way to use the software is with Docker. From the RFBniCS repository directory, run:

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --rm ghcr.io/fenics/dolfinx/dolfinx:stable
```

Then, inside the Docker container, install RFBniCS with:

```bash
python3 -m pip install -e .
```

## Repository structure

```text
RFBniCS/
├── comparisons/
│   ├── pybamm/
│   ├── rfbfoam/
│   └── wang/
├── examples/
│   ├── 1D/
│   ├── 2D/
│   └── 3D/
├── src/
│   └── rfbnics/
├── LICENSE
├── README.md
└── pyproject.toml
```

## Examples

### 1D model

The 1D model represents transport and electrochemistry through the thickness of the porous electrode.

Run the galvanostatic example with:

```bash
python examples/1D/1D_galvanostatic.py
```

### 2D model

The 2D model additionally resolves electrolyte transport along the flow direction.

Run the galvanostatic example with:

```bash
python examples/2D/2D_galvanostatic.py
```

### 3D model

The 3D model includes the porous electrode and adjacent flow channels.

First, generate the mesh:

```bash
python examples/3D/mesh_3d.py
```

Then run the simulation:

```bash
python examples/3D/3D_galvanostatic.py
```

## Verification and comparison cases

Scripts used for the verification studies in the paper are available in `comparisons/`:

- [`wang/`](comparisons/wang/): comparison with Wang et al.
- [`pybamm/`](comparisons/pybamm/): comparison with PyBaMM.
- [`rfbfoam/`](comparisons/rfbfoam/): comparison with RfbFoam.

## License

RFBniCS is distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

The Wang et al. and RfbFoam repositories used in the comparison studies are licensed under the GNU General Public License v3.0. Copies of the corresponding licenses are included in their respective comparison directories.

## Funding

This project has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No 101137725.
