# SGUSCHI

**SGUSCHI** (Solid-Gas in Ultra Small Coexistence with Hovering Interfaces) is a fork of [SLUSCHI](https://github.com/qjhong/SLUSCHI) for simulating pure O₂ oxidation environments using the small-cell methodology. It couples the SLUSCHI Fortran MD orchestrator with a Python analysis layer: every 80 VASP MD steps, the Python layer detects and removes non-O₂ gas molecules, tracks the void fraction, and conditionally inserts new O₂ molecules based on an exponentially smoothed count. Outputs gas management and xyz data for easy analysis.

## Requirements

- **Python** ≥ 3.8 with `numpy`, `pandas`, `scipy`
- **VASP** (tested with standard and MLFF modes)
- **Fortran compiler** (gfortran or ifort) to build the SLUSCHI binary
- **Job scheduler**: tested on **Slurm** and **PBS Torque**
- Optional (postprocessing): `plotly`, `tqdm`

## Installation

```bash
# Build the Fortran orchestrator
cd src/dependencies/SLUSCHI_mod
make
chmod +x *

# Install Python dependencies
pip install numpy pandas scipy
```

## Quick Start

The `example/` directory contains a ready-to-use starting point, with empty POTCAR. See `example/note.md` for a walkthrough. The general steps are:

1. Prepare a folder containing: `POSCAR` (supercell, no O atoms or gas fraction), `POTCAR`, `INCAR`, `KPOINTS`, `job.in`, `jobsub`, `OxParams`, `CovalentRadii`. See SLUSCHI documentation for description of job.in parameters, generally recommended not to touch it too much.
2. Run `PrepareWorkplace.py` from that folder (found in `src/preprocessing/`). This creates a `{Temperature}_{SimIndex}/` directory tree.
3. Submit an initial VASP job in each `Dir_VolSearch/` subfolder (e.g. `sbatch jobsub`).
4. Submit the `OxidationMaster` job. Resubmit as needed until simulations reach sufficient length.
5. Results are written to `xyz_files/`.

## Configuration Reference

### OxParams

| Key | Description |
|-----|-------------|
| `AtomicRadiusTol` | Multiplier applied to the sum of covalent radii for bond detection |
| `O2Tol` | Target O₂ count per unit void fraction |
| `OSmoothing` | Exponential smoothing factor α for O₂ count (default 0.001; heavily history-weighted) |
| `GasRatio` | Fraction by which the x-axis is expanded to create the gas region |
| `InitO2Count` | Number of O₂ molecules placed at initialisation |
| `Temperatures` | List of simulation temperatures in K |
| `NSims` | Number of parallel simulation replicas per temperature |

### CovalentRadii

Plain text file, one entry per line: `Element = radius_in_Angstroms`. Supports `#` and `!` comments.

### INCAR (required settings)

| Tag | Value | Reason |
|-----|-------|--------|
| `IBRION` | `0` | Molecular dynamics mode |
| `ISIF` | `2` | Fixed cell shape; ions relax |
| `NSW` | `80` | Steps per SLUSCHI cycle (overridden at runtime; do not change here) |

### job.in

SLUSCHI volume-search configuration. Preconfigured settings work well.

## Architecture

SGUSCHI wraps the SLUSCHI volume-search loop. After every 80 MD steps, the Fortran binary hands control to Python, which updates the gas environment and returns:

```
SLUSCHI Fortran binary (volsearch_cont)
    └─ every 80 MD steps → python OxidationStep.py
            ├─ Reads:  POSCAR, {LatestFolder}/OUTCAR, OxParams, CovalentRadii, RateAnalysis.csv
            ├─ Calls OxidationAnalysis: gas detection, smoothing, O2 placement logic
            ├─ Writes: updated POSCAR, RateAnalysis.csv, XYZ trajectory
            └─ Returns control to Fortran for next 80 steps
```

If `OxidationStep.py` exits with a non-zero status, `volsearch_cont` halts immediately and writes a failure marker file.

## Known Limitations

1. **Material system**: Void-fraction tracking uses Zr atoms as the solid reference. The code has been tested on **cubic Zr refractory materials** (e.g. ZrC, ZrN) in a pure O₂ environment only.
2. **Structure geometry**: Cubic bulk structures only. The origin-shifting heuristic in `OxidationPreprocessing.py` assumes roughly equal inter-atom spacing; non-cubic and slab geometries are not supported.
3. **Cell orientation**: The gas void region must lie along the **x-axis** (first lattice vector). Gas fraction tracking, O₂ placement, and surface area calculations all assume this orientation.
4. **Gas addition**: Only **pure O₂** can be added. The O–O bond length is hardcoded to 1.2 Å and velocities are drawn from a Maxwell–Boltzmann distribution for two O atoms.
5. **Gas removal**: Only molecules of **2–3 atoms** are detected (`MinimumComplexity=2`, `MaximumComplexity=3`). All detected non-O₂ molecules are removed each cycle.
6. **Oxygen in base structure**: The base POSCAR must not contain oxygen atoms. This combination has not been tested.
7. **Fixed cell**: Cell shape and volume are fixed during MD (`ISIF=2`). Variable-cell MD is not supported.
8. **MD cycle length**: One Python cycle runs every **80 VASP MD steps**. This is enforced by `volsearch_cont` at runtime and cannot be changed by editing `INCAR` or `job.in` alone; the SLUSCHI script source must be modified.
9. **Elemental masses**: Velocity initialisation covers O, C, Zr, and N only. Additional elements must be manually added to the mass dictionary in `src/workflow/OxidationAnalysis.py`.
