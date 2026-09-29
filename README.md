# SGUSCHI

**SGUSCHI** (Solid-Gas in Ultra Small Coexistence with Hovering Interfaces) is a fork of [SLUSCHI](https://github.com/qjhong/SLUSCHI) for simulating pure O₂ oxidation environments using the small-cell methodology. It couples the SLUSCHI cluster Fortran MD orchestrator with a Python analysis layer: every 80 VASP MD steps, the Python layer detects and removes non-O₂ gas molecules, tracks the void fraction, and conditionally inserts new O₂ molecules based on an exponentially smoothed count. Outputs gas management CSVs and xyz data for easy analysis.

<p align="center">
  <img src="docs/Visual_Abstract_git.png" width="85%" alt="Visual abstract">
</p>
Paper can be found at (insert doi here)

Insert lisence here



## Requirements

- **Python** ≥ 3.8 with `numpy`, `pandas`, `scipy`
- **VASP** licensed and working installation (tested with standard and MLFF modes)
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

# Optional: development tools (tests, formatting, linting)
pip install pytest black ruff
```

## Testing

The test suite lives in `src/test/` and must be run from that directory (`conftest.py`
adds `src/` to `sys.path`):

```bash
cd src/test
pytest                                  # full suite
pytest TestGasWorkflow.py               # a single module
pytest TestVaspioIo.py::TestName        # a single class
python RunTests.py                      # alternative runner
```

`TestPressureAvg.py` compiles the Fortran orchestrator and is auto-skipped if no
`ifort`/`ifx`/`gfortran` is on `PATH`.

## Quick Start

The `example/` directory contains a ready-to-use starting point (with empty POTCAR). See `example/note.md` for the quick-start steps and per-file customisation notes. The steps are:

1. Copy the `example/` folder to your workspace and populate it:
   `POSCAR` (supercell, no O atoms, no gas region), `POTCAR`, `INCAR`, `KPOINTS`,
   `job.in`, `jobsub`, `OxParams`, `CovalentRadii`.
2. Customise `OxidationMaster`: set the `#SBATCH` tags, `module load` lines for your
   cluster, and the path to `SGUSCHI.py`.
3. Submit and resubmit as needed:
   ```bash
   sbatch OxidationMaster
   ```
   `SGUSCHI.py` handles everything automatically: folder creation, initial VASP job
   submission, and running `volsearch_cont` in all simulation directories.
4. Results are written to `xyz_files/`.

> **Scheduler command:** `vaspcmd` in `job.in` controls how VASP jobs are submitted
> (e.g. `vaspcmd = sbatch` for Slurm, `vaspcmd = qsub` for PBS Torque). `SGUSCHI.py`
> reads this key to submit the initial VASP job in each `Dir_VolSearch` — set it to
> match your cluster scheduler before running.

> **Inspecting / preparing without submitting:** `SGUSCHI.py` accepts two flags useful
> on a login node:
> - `--dry-run` — print the new/pending/done classification and what *would* be done,
>   without creating folders or submitting anything.
> - `--prepare-only` — create the simulation folder trees (templated `INCAR`/`job.in`,
>   per-folder `POSCAR`, `Dir_VolSearch`/`Dir_OptUnitCell`, etc.) but do **not** submit
>   VASP jobs or launch `volsearch_cont`. Safe to run on a login node; idempotent
>   (existing folders are skipped).
>
> ```bash
> python src/SGUSCHI.py [WorkDir] --prepare-only
> ```

## Configuration Reference

### OxParams

For a single workspace, all scientific keys below are **required** except
`MaxRuntime`. `JobSpecs` is optional and disabled by default.

| Key | Description |
|-----|-------------|
| `JobSpecs` | *(optional; default `[]`)* List of child workspace directories to control together. An absent or empty list keeps the normal single-workspace behavior. See below. |
| `Temperatures` | List of simulation temperatures in K |
| `NSims` | Number of parallel simulation replicas per temperature |
| `GasRatio` | Fraction by which the x-axis is expanded to create the gas region |
| `InitO2Count` | Number of O₂ molecules placed at initialisation |
| `AtomicRadiusTol` | Multiplier applied to the sum of covalent radii for bond detection |
| `O2Tol` | Target O₂ count per unit void fraction |
| `OSmoothing` | Exponential smoothing factor α for O₂ count (default 0.001; heavily history-weighted) |
| `MaxRuntime` | *(optional)* Stop simulation after this many ps of simulated time. If unset, runs until convergence. |

> **Resuming after `MaxRuntime`:** To extend a time-capped simulation, just raise `MaxRuntime` in `OxParams` and resubmit `OxidationMaster`. On startup `SGUSCHI.py` compares each time-capped run's achieved runtime (last `Time (fs)` in `RateAnalysis.csv`) against the new cap and, when it is now below it, automatically clears `volsearch_is_done` + `maxruntime_reached` so the run continues — no need to delete markers by hand. A run that reached the cap is only reopened by *raising* it; resubmitting with an unchanged (or lower) `MaxRuntime` leaves it done. Naturally-converged runs (which have `volsearch_is_done` but no `maxruntime_reached`) are never reopened. (`SGUSCHI.py` also clears `job.exit`/`job.killed`/`sguschi_failed` on resubmit.)

### Multiple job specifications with one controller

To run different starting compositions or O₂-control settings together, put this
in the campaign's top-level `OxParams` (one line):

```python
JobSpecs = ["jobs/ZrC_low", "jobs/ZrC_high", "jobs/ZrN_low"]
```

This is an opt-in list, not a boolean. Leave it out or use `JobSpecs = []` to
disable it. When enabled, the top-level file only needs `JobSpecs`; scientific
settings in that file are ignored. Each listed workspace has its own complete
inputs, without inheritance from the campaign directory:

```text
campaign/
├── OxidationMaster
├── OxParams                     # JobSpecs list
└── jobs/
    ├── ZrC_low/
    │   ├── OxParams             # Temperatures, NSims, gas settings, MaxRuntime
    │   ├── POSCAR, POTCAR, INCAR, KPOINTS
    │   ├── job.in, jobsub, CovalentRadii
    │   ├── 873_1/Dir_VolSearch/  # Generated by SGUSCHI
    │   └── xyz_files/           # This specification's trajectories
    ├── ZrC_high/
    │   └── ...
    └── ZrN_low/
        └── ...
```

Use paths relative to the campaign, preferably with forward slashes. Directory
basenames are specification IDs and must be unique (ignoring case), using
letters, digits, dots, underscores or hyphens, starting with a letter or digit.
Directories must stay inside the campaign and cannot overlap. Nested `JobSpecs`
lists are not supported; omit the key or leave it empty in each child.
The campaign's `SimulationSummary`, `.simulation_summary`, and `logs` locations
are reserved for controller output and cannot contain specification workspaces.

Run the usual commands from the campaign directory:

```bash
python /path/to/SGUSCHI/src/SGUSCHI.py . --dry-run
python /path/to/SGUSCHI/src/SGUSCHI.py . --prepare-only
sbatch OxidationMaster
```

Every specification is checked before any jobs are submitted. Setup uses its
own inputs, and one controller launches all pending simulations concurrently.
For example, `ZrC_low/873_1` and `ZrN_low/873_1` have separate state and output.
Scheduler job names include the specification ID. A failed initial submission
does not prevent other specifications from launching; resubmission retries the
rejected first job without duplicating jobs already recorded as submitted.

After preparation or a normal launch, the campaign's `SimulationSummary.py`
output includes all selected specifications using the generated
`.simulation_summary/expected.tsv`. Local trajectory filenames remain unchanged,
so repair and trajectory-resume tools can be used with each specification's
directory as their workspace. Removing a specification from the list leaves its
files untouched and removes it from the campaign's expected run list.

**Resuming grouped runs:** the controller saves `.sguschi_inputs.json` in each
specification directory on first preparation. On later invocations it rejects
changes to scientific `OxParams` settings, the contents of `POSCAR`, `POTCAR`,
`INCAR`, `KPOINTS`, `CovalentRadii`, or scientific `job.in` settings. Create a new
specification directory containing just the new input files for those changes;
do not copy old trajectories or the input record. Adding temperatures/replicas,
changing `MaxRuntime`, and changing scheduler submission settings (`jobsub` or
`vaspcmd`) are allowed. Existing run inputs are not overwritten during setup;
changing root scheduler templates only affects newly prepared runs, so update
existing run copies explicitly when necessary. Runtime caps are evaluated per
specification. `--dry-run` writes nothing, and `--prepare-only` does not clear
completion markers or submit jobs.

For an existing workspace without an input record, the first grouped invocation
records its current inputs and reports that earlier changes cannot be verified.
Keep the record with the workspace. Do not edit scientific settings while runs
are active: checks happen when the controller starts. The record is also checked
when a tracked child workspace is run directly. Legacy workspaces without an
input record keep their previous input-editing behavior.

This feature isolates existing O₂-control settings; it does not add a prescribed
physical-flux controller or gas-mixture support. The material and geometry
limitations below still apply.

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

SGUSCHI wraps the SLUSCHI volume-search loop. `volsearch_cont` is a csh script that drives the full MD run. Each cycle it:

1. **Polls** for job completion (checks for `Total CPU` in OUTCAR every 60 s).
2. **Extracts pressure/stress** from the finished OUTCAR: Pulay stress, full stress tensor, kinetic pressure from temperature and volume.
3. **Predicts the next lattice** by running `DetermineSize.x` on the averaged pressure history (volume-search step; inherited from SLUSCHI).
4. **Adjusts INCAR tags**: `AdjustPOTIM` (timestep), `AdjustNBANDS` (band count), `AdjustBMIX` (mixing parameter).
5. **Archives the completed step**: creates a numbered folder (`1/`, `2/`, …), moves OUTCAR into it, copies INCAR/KPOINTS/POSCAR/OSZICAR, and touches an empty OUTCAR in place.
6. **Builds the next POSCAR**: header and species lines from CONTCAR, new lattice vectors from `DetermineSize.x`, then atom positions and velocities from CONTCAR (i.e. the last MD snapshot becomes the starting point).
7. **Calls `OxidationStep.py`**, which updates the gas environment (see below).
8. **Submits the next VASP job** via the configured `vaspcmd`.

```
volsearch_cont (csh)
    │  [startup] resolve sluschipath (env var sguschipath set by SGUSCHI.py, else ~/.sluschi.rc); set SIGMA/TEBEG/TEEND/NSW/SMASS in INCAR from job.in
    │
    └─ loop: poll OUTCAR → job done
            ├─ compute pressure, run DetermineSize.x → lattice_predict.out
            ├─ adjust INCAR (POTIM, NBANDS, BMIX)
            ├─ archive step N: mkdir N/, mv OUTCAR N/, cp inputs, touch OUTCAR
            ├─ build POSCAR: CONTCAR header + new lattice + CONTCAR positions/velocities
            ├─ python OxidationStep.py
            │       ├─ Reads:  POSCAR, {N}/OUTCAR, OxParams, CovalentRadii, RateAnalysis.csv
            │       ├─ Calls OxidationAnalysis: gas detection, smoothing, O2 placement
            │       └─ Writes: updated POSCAR, RateAnalysis.csv, XYZ trajectory
            └─ submit next VASP job; advance step counter
```

If `OxidationStep.py` exits with a non-zero status, `volsearch_cont` halts immediately and writes a `sguschi_failed` marker file.

### Job state markers

Each `Dir_VolSearch` carries marker files that record job state and drive recovery on
resubmission. `SGUSCHI.py` writes the `job.*` markers; the rest are written by the run
itself.

| Marker | Written by | Meaning |
|--------|-----------|---------|
| `job.started` | `SGUSCHI.py` | ISO timestamp when `volsearch_cont` was launched |
| `job.exit` | `SGUSCHI.py` | `volsearch_cont` exit code (`0` = done; `-1` = initial VASP submission failed) |
| `job.killed` | `SGUSCHI.py` | walltime/SIGTERM kill message |
| `volsearch_is_done` | `OxidationStep.py` | simulation reached its stopping condition |
| `maxruntime_reached` | `OxidationStep.py` | `MaxRuntime` (ps) cap was hit |
| `sguschi_failed` | `volsearch_cont` | `OxidationStep.py` raised an exception; run halted |

On every resubmission `SGUSCHI.py` automatically clears stale `job.exit`, `job.killed`,
and `sguschi_failed` markers, so a retried simulation is not misreported — you do **not**
need to delete the `job.*` markers by hand. To extend a `MaxRuntime`-capped run, see the
resuming note under [OxParams](#oxparams).

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
