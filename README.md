# SGUSCHI

**SGUSCHI** (Solid-Gas in Ultra Small Coexistence with Hovering Interfaces) extends
[SLUSCHI](https://github.com/qjhong/SLUSCHI) to simulate oxidation with small-cell
VASP molecular dynamics. Every 80 MD steps it removes detected non-O₂ gases,
tracks the available gas space and conditionally replenishes O₂. One controller
manages multiple temperatures, replicas and starting compositions.

<p align="center">
  <img src="docs/Visual_Abstract_git.png" width="85%" alt="Visual abstract">
</p>

## Supported systems

Tested on **cubic Zr-based materials**, including ZrC and ZrN, in **pure O₂**.

- Start with an **oxygen-free bulk supercell without a gas region**. Preprocessing
  creates the gas region along **x**; void tracking uses Zr atoms.
- O₂ replenishment is **count-based**, using a smoothed molecule count and the
  available gas fraction. It does not prescribe a physical impingement flux.
- Gas removal currently detects molecules containing **2–3 atoms**. Non-cubic
  structures, slab inputs and gas mixtures are unsupported.

See the [full limitations](docs/usage.md#known-limitations) before adapting the
workflow to another system.

## Requirements and installation

Use a cluster with **Python ≥ 3.8**, licensed **VASP**, a **Fortran compiler**
(`ifort`, `ifx` or `gfortran`), **C shell (`csh`)** and a job scheduler. Slurm and
PBS Torque are supported; the supplied submission templates use Slurm.

From the cloned repository root:

```bash
python -m pip install numpy pandas scipy
cd src/dependencies/SLUSCHI_mod
make CC=ifort                 # or CC=ifx / CC=gfortran
chmod +x *
cd ../../..
```

Optional postprocessing dependencies: `python -m pip install plotly tqdm`.
For development tools and tests, see the [developer guide](docs/development.md#testing).

## Quick start

1. Copy the [example directory](example/) into a separate simulation workspace.
   Customize the inputs for your system and cluster:

   | Files | What to configure |
   |---|---|
   | `POSCAR`, `POTCAR` | Starting bulk structure and matching potentials, including O last; the example POTCAR is empty |
   | `INCAR`, `KPOINTS` | VASP MD settings (`IBRION=0`, `ISIF=2`, `NSW=80`) and k-point mesh |
   | `OxParams`, `CovalentRadii` | Temperatures, replicas, gas settings and bond-detection radii |
   | `jobsub`, `job.in` | VASP submission script and `vaspcmd` (`sbatch` or `qsub`) |
   | `OxidationMaster` | Controller resources, modules and absolute path to `SGUSCHI.py` |

2. Edit the example `OxParams`, keeping the remaining gas-control settings.
   For two temperatures with four trajectories each:

   ```python
   JobSpecs = []
   Temperatures = [873, 973]
   NSims = 4
   # MaxRuntime = 140   # optional simulated-time cap in ps
   ```

3. From your simulation workspace, inspect and optionally prepare the run:

   ```bash
   python /path/to/SGUSCHI/src/SGUSCHI.py . --dry-run
   python /path/to/SGUSCHI/src/SGUSCHI.py . --prepare-only
   ```

   Replace `/path/to/SGUSCHI` with your clone's location. `--dry-run` only reports
   the plan; `--prepare-only` creates run folders and initializes their INCARs
   without submitting jobs. Preparation sets temperature-dependent tags and
   removes an explicit initial `NBANDS`; it preserves your chosen `POTIM`.
   See the [initial INCAR settings](docs/usage.md#incar-required-settings).

4. Submit the controller from the same workspace:

   ```bash
   sbatch OxidationMaster
   ```

   The controller prepares missing folders, submits initial VASP jobs and runs
   all pending trajectories. Preparation in step 3 is optional. For PBS, adapt
   the supplied scheduler scripts and submission command.

For individual parameters, see the [configuration reference](docs/usage.md#configuration-reference).

## Monitoring and outputs

The controller automatically refreshes `SimulationSummary` every 60 seconds
while it is running. A TSV copy is written to `logs/SimulationSummary.tsv`.

| Output | Contents |
|---|---|
| `xyz_files/873_1.xyz` | Extended XYZ trajectory for 873 K, replica 1 |
| `xyz_files/RateAnalysis_873_1.csv` | Gas supply/removal history and free gas fraction |
| `873_1/Dir_VolSearch/1/`, `2/`, … | Archived VASP segments |
| `873_1/log.out` | Trajectory workflow and recovery messages |

For grouped runs, trajectory outputs live inside each specification's workspace.
See [output formats and units](docs/usage.md#monitoring-and-outputs) or
[monitoring troubleshooting](docs/troubleshooting.md#simulation-summary).

## Resuming and extending runs

Once the previous controller has stopped, resubmit from the same workspace:

```bash
sbatch OxidationMaster
```

Existing trajectories are preserved and pending runs resume. To extend a run
stopped by `MaxRuntime`, raise that value before resubmitting; otherwise completed
runs remain stopped. Increase `NSims` or add temperatures to create more runs.

See [extending runs and changing inputs](docs/usage.md#resuming-and-extending-runs)
and [failure recovery](docs/troubleshooting.md#resume-after-walltime-or-a-failure).

## Multiple starting compositions or gas settings

One controller can manage separate workspaces, each with its own complete inputs.
List those directories in the campaign's top-level `OxParams`:

```python
JobSpecs = ["ZrC_low", "ZrC_high", "ZrN_low"]
```

Keep each child's `JobSpecs` empty. See the
[grouped-run guide](docs/usage.md#multiple-job-specifications-with-one-controller)
for directory layout, validation and input-change rules.

## Documentation

- [User guide](docs/usage.md): grouped runs, configuration, outputs and limitations.
- [Troubleshooting](docs/troubleshooting.md): restarts, rejected submissions and monitoring.
- [Developer guide](docs/development.md): architecture, state markers and tests.
- [Example inputs](example/): templates to customize before running.

## Citation and license

A publication DOI and project license have not yet been specified in this
repository.
