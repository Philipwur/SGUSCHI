# Developer guide

For installation and normal operation, see the [README](../README.md) and
[user guide](usage.md). For failures, see [Troubleshooting](troubleshooting.md).

- [Architecture](#architecture)
- [Job state markers](#job-state-markers)
- [Testing](#testing)

## Architecture

`OxidationMaster` launches `src/SGUSCHI.py`, which resolves specifications,
prepares new trajectories, reconciles runtime caps, submits initial VASP jobs,
and starts one `volsearch_cont` process per pending trajectory. The controller
also starts the campaign or workspace summary watcher.

`src/utils/InitialVasp.py` owns initial INCAR preparation. Workspace setup and
trajectory reconstruction share it, and the controller prepares all eligible
first jobs before submitting any. `volsearch_cont` calls the same guarded helper
for standalone startup; it leaves submitted/running and historical inputs alone.

`volsearch_cont` is a C-shell script that invokes the compiled SLUSCHI helpers.
For each MD segment it:

1. Polls `OUTCAR` for completion (`Total CPU`, checked every 60 seconds).
2. Extracts pressure and stress, including Pulay and kinetic contributions.
3. Uses pressure history and `DetermineSize.x` to predict lattice changes.
4. Copies the completed segment's inputs into its numbered folder, then adjusts
   the working INCAR (`POTIM`, `NBANDS`, and `BMIX`) for the next segment.
5. Archives OUTCAR and prepares the next POSCAR from CONTCAR and the predicted
   lattice. The archived INCAR retains the settings used for the completed segment.
6. Calls `OxidationStep.py` to detect/remove gases, update the O₂ count history,
   conditionally insert O₂, and write POSCAR, rate analysis, and XYZ output.
7. Submits the next VASP job unless a stopping condition or failure occurred.

```text
OxidationMaster
└── SGUSCHI.py
    ├── Resolve specifications and prepare/resume trajectories
    ├── SimulationSummary.py watcher
    └── volsearch_cont (one process per trajectory)
        ├── Poll VASP, calculate pressure/lattice, archive segment
        ├── OxidationStep.py
        │   ├── Reads: POSCAR, archived OUTCAR, local OxParams and radii
        │   └── Writes: POSCAR, RateAnalysis.csv, XYZ trajectory
        └── Submit next VASP segment
```

SGUSCHI sets the `sguschipath` environment variable to the bundled SLUSCHI
scripts; their fallback is `~/.sluschi.rc`. A nonzero `OxidationStep.py` exit
halts the script: if `volsearch_is_done` exists, this is a clean stop; otherwise
it records `sguschi_failed` and reports failure.

The master checks its local `volsearch_cont` processes every 30 seconds and
records each exit independently, even while other trajectories are still running.
This check uses only local process status; it adds no scheduler queries or file
scans. The summary watcher retains its separate 60-second refresh interval.

## Job state markers

These files live in each trajectory's `Dir_VolSearch`:

| File | Written by | Purpose |
|---|---|---|
| `sim_log.tsv` | Controller and SLUSCHI script | Append-only `started`, `submitted`, `exit`, `killed`, and `await` events |
| `job.exit` | Controller | Script exit code; `-1` for initial submission or launch failure |
| `volsearch_is_done` | Simulation workflow | Trajectory reached a stopping condition |
| `maxruntime_reached` | `OxidationStep.py` | Identifies a stop caused by the simulated-time cap |
| `sguschi_failed` | SLUSCHI script | Workflow/helper failure; trajectory halted |
| `.vasp_submitted_step` | Controller and SLUSCHI script | Submission guard for a numbered VASP segment |
| `awaiting_manual_submission` | SLUSCHI script | Submission was rejected; waiting for operator recovery |

Before relaunch, the controller clears stale `job.exit`, `sguschi_failed`, and
`awaiting_manual_submission` markers and appends a new `started` event. The event
history is retained. Current start/kill events are recorded in `sim_log.tsv`,
rather than separate `job.started` and `job.killed` files. Runtime-cap markers
are reconciled as described under [resuming](usage.md#resuming-and-extending-runs).

## Testing

From the repository root, install the development tools and enter the test
directory:

```bash
python -m pip install pytest black ruff
cd src/test

# Full suite: include the repository's PascalCase and lowercase test names
python -m pytest . -q -o 'python_files=Test*.py' -o 'python_functions=Test* test_*'

# One module
python -m pytest TestJobSpecs.py -q -o 'python_functions=Test* test_*'
```

`conftest.py` adds `src/` to the import path. The explicit discovery options
ensure that pytest collects both naming styles; a plain `pytest` invocation
may miss tests. `RunTests.py` is an alternative convenience runner for the
PascalCase tests.

`TestPressureAvg.py` compiles and runs the `PressureAvg.f90` helper. Its tests
are skipped if no `ifort`, `ifx`, or `gfortran` is on `PATH`. The runtime
C-shell check is skipped if `csh` is unavailable. Controller tests mock scheduler
calls; passing them does not constitute a live Slurm/PBS/VASP run.
