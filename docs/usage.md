# User guide

For installation and a first run, start with the [README](../README.md).
For errors and manual recovery, see [Troubleshooting](troubleshooting.md).

- [Multiple job specifications](#multiple-job-specifications-with-one-controller)
- [Monitoring and outputs](#monitoring-and-outputs)
- [Resuming and extending runs](#resuming-and-extending-runs)
- [Configuration reference](#configuration-reference)
- [Known limitations](#known-limitations)

## Multiple job specifications with one controller

Use grouped specifications to run different starting compositions or O₂-control
settings together. Each specification is a complete workspace with its own
inputs, temperatures, replica count, and optional runtime cap.

1. Create a campaign directory with one `OxidationMaster` and one top-level
   `OxParams`. Create a child directory for each specification, using the same
   input files as in the [quick start](../README.md#quick-start):

   ```text
   campaign/
   ├── OxidationMaster
   ├── OxParams                     # JobSpecs list
   ├── ZrC_low/
   │   ├── OxParams                 # Scientific settings, temperatures, replicas
   │   ├── POSCAR, POTCAR, INCAR, KPOINTS
   │   ├── job.in, jobsub, CovalentRadii
   │   ├── 873_1/Dir_VolSearch/      # Generated during preparation
   │   └── xyz_files/               # This specification's trajectories
   ├── ZrC_high/
   │   └── ...
   └── ZrN_low/
       └── ...
   ```

2. Put the specification list on one line in the top-level `OxParams`:

   ```python
   JobSpecs = ["ZrC_low", "ZrC_high", "ZrN_low"]
   ```

   Quotes are optional for simple paths, and quoted/unquoted entries can be
   mixed. This is equivalent:

   ```text
   JobSpecs = [ZrC_low, ZrC_high, ZrN_low]
   ```

   Every unquoted entry is read as literal text: `1.20` and `001` retain their
   exact spelling. This behavior applies only to `JobSpecs`; expressions and
   malformed lists are rejected. Use quotes around parent paths containing
   spaces or commas.

   `JobSpecs` is optional and disabled by default: omitting it or setting it to
   `[]` uses the single-workspace layout. When enabled, the top-level file only
   needs this list; scientific settings there are ignored. Child workspaces do
   not inherit inputs or settings from the campaign root.
3. Customize each child's complete inputs. Leave its `JobSpecs` empty or omit
   it. `NSims` applies **per temperature within that specification**, and can
   differ between children. For example, two temperatures with `NSims = 3`
   create six trajectories in that child.
4. Configure the top-level `OxidationMaster` as in the quick start. From the
   campaign directory, use the same inspection, preparation, and submission
   commands:

   ```bash
   python /path/to/SGUSCHI/src/SGUSCHI.py . --dry-run
   python /path/to/SGUSCHI/src/SGUSCHI.py . --prepare-only
   sbatch OxidationMaster
   ```

Use paths relative to the campaign, preferably with forward slashes. Workspaces
can sit directly under the campaign root; a `jobs/` folder is optional. If you
use one, include it in each path, e.g. `jobs/ZrC_low`. Every listed path must be
an existing directory with the required inputs; a missing directory or a path
pointing to a file stops the controller before preparation or submission.

Directory basenames are specification IDs and must be unique, ignoring case.
Names must start with a letter or digit and contain only letters, digits, dots, underscores,
or hyphens. Directories must stay inside the campaign and cannot overlap.
Nested `JobSpecs` lists are not supported. The campaign's `SimulationSummary`,
`.simulation_summary`, and `logs` locations are reserved for controller output.

All specifications are validated before submission, and one controller launches
all pending trajectories concurrently. Runs such as `ZrC_low/873_1` and
`ZrN_low/873_1` have separate state and output; scheduler job names include the
specification ID. A failed initial submission does not prevent other eligible
runs from launching. See [Resuming and extending runs](#resuming-and-extending-runs)
before changing inputs for an existing specification.

## Monitoring and outputs

**Monitoring starts automatically when the controller launches trajectories.**
The summary refreshes every **60 seconds** while the controller is alive;
no separate monitoring command is needed. `--prepare-only` does not start it.

### Simulation summary

| Location, relative to the workspace or campaign | Contents |
|---|---|
| `SimulationSummary` | Readable text table |
| `logs/SimulationSummary.tsv` | Tab-separated version of the same table |

Both contain **one row per trajectory**, with separate job-folder and trajectory
columns: for example, `JobFolder = ZrC_low`, `Trajectory = 873_1`. `JobFolder`
is the specification directory's basename, or `-` for an ungrouped workspace.
Other columns show status, simulated time in **ps**, O₂ added, molecules removed,
step/history counts, activity age, and estimated wall/queue times.

Common statuses include `NOT_STARTED`, `RUNNING` (including queued jobs), `DONE`,
`FAILED`, `KILLED`, `AWAITING` manual submission, and `STUCK`. Read `Detail` and
the logs listed below when action is needed. See [resuming and recovery](#resuming-and-extending-runs)
or [summary troubleshooting](troubleshooting.md#simulation-summary).

### Output locations

These paths are relative to a **specification workspace**: the main workspace
in single-specification mode, or a child such as `ZrC_low` in grouped mode.

| Location | Contents |
|---|---|
| `xyz_files/873_1.xyz` | Extended XYZ trajectory for temperature 873 K, replica 1 |
| `xyz_files/RateAnalysis_873_1.csv` | Exported gas-management history for analysis |
| `873_1/Dir_VolSearch/RateAnalysis.csv` | Working history used to continue the trajectory; same rows as the export |
| `873_1/Dir_VolSearch/1/`, `2/`, … | Archived VASP segments |
| `873_1/log.out` | Controller-script output and recovery messages |
| `873_1/Dir_VolSearch/sim_log.tsv` | Lifecycle and submission events |

### XYZ trajectories

Completed segments are appended to the trajectory, with one frame per parsed
MD step. Each frame contains an atom count, a metadata line, then one
`element x y z` line per atom with Cartesian coordinates in **Å**.
Metadata includes the cell lattice, periodic boundaries, cumulative `Step` and
`Time` (**fs**), and available energies (**eV**), temperature (**K**), and
pressure (**kbar**).

Frames describe the VASP trajectory before gas removal or insertion at the end
of that segment. These changes appear in subsequent segments, so the number of
atoms can change between segments.

### RateAnalysis CSVs

The history starts with an initialization row at **0 fs**, followed by one row
per completed gas-analysis cycle (80 MD steps). Both CSV copies are updated
after each cycle.

| Column | Meaning |
|---|---|
| `Time (fs)` | Cumulative simulated time at the end of the segment |
| `O2 Count` | O₂ molecules detected in the final MD frame, plus any molecule inserted for the next segment |
| `Smoothed O2 Count` | O₂ count smoothed over MD frames using `OSmoothing`, before the new insertion |
| `O2 Added` | Cumulative O₂ molecules supplied, including `InitO2Count` |
| `Gas Removed` | Molecules removed in this cycle, stored as a list of element tuples, e.g. `[('C', 'O', 'O')]` for one CO₂; `[]` means none |
| `Free Gas Fraction` | Estimated gas-space width relative to its starting width, using Zr positions along x; dimensionless and capped at 1 |

The initialization row uses `InitO2Count` for the three O₂ columns, `[]` for
removed gas, and 1 for the free gas fraction. `Gas Removed` is per cycle;
`O2 Added` is cumulative.

## Resuming and extending runs

For restarts after walltime, failures or rejected submissions, see
[recovery instructions](troubleshooting.md#resume-after-walltime-or-a-failure).

### Extend simulated time or add trajectories

To extend a run stopped by `MaxRuntime`, raise that value in the specification's
`OxParams` and resubmit. The controller compares the recorded simulated time
against the new cap and reopens time-capped runs that are below it. An unchanged
or lower cap leaves completed runs stopped, and naturally completed runs are
not reopened. Unreadable runtime history is left stopped for inspection.

Increase `NSims` or add entries to `Temperatures` to create more trajectories on
the next preparation or submission. Existing trajectories keep their state.
In grouped mode, these settings are independent for each child. Add another
path to the campaign's `JobSpecs` to include a new specification. Removing a
path leaves its files untouched and removes its runs from the expected list on
the next preparation or startup.

### Change inputs for a grouped specification

On first preparation, the controller saves `.sguschi_inputs.json` in each child.
On subsequent invocations it compares the scientific inputs against that record.

| Change | How to apply it |
|---|---|
| Temperatures, `NSims`, or `MaxRuntime` | Edit the child's `OxParams`, then prepare or resubmit |
| Composition, gas settings, `POSCAR`, `POTCAR`, `INCAR`, `KPOINTS`, `CovalentRadii`, or scientific `job.in` settings | Create a fresh specification directory containing only the new inputs; add it to `JobSpecs` |
| Scheduler submission settings (`jobsub` or `vaspcmd`) | Changes are allowed; existing run copies must be updated explicitly if they need the changes |

Do not copy old trajectories or the input record into a fresh specification.
Setup does not overwrite existing run inputs: edited root templates affect newly
prepared trajectories. Keep the input record with its workspace, and avoid
editing scientific settings while a run is active; checks happen at controller
startup. The record is also checked when a tracked child is launched directly.

For a pre-existing workspace without a record, the first grouped invocation
records its current inputs and reports that earlier changes cannot be verified.
Legacy workspaces without a record retain their previous input-editing behavior.

## Configuration reference

### Input files

Each specification needs the scientific and VASP workflow inputs below. Use one
customized `OxidationMaster` per workspace or campaign. At a grouped campaign's
top level, the only input files needed are `OxParams` and `OxidationMaster`.

| File | Purpose |
|---|---|
| `OxParams` | Temperatures, replicas, gas-control settings, optional runtime cap |
| `POSCAR` | Starting bulk supercell; preprocessing adds vacuum and O₂ |
| `POTCAR` | Pseudopotentials matching POSCAR species order, with O last for inserted oxygen |
| `INCAR` | VASP settings; required MD tags are listed below |
| `KPOINTS` | VASP k-point mesh |
| `job.in` | SLUSCHI control parameters and scheduler submission command |
| `jobsub` | Cluster-specific VASP submission script |
| `CovalentRadii` | Element radii for bond detection |
| `OxidationMaster` | Scheduler script that runs the SGUSCHI controller; one per workspace or campaign |

Supply a compact `POSCAR` without an existing gas region: preprocessing adds
that region itself. It opens the largest interlayer gap; when the gap across the
cell boundary is tied for largest (within `1e-12` in fractional coordinates), it
prefers that boundary. This preserves the chosen surface planes of an ideal SQS
whose vacuum has been removed. A distinctly larger interior gap still takes
precedence, so a layer split across the periodic boundary remains intact.

### OxParams

Scientific keys are required in each specification's `OxParams` except for
`MaxRuntime`. `JobSpecs` is optional. The parser supports `#` and `!` comments.

| Key | Description |
|---|---|
| `JobSpecs` | Optional list of child workspaces; absent or `[]` disables grouping. A nonempty list makes the root file a campaign configuration; see [multiple specifications](#multiple-job-specifications-with-one-controller). |
| `Temperatures` | List of simulation temperatures in K |
| `NSims` | Number of trajectories **per temperature**, within this specification |
| `GasRatio` | Fraction by which the x-axis is expanded to create the gas region |
| `InitO2Count` | Initial number of O₂ molecules |
| `AtomicRadiusTol` | Multiplier on the sum of covalent radii for bond detection |
| `O2Tol` | O₂ count threshold, scaled by the current gas fraction during the run |
| `OSmoothing` | Exponential smoothing factor α; the example uses `0.001` (heavily history-weighted). This key must be supplied. |
| `MaxRuntime` | Optional simulated-time cap in ps, checked against cumulative time after each completed MD segment. The final segment can overshoot the cap. If omitted, no Python time cap is applied. Other stopping conditions and scheduler walltime still apply. |

These settings control the existing O₂ count-based replenishment algorithm.
They do not prescribe a physical impingement flux or enable gas mixtures.

### CovalentRadii

One entry per line, with radii in Å, for every element used by bond detection:

```text
Zr = 1.45
C = 0.76
O = 0.66
```

`#` and `!` comments are supported.

### INCAR (required settings)

| Tag | Value | Reason |
|---|---|---|
| `IBRION` | `0` | Molecular dynamics |
| `ISIF` | `2` | Fixed lattice during each VASP segment; ions move |
| `NSW` | `80` | Steps per cycle; set before the first submission |

Preparation applies the same initial settings previously applied at
`volsearch_cont` startup:

| Tag | Initial value |
|---|---|
| `TEBEG`, `TEEND` | Trajectory temperature from `Temperatures` |
| `SIGMA` | `0.000086 × temperature` eV, retaining the existing SLUSCHI formula |
| `NSW` | `80` |
| `SMASS` | `0` |
| `NBANDS` | Explicit value removed so VASP chooses its initial default |

These settings are ready in the generated `INCAR` files before either automatic
or manual submission; workspace input templates are unchanged. `POTIM` is
preserved from your template: use `POTIM = 1.0` for a 1 fs initial timestep, or
`0.8` for 0.8 fs. Set `adj_potim = 0` in `job.in` to keep the timestep fixed;
otherwise SLUSCHI may adjust it after completed segments.

Preparation stops before any initial submission if initialization fails.
Resuming a submitted or previously started trajectory does not reapply initial
settings or remove its adapted `NBANDS`. Rebuilding from XYZ with
`ResumeFromTrajectory.py` initializes the newly reconstructed inputs explicitly.

### job.in

Start with [example/job.in](../example/job.in). Set `vaspcmd` to your scheduler's
submission command, such as `sbatch` or `qsub`; both SGUSCHI and SLUSCHI use it.
Setup sets `temp` for each trajectory and sets `navg = 10000000` in the active
`Dir_VolSearch/job.in`. Other SLUSCHI stopping and adjustment settings remain
in effect. Changing `INCAR` or `job.in` alone does not change the enforced
80-step gas-analysis interval.

## Known limitations

1. **Material system:** void-fraction tracking uses Zr atoms as the solid
   reference. Testing has focused on cubic Zr refractory materials in pure O₂.
2. **Structure geometry:** cubic bulk structures only. The origin-shifting
   heuristic assumes roughly equal inter-atom spacing; non-cubic and slab
   geometries are unsupported.
3. **Cell orientation:** the gas region must lie along the x-axis (first lattice
   vector). Void tracking, O₂ placement, and surface-area calculations assume it.
4. **Gas addition:** only O₂ can be added. Its bond length is hardcoded to 1.2 Å,
   with velocities sampled for two O atoms. Gas mixtures are unsupported.
5. **Gas removal:** only molecules of 2–3 atoms are detected by the current
   settings. All detected non-O₂ molecules are removed each cycle.
6. **Oxygen in the starting structure:** use a base POSCAR without oxygen; the
   preprocessing workflow has not been tested with oxygen already present.
7. **Cell dynamics:** VASP segments use `ISIF=2`; variable-cell VASP MD is
   unsupported. Inherited SLUSCHI logic can adjust lattice vectors between segments.
8. **MD cycle length:** gas analysis runs every 80 VASP MD steps. Changing this
   requires modifying `volsearch_cont`, not just the input files.
9. **Elemental masses:** velocity initialization covers O, C, Zr, and N only.
   Other elements require extending `src/workflow/OxidationAnalysis.py`.
