This directory shows a typical starting point for running SGUSCHI simulations.
For the full walkthrough, see the [README quick start](../README.md#quick-start).

## Quick start

**Step 1 — Customise for your system**

Edit the files in this folder before running anything:

| File | What to set |
|------|-------------|
| `OxParams` | Temperatures, NSims, GasRatio, InitO2Count, tolerances; optional `MaxRuntime` (ps) |
| `POSCAR` | Base crystal structure (cubic, no gas region, no oxygen atoms) |
| `POTCAR` | Pseudopotentials matching POSCAR element order (O last) |
| `INCAR` | VASP MD settings (IBRION=0, ISIF=2, NSW=80) |
| `KPOINTS` | K-point mesh |
| `job.in` | SLUSCHI config; **set `vaspcmd`** to your scheduler command (e.g. `sbatch`) |
| `CovalentRadii` | Element radii in Å used by the gas detection algorithm |
| `OxidationMaster` | Set `#SBATCH` tags, `module load` lines, and the path to `SGUSCHI.py` |

**Step 2 — Submit**

    sbatch OxidationMaster

SGUSCHI.py (called by OxidationMaster on the compute node) will:
1. Create simulation folder trees from OxParams (Temperatures × NSims), including
   [initial INCAR preparation](../README.md#incar-required-settings)
2. Submit the initial VASP job in each `Dir_VolSearch` using `vaspcmd` from `job.in`
3. Start `volsearch_cont` in all folders and run until completion or walltime

To prepare and inspect inputs before submission, run
`python /path/to/SGUSCHI/src/SGUSCHI.py . --prepare-only`. This applies the initial
INCAR settings and preserves your chosen `POTIM`, without submitting jobs.

**Step 3 — Extend or recover**

To extend simulations or recover after a walltime failure, simply resubmit:

    sbatch OxidationMaster

Existing folders and finished simulations are skipped automatically.
See [resuming and extending runs](../README.md#resuming-and-extending-runs) for
runtime caps and grouped-input restrictions.

**Step 4 — Monitor results**

Results accumulate in the workspace's `xyz_files/` folder (inside each child
workspace for grouped runs).
Monitor progress from the workspace directory with:

    python /path/to/SGUSCHI/src/utils/SimulationSummary.py . --stdout

Omit `--stdout` to write the text and TSV summaries. See
[monitoring and outputs](../README.md#monitoring-and-outputs) for columns and paths.

## Notes

- `JobSpecs = []` is disabled by default. To control several different jobs from
  one `OxidationMaster`, create a campaign directory with a top-level `OxParams`
  containing e.g. `JobSpecs = [ZrC_low, ZrN_high]` (quotes are optional). These
  workspaces can sit directly in the campaign root; no `jobs/` folder is needed.
  Copy the input files in this example into each child directory and customize
  them separately.
  Keep each child's `JobSpecs` empty. See
  [multiple specifications](../README.md#multiple-job-specifications-with-one-controller)
  for setup and directory rules.

- The POSCAR should be a simple cubic structure. The x-axis will be expanded by
  `GasRatio` and filled with `InitO2Count` O₂ molecules during setup.
- `vaspcmd` in `job.in` is read by both SLUSCHI (for VASP job submission) and
  SGUSCHI.py (for the initial VASP job). Make sure it matches your cluster scheduler.
- `navg` in `job.in` is automatically set to 10000000 in each `Dir_VolSearch` so
  that `volsearch_cont` runs indefinitely until the walltime is reached.
