This directory shows a typical starting point for running SGUSCHI simulations.

## Quick start

**Step 1 — Customise for your system**

Edit the files in this folder before running anything:

| File | What to set |
|------|-------------|
| `OxParams` | Temperatures, NSims, GasRatio, InitO2Count, tolerances |
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
1. Create simulation folder trees from OxParams (Temperatures × NSims)
2. Submit the initial VASP job in each `Dir_VolSearch` using `vaspcmd` from `job.in`
3. Start `volsearch_cont` in all folders and run until completion or walltime

**Step 3 — Extend or recover**

To extend simulations or recover after a walltime failure, simply resubmit:

    sbatch OxidationMaster

Existing folders and finished simulations are skipped automatically.

**Step 4 — Monitor results**

Results accumulate in each simulation's `xyz_files/` folder.
Monitor progress from the workspace directory with:

    python /path/to/SGUSCHI/src/utils/SimulationSummary.py .

## Notes

- The POSCAR should be a simple cubic structure. The x-axis will be expanded by
  `GasRatio` and filled with `InitO2Count` O₂ molecules during setup.
- `vaspcmd` in `job.in` is read by both SLUSCHI (for VASP job submission) and
  SGUSCHI.py (for the initial VASP job). Make sure it matches your cluster scheduler.
- `navg` in `job.in` is automatically set to 10000000 in each `Dir_VolSearch` so
  that `volsearch_cont` runs indefinitely until the walltime is reached.
