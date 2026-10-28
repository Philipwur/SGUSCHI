# Automatic bulk relaxation

SGUSCHI can run the bundled SLUSCHI `optunitcell` before adding vacuum and
oxygen. One bulk relaxation per specification supplies every temperature and
replica. It is disabled by default and cannot be retrofitted onto trajectories.

## Configure

Relaxation is optional. In a single workspace, set it in OxParams; in a grouped
campaign, set it separately in each specification's OxParams.

```text
Relaxation = off    # Skip optunitcell; prepare MD from the supplied POSCAR.
```

Omitting the parameter has the same effect. Change its value to `bulk` to run
optunitcell before MD. No separate templates or additional enable/disable flag
are required. These choices apply when preparing a workspace; changing the mode
does not cancel an already submitted job or change existing MD trajectories.

| Setting | Default | Meaning |
|---|---|---|
| Relaxation | off | Optional: bulk runs optunitcell; off skips it |
| RelaxForceTol | 0.02 | Maximum final force, eV/Å |
| RelaxPressureTol | 1.0 | Allowed residual pressure relative to press, kbar |
| RelaxMaxSteps | 200 | Ionic steps per pass |
| RelaxENCUT | Inherit | Optional cutoff override, eV |
| RelaxKMesh | Inherit | Optional Gamma-centred mesh, e.g. [3,4,4] |
| RelaxWalltime | Inherit | Optional Slurm [days-]HH:MM:SS or PBS hours:MM:SS |

The existing POSCAR, INCAR, KPOINTS, POTCAR, job.in and jobsub are source inputs.
Cutoff and k-point sampling are inherited **without a convergence claim**.
For example, the miscellaneous bulk preparation used:

```text
Relaxation = bulk
RelaxENCUT = 700
RelaxKMesh = [3,4,4]
RelaxWalltime = 24:00:00
```

These are an example for those cells, not universal convergence settings.
Volume relaxation is sensitive to basis-set/Pulay stress; check sampling and
cutoff for your material. See [VASP volume relaxation](https://vasp.at/wiki/Volume_relaxation).

The static profile uses IBRION=2, ISIF=8, POTIM=0.3, PREC=Accurate, Gaussian
smearing at 0.05 eV, EDIFF no looser than 1e-6, NELM at least 200 and
EDIFFG=-RelaxForceTol. Spin, mixing, LREAL and other electronic settings are
otherwise inherited. MD thermostat settings and velocities are removed.
Each pass starts from atomic charge densities without reusing WAVECAR/CHGCAR.
The pressure target is press from job.in, defaulting to zero if absent.

ISIF=8 relaxes positions and volume at fixed cell shape and requires
[VASP 6.4.1 or later](https://vasp.at/wiki/ISIF). The output version is checked.
Supported inputs are positive orthogonal oxygen-free bulk cells without frozen
atoms. Subsequent MD preparation does not carry selective-dynamics constraints.

POTCAR blocks are selected by element in POSCAR order; the oxygen potential
appended for oxidation is excluded. Original files are never overwritten.
The copied jobsub retains modules, executable and resources, with a relaxation
job name and optional walltime override.

## Run and inspect

From a specification or grouped campaign:

```bash
python /path/to/SGUSCHI/src/SGUSCHI.py . --dry-run
python /path/to/SGUSCHI/src/SGUSCHI.py . --prepare-only
sbatch OxidationMaster
```

Normal execution continues automatically to oxidation after acceptance.
Preparation is optional: the normal SGUSCHI command performs it as needed.
The standalone preprocessing/WorkSpaceSetup.py entry point delegates to the
same preparation-only flow, including grouped campaigns:

| State | Preparation stops at |
|---|---|
| bulk, no accepted result yet | BulkRelaxation/Dir_OptUnitCell, once per specification |
| off or omitted | temperature_replica/Dir_VolSearch, ready for initial MD |
| bulk, result already accepted | temperature_replica/Dir_VolSearch, using the accepted bulk |

Preparation never submits jobs. Normal execution submits both relaxation passes
when enabled, then creates the MD folders and launches oxidation automatically.
Use a controller allocation with enough walltime for queue waits, both passes
and MD; RelaxWalltime controls the individual VASP relaxation jobs separately.

For a review checkpoint, have the controller call:

```bash
python /path/to/SGUSCHI/src/SGUSCHI.py . --relax-only
```

Relaxation-only submits and waits for jobs; use the controller allocation
rather than a long-lived login-shell process. Combine it with --prepare-only
to write only relaxation inputs.

- --dry-run writes nothing and submits nothing.
- --prepare-only writes relaxation inputs while the result is pending. With
  accepted results, it creates trajectory inputs without submitting jobs.
- Specifications relax concurrently. When workers finish, accepted specs
  continue together; failed specs are withheld.
- Controller restart reattaches to recorded jobs and reuses completed passes.
  Adding temperatures or replicas does not repeat relaxation.

```text
specification/
├── POSCAR, INCAR, KPOINTS, POTCAR, job.in, jobsub  # original inputs
├── BulkRelaxation/
│   ├── POSCAR, INCAR, KPOINTS_OPT, POTCAR, job.in, jobsub
│   ├── Inputs.json, Status.json, optunitcell.log
│   ├── Accepted.POSCAR
│   └── Dir_OptUnitCell/
│       ├── pass1/                                # inputs, outputs, Pass.json
│       ├── pass2/
│       └── optunitcell_is_done
└── 973_1/
    ├── BulkRelaxation.sha256
    └── Dir_VolSearch/
```

SimulationSummary includes a separate bulk-relaxation section. Status.json
records the pass, scheduler ID, validation metrics and failure reason.
Each pass has Submission.log; scheduler output/error locations follow jobsub.

## Acceptance and recovery

Pass 1 must finish normally, with electronically converged steps and a valid
CONTCAR. It may exhaust its ionic-step budget. Pass 2 must also reach ionic
convergence and meet the force and residual-pressure tolerances. Both must
preserve cell shape and atom inventory. An electronic iteration cap is
conservatively treated as failure. Inputs and validation outputs are checksummed.

The cleavage is selected before relaxation. Coordinates are unwrapped by
original atom identity after removing rigid translation. A newly larger
interior gap cannot silently expose a different SQS decoration. Ambiguous or
large reconstruction is flagged for review.

OS locks and persistent submission records prevent duplicate jobs. Failed,
timed-out or unconverged jobs stop the specification. An uncertain submission
or unavailable scheduler status never triggers automatic resubmission.
Restarting the controller rechecks a known job when status was unavailable.
Controller termination stops monitoring processes, not external VASP jobs.

For a failed/uncertain attempt, inspect logs and scheduler state before starting
a new attempt. Use a fresh specification for changed scientific inputs. Never
delete a submission record to force a retry while its job may still be live.
There is no automatic third pass or parameter-tuning campaign.

Previously relaxed external structures can be supplied as the ordinary POSCAR
with Relaxation=off. Bulk relaxation does not replace oxygen-free thermal slab
equilibration or determine a finite-temperature equilibrium lattice parameter.

## Implementation and tests

The existing optunitcell owns the two-pass sequence via its --sguschi entry
point; its standalone invocation retains legacy behaviour.
utils/BulkRelaxation.py derives inputs, handles one pass, validates results and
publishes the accepted geometry before workspace setup.

```bash
python -m pytest src/test/TestBulkRelaxation.py -q -o 'python_functions=Test* test_*'
```

Tests use synthetic POTCAR/VASP outputs and fake scheduler commands. The real
C-shell integration test requires csh/tcsh; it invokes the bundled routine with
a fake sbatch and never runs VASP or submits a cluster calculation.
