# Troubleshooting

For initial setup, see the [README quick start](../README.md#quick-start).
For configuration, outputs and extending runs, see the [user guide](usage.md).

- [Resume after walltime or a failure](#resume-after-walltime-or-a-failure)
- [Manual first VASP submission](#manual-first-vasp-submission)
- [Simulation summary](#simulation-summary)

## Resume after walltime or a failure

After the previous controller has stopped, resubmit from the same workspace or
campaign root:

```bash
sbatch OxidationMaster
```

Existing folders are preserved, completed trajectories are skipped, and pending
trajectories are resumed. A definitely rejected initial VASP submission can be
retried; recorded submissions are guarded against duplicate queuing. Inspect the
trajectory's `log.out` and scheduler output to resolve the cause of a failure.

If a running controller reports `AWAITING`, a later VASP submission was rejected
and that trajectory is waiting for manual submission. Follow the recovery
command printed in its `log.out`: from the affected `Dir_VolSearch`, submit
`jobsub` using the configured scheduler command. The master can remain running.

## Manual first VASP submission

For a manual first VASP submission after `--prepare-only`, work inside the
chosen trajectory's `Dir_VolSearch`. With the controller stopped and no VASP
job already queued or running there, record the first-step submission guard
before submitting:

```bash
cd 873_1/Dir_VolSearch  # In grouped mode, include the specification's path
printf '1\n' > .vasp_submitted_step && sbatch jobsub
```

Use `qsub` instead if configured for PBS. This guard prevents a later controller
start from submitting the same job again while the manual job is still queued
and has no `OUTCAR`. If submission is definitely rejected, remove the guard
before retrying; after an ambiguous scheduler timeout, check the queue first.
Normally, submitting `OxidationMaster` handles this bookkeeping automatically.

## Simulation summary

### Inspect a workspace when the controller is stopped

Normal controller startup automatically starts the summary watcher. Preparation
alone does not start monitoring, and automatic updates end when the controller
exits. To create or refresh a summary once, run this from the workspace or campaign
root:

```bash
python /path/to/SGUSCHI/src/utils/SimulationSummary.py .
```

Replace `/path/to/SGUSCHI` with your clone's location. This writes
`SimulationSummary` and `logs/SimulationSummary.tsv` without submitting jobs or
starting a watcher. Running it from an individual specification directory gives
a local summary with `JobFolder` shown as `-`.

### Missing trajectories or stale selection

Grouped summaries use `.simulation_summary/expected.tsv`, generated during
preparation or normal controller startup. Only selected trajectories are included.
After changing `JobSpecs`, `Temperatures`, or `NSims`, run preparation or restart
the stopped controller to update this list. `--dry-run` does not write it, and a
manual summary refresh does not rebuild it.

### No updates while trajectories are running

Confirm that the master controller is still running; queued or running VASP jobs
can outlive it. Check `.simulation_summary/watcher.log` for watcher errors and the
master job's scheduler output for controller errors. Individual trajectory logs
are in `<trajectory>/log.out` within each specification workspace.
