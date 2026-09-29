# Troubleshooting

For normal operation, see [Monitoring and outputs](../README.md#monitoring-and-outputs).
For failed trajectories or rejected submissions, see
[Resuming and extending runs](../README.md#resuming-and-extending-runs).

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
