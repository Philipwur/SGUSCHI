"""Resolve optional grouped job specifications before the controller starts work."""

import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


REQUIRED_KEYS = [
    "Temperatures", "NSims", "GasRatio", "InitO2Count",
    "AtomicRadiusTol", "O2Tol", "OSmoothing",
]
INPUT_FILES = [
    "POSCAR", "POTCAR", "INCAR", "KPOINTS", "job.in", "jobsub", "CovalentRadii",
]
INPUT_RECORD = ".sguschi_inputs.json"


@dataclass
class JobSpec:
    """A workspace with local inputs and an optional campaign-wide identifier."""

    Id: str
    Root: Path
    Params: Dict[str, str]

    def RunId(self, Label: str) -> str:
        return "{}/{}".format(self.Id, Label) if self.Id else Label


def ParseJobSpecs(Params: dict) -> List[str]:
    """Absent or empty JobSpecs disables grouping; a nonempty list enables it."""
    Raw = Params.get("JobSpecs", "[]")
    try:
        Entries = ast.literal_eval(Raw)
    except (ValueError, SyntaxError, TypeError) as Error:
        raise ValueError("JobSpecs must be a list of directory strings, or [] to disable it") from Error
    if not isinstance(Entries, list) or any(
        not isinstance(Entry, str) or not Entry.strip() for Entry in Entries
    ):
        raise ValueError("JobSpecs must be a list of nonempty directory strings")
    return Entries


def ValidateGroupedParams(Params: dict) -> None:
    """Reject malformed plans before any workspace is prepared or submitted."""
    Temperatures = ast.literal_eval(Params["Temperatures"])
    if not isinstance(Temperatures, list) or not Temperatures:
        raise ValueError("Temperatures must be a nonempty list")
    for Temp in Temperatures:
        if isinstance(Temp, bool) or not isinstance(Temp, (int, float)) or not math.isfinite(Temp) or Temp <= 0:
            raise ValueError("Temperatures must contain positive finite numbers")
    if len(set(Temperatures)) != len(Temperatures):
        raise ValueError("Temperatures must not contain duplicates")
    if int(Params["NSims"]) < 1:
        raise ValueError("NSims must be a positive integer")
    if int(Params["InitO2Count"]) < 0:
        raise ValueError("InitO2Count must be a nonnegative integer")
    for Key in ("GasRatio", "AtomicRadiusTol", "O2Tol", "OSmoothing", "MaxRuntime"):
        if Key not in Params:
            continue
        Value = float(Params[Key])
        if not math.isfinite(Value) or Value < 0 or (Value == 0 and Key != "O2Tol"):
            raise ValueError("{} must be {}finite number".format(
                Key, "a nonnegative " if Key == "O2Tol" else "a positive "))
        if Key == "OSmoothing" and Value > 1:
            raise ValueError("OSmoothing must be at most 1")


def LoadJobSpecs(WorkDir: Path) -> List[JobSpec]:
    """Read and validate the entire campaign without creating files.

    Group roots must be distinct, non-overlapping subdirectories of WorkDir.
    Their basenames are the stable IDs used in controller and scheduler logs.
    The legacy single-workspace mode retains its existing validation behavior.
    """
    from workflow import VaspIO as vio

    WorkDir = Path(WorkDir).resolve()
    RootParams = vio.ReadKeyValueFile(WorkDir / "OxParams")
    Entries = ParseJobSpecs(RootParams)
    if not Entries:
        Params = vio.ReadKeyValueFile(WorkDir / "OxParams", RequiredKeys=REQUIRED_KEYS)
        return [JobSpec("", WorkDir, Params)]

    Specs: List[JobSpec] = []
    SeenIds = set()
    for Entry in Entries:
        Relative = Path(Entry)
        if Relative.is_absolute():
            raise ValueError("JobSpecs entries must be relative to the campaign: {!r}".format(Entry))
        Root = (WorkDir / Relative).resolve()
        if Root == WorkDir or WorkDir not in Root.parents:
            raise ValueError("JobSpec must be a subdirectory of the campaign: {!r}".format(Entry))
        Reserved = {"simulationsummary", ".simulation_summary", "logs"}
        if Root.relative_to(WorkDir).parts[0].casefold() in Reserved:
            raise ValueError("JobSpec overlaps a controller output location: {!r}".format(Entry))
        if not Root.is_dir():
            raise ValueError("JobSpec directory does not exist: {}".format(Root))
        SpecId = Root.name
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", SpecId):
            raise ValueError("JobSpec directory name must use letters, digits, '.', '_' or '-': {}".format(SpecId))
        if SpecId.casefold() in SeenIds:
            raise ValueError("Duplicate JobSpec ID: {} (directory basenames must be unique)".format(SpecId))
        if any(Root == Spec.Root or Root in Spec.Root.parents or Spec.Root in Root.parents for Spec in Specs):
            raise ValueError("JobSpec directories must not overlap: {}".format(Root))
        SeenIds.add(SpecId.casefold())
        try:
            Params = vio.ReadKeyValueFile(Root / "OxParams", RequiredKeys=REQUIRED_KEYS)
            if ParseJobSpecs(Params):
                raise ValueError("Nested JobSpecs are not supported")
            ValidateGroupedParams(Params)
            Missing = [Name for Name in INPUT_FILES if not (Root / Name).is_file()]
            if Missing:
                raise ValueError("Missing input files: {}".format(", ".join(Missing)))
        except (OSError, ValueError, SyntaxError, TypeError) as Error:
            raise ValueError("JobSpec {}: {}".format(SpecId, Error)) from Error
        Specs.append(JobSpec(SpecId, Root, Params))
    return Specs


def CheckScientificInputs(Spec: JobSpec) -> dict:
    """Check a grouped spec against its first prepared scientific inputs.

    Selection, stopping time and scheduler commands may change on resubmission.
    Existing workspaces without a record establish their baseline on first use.
    This check is read-only, including when invoked for --dry-run.
    """
    from workflow import VaspIO as vio

    Mutable = {"Temperatures", "NSims", "MaxRuntime", "JobSpecs"}
    Params = {Key: Value for Key, Value in Spec.Params.items() if Key not in Mutable}
    Hashes = {}
    for Name in ("POSCAR", "POTCAR", "INCAR", "KPOINTS", "CovalentRadii"):
        Digest = hashlib.sha256()
        with (Spec.Root / Name).open("rb") as File:
            for Block in iter(lambda: File.read(1024 * 1024), b""):
                Digest.update(Block)
        Hashes[Name] = Digest.hexdigest()
    JobParams = vio.ReadKeyValueFile(Spec.Root / "job.in")
    JobParams.pop("vaspcmd", None)
    Record = {"version": 1, "parameters": Params, "files": Hashes, "job.in": JobParams}
    RecordPath = Spec.Root / INPUT_RECORD
    if RecordPath.exists():
        try:
            Previous = json.loads(RecordPath.read_text(encoding="utf-8"))
        except (OSError, ValueError) as Error:
            raise ValueError("Cannot read scientific input record for {}: {}".format(Spec.Id or Spec.Root.name, Error)) from Error
        if Previous != Record:
            raise ValueError(
                "Scientific inputs changed for JobSpec {}. Create a new specification "
                "directory for changed composition, gas settings or VASP inputs. "
                "Only temperatures, replicas, MaxRuntime and scheduler submission "
                "settings may change when resuming.".format(Spec.Id or Spec.Root.name)
            )
    return Record


def RecordScientificInputs(Spec: JobSpec, Record: dict) -> None:
    """Save the baseline before setup, so partial setup also has provenance."""
    RecordPath = Spec.Root / INPUT_RECORD
    if RecordPath.exists():
        return
    if any(Spec.Root.glob("*/Dir_VolSearch")):
        print("[{}] Existing runs have no input record; recording current inputs "
              "as their baseline. Earlier input changes cannot be verified.".format(Spec.Id))
    Temporary = RecordPath.with_suffix(".json.tmp")
    Temporary.write_text(json.dumps(Record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Temporary.replace(RecordPath)
