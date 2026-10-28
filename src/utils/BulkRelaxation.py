"""Managed bulk preparation for the bundled SLUSCHI optunitcell routine.

optunitcell owns the two-pass sequence. This module derives inputs, executes one
pass at a time, validates VASP outputs, and publishes the accepted geometry.
The source workspace and existing trajectories are never overwritten.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import StatusLog

NAME = "BulkRelaxation"
SCRIPT = Path(__file__).resolve().parents[1] / "dependencies/SLUSCHI_mod/optunitcell"
POLL_SECONDS = 60
GRACE_SECONDS = 120


def Enabled(Params):
    Mode = str(Params.get("Relaxation", "off")).strip().strip("\"'").lower()
    if Mode not in ("off", "bulk"):
        raise ValueError("Relaxation must be off or bulk")
    return Mode == "bulk"


def Digest(Data):
    return hashlib.sha256(Data).hexdigest()


def FileHash(PathFile):
    with Path(PathFile).open("rb") as Handle:
        Hash = hashlib.sha256()
        for Block in iter(lambda: Handle.read(1024 * 1024), b""):
            Hash.update(Block)
    return Hash.hexdigest()


def AtomicText(PathFile, Text):
    PathFile = Path(PathFile)
    Temporary = PathFile.with_name(PathFile.name + ".tmp")
    with Temporary.open("w", encoding="utf-8", newline="\n") as Handle:
        Handle.write(Text)
        Handle.flush()
        os.fsync(Handle.fileno())
    # OneDrive/antivirus may briefly hold the destination open on Windows.
    # Retry the atomic rename; never fall back to a partial direct overwrite of
    # authoritative submission/provenance records.
    for Attempt in range(6):
        try:
            Temporary.replace(PathFile)
            break
        except PermissionError:
            if Attempt == 5:
                raise
            time.sleep(.05 * 2 ** Attempt)


def WriteJson(PathFile, Value):
    AtomicText(PathFile, json.dumps(Value, indent=2, sort_keys=True) + "\n")


def ReadJson(PathFile):
    return json.loads(Path(PathFile).read_text(encoding="utf-8"))


@contextmanager
def Lock(Root):
    """An OS lock survives neither a crash nor a controller restart."""
    with (Root / ".lock").open("a+b") as Handle:
        if os.name == "nt":
            import msvcrt
            Handle.seek(0)
            if not Handle.read(1):
                Handle.write(b"0")
                Handle.flush()
            Handle.seek(0)
            try:
                msvcrt.locking(Handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as Error:
                raise ValueError("Another relaxation controller owns " + str(Root)) from Error
            try:
                yield
            finally:
                Handle.seek(0)
                msvcrt.locking(Handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(Handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as Error:
                raise ValueError("Another relaxation controller owns " + str(Root)) from Error
            try:
                yield
            finally:
                fcntl.flock(Handle, fcntl.LOCK_UN)


def IncarValues(Text):
    Result = {}
    for Line in Text.splitlines():
        for Part in re.split(r"[#!]", Line, maxsplit=1)[0].split(";"):
            if "=" in Part:
                Key, Value = Part.split("=", 1)
                Result[Key.strip().upper()] = Value.strip()
    return Result


def Number(Value, Name, Positive=True):
    Result = float(str(Value).replace("D", "E").replace("d", "e"))
    if not math.isfinite(Result) or (Positive and Result <= 0):
        raise ValueError(Name + " must be finite" + (" and positive" if Positive else ""))
    return Result


def ReadStructure(PathFile):
    """Strict VASP5 reader: keep atom order, constraints, and Cartesian scaling."""
    try:
        Lines = Path(PathFile).read_text(encoding="utf-8").splitlines()
        Scale = float(Lines[1])
        Cell = np.array([list(map(float, L.split()[:3])) for L in Lines[2:5]])
        if Scale < 0:
            Scale = (-Scale / abs(np.linalg.det(Cell))) ** (1 / 3)
        Cell *= Scale
        Species = Lines[5].split()
        Counts = [int(V) for V in Lines[6].split()]
        if len(Species) != len(Counts) or any(C <= 0 for C in Counts):
            raise ValueError("invalid species/counts")
        if any(not re.fullmatch(r"[A-Z][a-z]?", S) for S in Species) or len(set(Species)) != len(Species):
            raise ValueError("requires unique VASP5 element labels")
        Index = 7
        Selective = Lines[Index].strip().lower().startswith("s")
        Index += int(Selective)
        Mode = Lines[Index].strip().lower()
        if not Mode.startswith(("d", "c", "k")):
            raise ValueError("missing coordinate mode")
        Rows = [L.split() for L in Lines[Index + 1:Index + 1 + sum(Counts)]]
        Frac = np.array([list(map(float, L[:3])) for L in Rows])
        Flags = np.ones((sum(Counts), 3), dtype=bool)
        if Selective:
            if any(len(L) < 6 or any(V.upper() not in ("T", "F") for V in L[3:6]) for L in Rows):
                raise ValueError("invalid selective dynamics")
            Flags = np.array([[V.upper() == "T" for V in L[3:6]] for L in Rows])
        if not Mode.startswith("d"):
            Frac = (Frac * Scale) @ np.linalg.inv(Cell)
        if Frac.shape != (sum(Counts), 3) or not np.isfinite(Frac).all() or not np.isfinite(Cell).all():
            raise ValueError("incomplete/nonfinite coordinates")
        if np.linalg.det(Cell) <= 0:
            raise ValueError("invalid cell volume")
        return Species, Counts, Cell, Frac % 1.0, Flags
    except (ValueError, IndexError, np.linalg.LinAlgError) as Error:
        raise ValueError("Invalid structure {}: {}".format(PathFile, Error)) from Error


def StructureText(Structure):
    Species, Counts, Cell, Frac, Flags = Structure
    Text = "SGUSCHI bulk relaxation (velocities omitted)\n1.0\n"
    Text += "".join(" ".join("{:.16f}".format(V) for V in Row) + "\n" for Row in Cell)
    Text += " ".join(Species) + "\n" + " ".join(map(str, Counts)) + "\n"
    Selective = not Flags.all()
    if Selective:
        Text += "Selective dynamics\n"
    Text += "Direct\n"
    for Row, Mask in zip(Frac, Flags):
        Text += " ".join("{:.16f}".format(V) for V in Row)
        if Selective:
            Text += " " + " ".join("T" if V else "F" for V in Mask)
        Text += "\n"
    return Text


def BulkPotential(Text, Species):
    Blocks = re.findall(r".*?End of Dataset[^\n]*\n?", Text, flags=re.S)
    Selected = {}
    for Block in Blocks:
        Match = re.search(r"TITEL\s*=\s*\S+\s+([A-Z][a-z]?)(?:[_\s])", Block)
        if not Match:
            raise ValueError("Cannot identify POTCAR block from TITEL")
        Element = Match[1]
        if Element in Selected:
            raise ValueError("Duplicate POTCAR block for " + Element)
        Selected[Element] = Block
    if any(S not in Selected for S in Species):
        raise ValueError("POTCAR must contain a complete block for each bulk species")
    return "".join(Selected[S] for S in Species)


def BuildInputs(SpecRoot, Params):
    """Read-only preflight; generate exactly the inputs the routine will consume."""
    from workflow.VaspIO import ReadKeyValueFile
    SpecRoot = Path(SpecRoot)
    Structure = ReadStructure(SpecRoot / "POSCAR")
    Species, Counts, Cell, Frac, Flags = Structure
    if "O" in Species:
        raise ValueError("Automatic bulk relaxation requires an oxygen-free input")
    if not np.allclose(Cell, np.diag(np.diag(Cell)), atol=1e-8) or (np.diag(Cell) <= 0).any():
        raise ValueError("Automatic bulk preparation currently requires a positive orthogonal cell")
    # Existing oxidation preparation does not carry selective-dynamics flags.
    if not Flags.all():
        raise ValueError("Constrained atoms are not supported by automatic bulk-to-MD preparation")
    Force = Number(Params.get("RelaxForceTol", .02), "RelaxForceTol")
    Pressure = Number(Params.get("RelaxPressureTol", 1), "RelaxPressureTol")
    Steps = Number(Params.get("RelaxMaxSteps", 200), "RelaxMaxSteps")
    if Steps != int(Steps):
        raise ValueError("RelaxMaxSteps must be an integer")
    Values = IncarValues((SpecRoot / "INCAR").read_text())
    EDiff = min(Number(Values.get("EDIFF", "1e-4"), "EDIFF"), 1e-6)
    Nelm = max(int(Values.get("NELM", 60)), 200)
    JobParams = ReadKeyValueFile(SpecRoot / "job.in")
    Target = Number(JobParams.get("press", 0), "press", Positive=False)
    for Key in ("TEBEG", "TEEND", "SMASS", "MDALGO", "LANGEVIN_GAMMA", "LANGEVIN_GAMMA_L", "PMASS", "ANDERSEN_PROB", "NBLOCK", "KBLOCK", "NPACO", "APACO"):
        Values.pop(Key, None)
    Values.update(IBRION="2", ISIF="8", POTIM="0.3", PREC="Accurate", EDIFF=str(EDiff),
                  EDIFFG=str(-Force), ISMEAR="0", SIGMA="0.05", NELM=str(Nelm),
                  NSW=str(int(Steps)), PSTRESS=str(Target), ISTART="0", ICHARG="2", NWRITE="2")
    if "RelaxENCUT" in Params:
        Values["ENCUT"] = str(Number(Params["RelaxENCUT"], "RelaxENCUT"))
    Kpoints = (SpecRoot / "KPOINTS").read_text()
    if "RelaxKMesh" in Params:
        Mesh = ast.literal_eval(str(Params["RelaxKMesh"]))
        if not isinstance(Mesh, list) or len(Mesh) != 3 or any(type(N) is not int or N < 1 for N in Mesh):
            raise ValueError("RelaxKMesh must be three positive integers")
        Kpoints = "SGUSCHI relaxation mesh\n0\nGamma\n{}\n0 0 0\n".format(" ".join(map(str, Mesh)))
    Command = shlex.split(JobParams.get("vaspcmd", "sbatch"))
    if not Command or Path(Command[0]).name not in ("sbatch", "qsub"):
        raise ValueError("Managed relaxation supports sbatch or qsub via vaspcmd")
    Job = (SpecRoot / "jobsub").read_text()
    Label = "opt_" + re.sub(r"[^A-Za-z0-9_.-]", "_", SpecRoot.name)
    Job = re.sub(r"(?m)^#SBATCH\s+(?:--job-name(?:=|\s+)|-J\s+).*\n?", "", Job)
    Job = re.sub(r"(?m)^#PBS\s+-N\s+.*\n?", "", Job)
    Directive = "#SBATCH --job-name=" + Label if Path(Command[0]).name == "sbatch" else "#PBS -N " + Label
    Lines = Job.splitlines(keepends=True)
    if not Lines or not Lines[0].startswith("#!"):
        raise ValueError("jobsub requires a shebang")
    Lines.insert(1, Directive + "\n")
    Job = "".join(Lines)
    Walltime = str(Params.get("RelaxWalltime", "")).strip().strip("\"'")
    if Walltime:
        if not re.fullmatch(r"(?:\d+-)?\d+:\d{2}:\d{2}", Walltime):
            raise ValueError("RelaxWalltime must be [days-]HH:MM:SS")
        if Path(Command[0]).name == "sbatch":
            Job = re.sub(r"(?m)^#SBATCH\s+(?:--time(?:=|\s+)|-t\s+).*\n?", "", Job)
            Job = Job.replace("\n", "\n#SBATCH --time=" + Walltime + "\n", 1)
        else:
            if "-" in Walltime:
                raise ValueError("PBS RelaxWalltime uses hours:minutes:seconds")
            if re.search(r"walltime=", Job):
                Job = re.sub(r"walltime=[^,\s]+", "walltime=" + Walltime, Job)
            else:
                Job = Job.replace("\n", "\n#PBS -l walltime=" + Walltime + "\n", 1)
    Files = {"POSCAR": StructureText(Structure), "INCAR": "".join(K + " = " + V + "\n" for K, V in Values.items()),
             "POTCAR": BulkPotential((SpecRoot / "POTCAR").read_text(), Species),
             "KPOINTS_OPT": Kpoints, "jobsub": Job, "job.in": (SpecRoot / "job.in").read_text()}
    Config = {"force": Force, "pressure": Pressure, "steps": int(Steps), "target_pressure": Target, "nelm": Nelm}
    Scientific = {Name: Digest(Files[Name].encode()) for Name in ("POSCAR", "INCAR", "POTCAR", "KPOINTS_OPT")}
    Scientific["config"] = Config
    Fingerprint = Digest(json.dumps(Scientific, sort_keys=True).encode())
    return {"files": Files, "fingerprint": Fingerprint, "config": Config, "command": Command,
            "source_hashes": {Name: FileHash(SpecRoot / Name) for Name in ("POSCAR", "INCAR", "KPOINTS", "POTCAR")}}


def CheckPlan(SpecRoot, Plan):
    Root = Path(SpecRoot) / NAME
    if (Root / "Inputs.json").exists():
        if ReadJson(Root / "Inputs.json")["fingerprint"] != Plan["fingerprint"]:
            raise ValueError("Relaxation inputs changed; use a fresh specification: " + str(SpecRoot))
    elif any(Path(SpecRoot).glob("*/Dir_VolSearch")):
        raise ValueError("Cannot retrofit relaxation onto existing trajectories: " + str(SpecRoot))
    elif Root.exists() and any(P.name != ".lock" for P in Root.iterdir()):
        raise ValueError("Unrecorded BulkRelaxation directory; inspect before reuse: " + str(Root))
    if (Root / "Status.json").exists() and ReadJson(Root / "Status.json").get("state") == "accepted":
        AcceptedHash(SpecRoot)


def Prepare(SpecRoot, Plan):
    Root = Path(SpecRoot) / NAME
    Root.mkdir(exist_ok=True)
    with Lock(Root):
        CheckPlan(SpecRoot, Plan)
        if (Root / "Inputs.json").exists():
            # Resource changes affect only a pass that has not been prepared.
            for Name in ("jobsub", "job.in"):
                if (Root / Name).read_text() != Plan["files"][Name]:
                    AtomicText(Root / Name, Plan["files"][Name])
            Inputs = ReadJson(Root / "Inputs.json")
            if Inputs["command"] != Plan["command"]:
                Inputs["command"] = Plan["command"]
                WriteJson(Root / "Inputs.json", Inputs)
            return
        for Name, Text in Plan["files"].items():
            AtomicText(Root / Name, Text)
        (Root / "Dir_OptUnitCell").mkdir(exist_ok=True)
        Inputs = {K: V for K, V in Plan.items() if K != "files"}
        Inputs["generated_hashes"] = {N: FileHash(Root / N) for N in ("POSCAR", "INCAR", "POTCAR", "KPOINTS_OPT")}
        WriteJson(Root / "Inputs.json", Inputs)
        SetStatus(Root, "prepared", 0, "Inputs prepared; no VASP job submitted")


def SetStatus(Root, State, Pass, Detail, **Extra):
    Record = {"state": State, "pass": Pass, "detail": Detail, "updated": time.time()}
    Record.update(Extra)
    WriteJson(Root / "Status.json", Record)
    StatusLog.Append(Root, "optunitcell", State, "pass {}: {}".format(Pass, Detail))


def VerifyGenerated(Root):
    Inputs = ReadJson(Root / "Inputs.json")
    for Name, Expected in Inputs["generated_hashes"].items():
        if FileHash(Root / Name) != Expected:
            raise ValueError("Generated relaxation input changed: " + Name)
    return Inputs


def VerifyCompleted(Root, Pass):
    Folder = Root / "Dir_OptUnitCell" / ("pass" + str(Pass))
    Record = ReadJson(Folder / "Pass.json")
    if Record["state"] != "complete":
        raise ValueError("Pass {} is not complete".format(Pass))
    for Name, Expected in Record["metrics"]["hashes"].items():
        if FileHash(Folder / Name) != Expected:
            raise ValueError("Completed pass output/input changed: " + str(Folder / Name))
    return Record


def ValidatePass(Root, Pass):
    Inputs = VerifyGenerated(Root)
    Folder = Root / "Dir_OptUnitCell" / ("pass" + str(Pass))
    Config = Inputs["config"]
    Submitted = ReadJson(Folder / "Pass.json")
    for Name, Expected in Submitted["inputs"].items():
        if FileHash(Folder / Name) != Expected:
            raise ValueError("Submitted input changed during VASP: " + Name)
    Text = (Folder / "OUTCAR").read_text(errors="replace")
    if "General timing and accounting" not in Text or "Total CPU" not in Text:
        raise ValueError("VASP has not finished normally")
    Version = re.search(r"vasp\.(\d+)\.(\d+)\.(\d+)", Text, re.I)
    if not Version or tuple(map(int, Version.groups())) < (6, 4, 1):
        raise ValueError("ISIF=8 requires identifiable VASP >= 6.4.1")
    Original = ReadStructure(Root / "POSCAR")
    Result = ReadStructure(Folder / "CONTCAR")
    if Result[:2] != Original[:2] or not np.array_equal(Result[4], Original[4]):
        raise ValueError("Relaxed atom inventory/order or constraints changed")
    Scale = (np.linalg.det(Result[2]) / np.linalg.det(Original[2])) ** (1 / 3)
    if not np.allclose(Result[2], Original[2] * Scale, atol=1e-5, rtol=1e-6):
        raise ValueError("Relaxation changed cell shape or aspect ratio")
    Oszicar = (Folder / "OSZICAR").read_text(errors="replace")
    N = 0
    Steps = 0
    for Line in Oszicar.splitlines():
        Match = re.match(r"\s*(?:DAV|RMM|CG|DMP):\s*(\d+)", Line)
        if Match:
            N = int(Match[1])
        if re.match(r"\s*\d+\s+F=", Line):
            Steps += 1
            if not N or N >= Config["nelm"]:
                raise ValueError("Missing electronic history or iteration cap at ionic step " + str(Steps))
            N = 0
    if not Steps or N:
        raise ValueError("Missing or incomplete ionic history")
    Lines = Text.splitlines()
    Force = None
    for Index, Line in enumerate(Lines):
        if "POSITION" in Line and "TOTAL-FORCE" in Line:
            try:
                Block = np.array([list(map(float, L.split()[:6])) for L in Lines[Index + 2:Index + 2 + sum(Result[1])]])
                Force = np.linalg.norm(Block[:, 3:6], axis=1).max() if Block.shape == (sum(Result[1]), 6) else None
            except ValueError:
                Force = None
    Pressure = re.findall(r"external pressure\s*=\s*([-+\d.Ee]+)", Text)
    if Force is None or not np.isfinite(Force) or not Pressure:
        raise ValueError("Missing final forces or pressure")
    Pressure = float(Pressure[-1])
    if not math.isfinite(Pressure):
        raise ValueError("Nonfinite final pressure")
    # VASP's external pressure is the trace of the computed stress; PSTRESS is
    # the imposed target. Compare the computed pressure with that target.
    Residual = Pressure - Config["target_pressure"]
    if Pass == 2:
        if "reached required accuracy" not in Text or Force > Config["force"] + 1e-5:
            raise ValueError("Final ionic convergence/force criterion not met")
        if abs(Residual) > Config["pressure"]:
            raise ValueError("Final residual pressure exceeds tolerance")
    Energy = re.findall(r"free\s+energy\s+TOTEN\s*=\s*([-+\d.Ee]+)", Text)
    return {"ionic_steps": Steps, "max_force_eVA": float(Force), "pressure_kbar": Pressure,
            "residual_pressure_kbar": Residual, "volume_A3": float(np.linalg.det(Result[2])),
            "energy_eV": float(Energy[-1]) if Energy else None,
            "hashes": {N: FileHash(Folder / N) for N in ("POSCAR", "INCAR", "KPOINTS", "POTCAR", "OUTCAR", "OSZICAR", "CONTCAR")}}


def QueryJob(Command, JobId):
    """Return active, gone, or unknown; never infer failure from a query error."""
    try:
        if Path(Command[0]).name == "sbatch":
            Result = subprocess.run(["squeue", "-h", "-j", JobId, "-o", "%i"], capture_output=True, text=True, timeout=15)
            if Result.returncode == 0:
                return "active" if Result.stdout.strip() else "gone"
            # Older Slurm releases return an error for a job no longer in the
            # active queue. Accounting distinguishes this from a query outage.
            Result = subprocess.run(["sacct", "-n", "-P", "-j", JobId, "-o", "JobIDRaw,State"],
                                    capture_output=True, text=True, timeout=15)
            if Result.returncode == 0:
                for Line in Result.stdout.splitlines():
                    Parts = Line.strip().split("|")
                    if len(Parts) >= 2 and Parts[0] == JobId:
                        State = Parts[1].split()[0].rstrip("+")
                        if State in ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY",
                                     "PREEMPTED", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "REVOKED"):
                            return "gone"
                        if State in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED",
                                     "RESIZING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD"):
                            return "active"
        else:
            Result = subprocess.run(["qstat", "-f", JobId], capture_output=True, text=True, timeout=15)
            if Result.returncode == 0:
                Match = re.search(r"job_state\s*=\s*(\w)", Result.stdout)
                if Match:
                    return "gone" if Match[1] in ("C", "F") else "active"
            if "unknown job" in (Result.stdout + Result.stderr).lower():
                return "gone"
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def JobIdFrom(Text):
    Match = re.search(r"Submitted batch job (\d+)", Text)
    if Match:
        return Match[1]
    Match = re.fullmatch(r"\s*(\d+(?:\.[A-Za-z0-9_.-]+)?)(?:;[\w.-]+)?\s*", Text)
    return Match[1] if Match else None


def RunPass(Root, Pass):
    Root = Path(Root)
    with Lock(Root):
        Inputs = VerifyGenerated(Root)
        Folder = Root / "Dir_OptUnitCell" / ("pass" + str(Pass))
        Folder.mkdir(exist_ok=True)
        RecordPath = Folder / "Pass.json"
        if RecordPath.exists():
            Record = ReadJson(RecordPath)
            if Record["state"] == "complete":
                VerifyCompleted(Root, Pass)
                return
            if Record["state"] != "submitted":
                raise ValueError("Pass is {}: inspect before a new attempt".format(Record["state"]))
        else:
            if any(Folder.iterdir()):
                raise ValueError("Unrecorded pass files; refusing to overwrite " + str(Folder))
            if Pass == 2:
                VerifyCompleted(Root, 1)
                ValidatePass(Root, 1)
            for Name in ("INCAR", "POTCAR", "jobsub"):
                shutil.copyfile(Root / Name, Folder / Name)
            shutil.copyfile(Root / "KPOINTS_OPT", Folder / "KPOINTS")
            Source = Root / "POSCAR" if Pass == 1 else Root / "Dir_OptUnitCell/pass1/CONTCAR"
            AtomicText(Folder / "POSCAR", StructureText(ReadStructure(Source)))
            Record = {"state": "submitting", "submitted_at": time.time(), "pass": Pass, "command": Inputs["command"],
                      "inputs": {N: FileHash(Folder / N) for N in ("POSCAR", "INCAR", "POTCAR", "KPOINTS", "jobsub")}}
            # Persist intent BEFORE calling sbatch: uncertainty must never cause a duplicate.
            WriteJson(RecordPath, Record)
            SetStatus(Root, "submitting", Pass, "Submitting VASP")
            try:
                Result = subprocess.run(Inputs["command"] + ["jobsub"], cwd=Folder, capture_output=True, text=True, timeout=60)
                AtomicText(Folder / "Submission.log", Result.stdout + Result.stderr)
                JobId = JobIdFrom(Result.stdout)
                if JobId:
                    Record["job_id"] = JobId
                if Result.returncode != 0 or not JobId:
                    Record["state"] = "uncertain" if JobId or Result.returncode == 0 or "timed out" in (Result.stdout + Result.stderr).lower() else "rejected"
                    WriteJson(RecordPath, Record)
                    raise ValueError("Submission {}: see {}".format(Record["state"], Folder / "Submission.log"))
            except (OSError, subprocess.TimeoutExpired) as Error:
                Record["state"] = "uncertain" if isinstance(Error, subprocess.TimeoutExpired) else "rejected"
                WriteJson(RecordPath, Record)
                raise ValueError("Submission {}: {}".format(Record["state"], Error)) from Error
            Record.update(state="submitted", job_id=JobId)
            WriteJson(RecordPath, Record)
        for Name, Expected in Record["inputs"].items():
            if FileHash(Folder / Name) != Expected:
                raise ValueError("Submitted input changed: " + str(Folder / Name))
        SetStatus(Root, "running", Pass, "Waiting for VASP", job_id=Record["job_id"])
        GoneSince = None
        while True:
            Outcar = Folder / "OUTCAR"
            Text = Outcar.read_text(errors="replace") if Outcar.exists() else ""
            if "General timing and accounting" in Text and "Total CPU" in Text:
                break
            Queue = QueryJob(Record["command"], Record["job_id"])
            if Queue == "unknown":
                SetStatus(Root, "awaiting", Pass, "Scheduler status unavailable; job will not be resubmitted", job_id=Record["job_id"])
                raise ValueError("Scheduler status unavailable; resume controller to recheck the recorded job")
            if Queue == "gone":
                GoneSince = GoneSince or time.time()
                if time.time() - GoneSince >= GRACE_SECONDS:
                    Record["state"] = "failed"
                    WriteJson(RecordPath, Record)
                    raise ValueError("Scheduler job ended without complete VASP outputs")
            else:
                GoneSince = None
            time.sleep(POLL_SECONDS)
        try:
            Metrics = ValidatePass(Root, Pass)
        except (OSError, ValueError) as Error:
            Record.update(state="failed", error=str(Error))
            WriteJson(RecordPath, Record)
            raise
        Record.update(state="complete", metrics=Metrics)
        WriteJson(RecordPath, Record)
        SetStatus(Root, "pass_complete", Pass, "Pass validated", job_id=Record["job_id"], metrics=Metrics)


def Accept(Root):
    Root = Path(Root)
    with Lock(Root):
        VerifyGenerated(Root)
        for Pass in (1, 2):
            Record = VerifyCompleted(Root, Pass)
        Metrics = ValidatePass(Root, 2)
        Original = ReadStructure(Root / "POSCAR")
        Result = ReadStructure(Root / "Dir_OptUnitCell/pass2/CONTCAR")
        from preprocessing.OxidationPreprocessing import PreservedSlabCoordinates
        # Validate the cut before publishing any accepted structure.
        PreservedSlabCoordinates(Original[3], Result[3])
        AtomicText(Root / "Accepted.POSCAR", StructureText(Result))
        SetStatus(Root, "accepted", 2, "Bulk relaxation accepted", metrics=Metrics,
                  job_id=Record["job_id"],
                  accepted_sha256=FileHash(Root / "Accepted.POSCAR"),
                  fingerprint=ReadJson(Root / "Inputs.json")["fingerprint"])
        AtomicText(Root / "Dir_OptUnitCell/optunitcell_is_done", "Validated by SGUSCHI\n")


def AcceptedHash(SpecRoot):
    Root = Path(SpecRoot) / NAME
    if not (Root / "Status.json").exists():
        return None
    Status = ReadJson(Root / "Status.json")
    if Status.get("state") != "accepted":
        return None
    Inputs = VerifyGenerated(Root)
    if Status["fingerprint"] != Inputs["fingerprint"] or FileHash(Root / "Accepted.POSCAR") != Status["accepted_sha256"]:
        raise ValueError("Accepted relaxation geometry/provenance changed")
    for Pass in (1, 2):
        VerifyCompleted(Root, Pass)
    return Status["accepted_sha256"]


def Geometry(SpecRoot):
    """Return the explicit relaxed input plus the original cleavage reference."""
    from workflow import VaspIO
    Params = VaspIO.ReadKeyValueFile(Path(SpecRoot) / "OxParams")
    CheckPlan(SpecRoot, BuildInputs(SpecRoot, Params))
    if not AcceptedHash(SpecRoot):
        raise ValueError("Bulk relaxation is not accepted: " + str(SpecRoot))
    Root = Path(SpecRoot) / NAME
    Position, Cell = VaspIO.ReadPoscar(Root, "Accepted.POSCAR")
    Original = ReadStructure(Root / "POSCAR")[3]
    return Position, Cell, Original


def Advance(Specs, Args):
    """Preflight every spec; run optunitcell concurrently; return eligible specs."""
    Plans = {}
    for Spec in Specs:
        if Enabled(Spec.Params):
            Plan = BuildInputs(Spec.Root, Spec.Params)
            CheckPlan(Spec.Root, Plan)
            Plans[Spec.Id] = Plan
    if not Plans:
        if getattr(Args, "relax_only", False):
            print("No bulk relaxation enabled (set Relaxation = bulk in each specification).")
        return Specs, False
    if Args.dry_run:
        for Spec in Specs:
            if Spec.Id in Plans:
                Action = "reuse accepted bulk" if AcceptedHash(Spec.Root) else "run two-pass optunitcell before trajectory setup"
                print("[{}] Would {} (MD cutoff/mesh inherited unless overridden).".format(Spec.Id or Spec.Root.name, Action))
        # Only accepted/off specs can already prepare trajectories in this dry plan.
        return [S for S in Specs if S.Id not in Plans or AcceptedHash(S.Root)], False
    for Spec in Specs:
        if Spec.Id in Plans:
            Prepare(Spec.Root, Plans[Spec.Id])
            print("[{}] Bulk relaxation: {}".format(Spec.Id or Spec.Root.name,
                  ReadJson(Spec.Root / NAME / "Status.json")["state"]))
    if Args.prepare_only:
        return [S for S in Specs if S.Id not in Plans or AcceptedHash(S.Root)], False
    Processes = []
    PreviousSignal = signal.getsignal(signal.SIGTERM)
    def Stop(Signum, Frame):
        raise SystemExit(128 + Signum)
    signal.signal(signal.SIGTERM, Stop)
    try:
        for Spec in Specs:
            if Spec.Id not in Plans or AcceptedHash(Spec.Root):
                continue
            Root = Spec.Root / NAME
            Status = ReadJson(Root / "Status.json")
            if Status["state"] in ("failed", "uncertain", "rejected"):
                print("[{}] Relaxation requires review: {}".format(Spec.Id, Status["detail"]))
                continue
            Csh = shutil.which("csh") or shutil.which("tcsh")
            if not Csh:
                raise ValueError("C-shell is required to run SLUSCHI optunitcell")
            Env = os.environ.copy()
            Env.update(sguschipath=str(SCRIPT.parent), SGUSCHI_PYTHON=sys.executable)
            Log = (Root / "optunitcell.log").open("a", encoding="utf-8")
            try:
                try:
                    Process = subprocess.Popen([Csh, "-f", str(SCRIPT), "--sguschi"],
                                               cwd=Root / "Dir_OptUnitCell", env=Env,
                                               stdout=Log, stderr=subprocess.STDOUT, start_new_session=(os.name != "nt"))
                except OSError as Error:
                    with Lock(Root):
                        SetStatus(Root, "failed", 0, "Could not launch optunitcell: " + str(Error))
                    continue
            finally:
                Log.close()
            Processes.append((Spec, Process))
        from utils.SimulationSummary import WriteSummaryOnce
        def RefreshSummary():
            try:
                WriteSummaryOnce(Path(Args.workdir).resolve(), Quiet=True)
            except (OSError, ValueError) as Error:
                # The per-specification Status.json is authoritative. An
                # advisory summary failure must not stop running workers.
                print("Could not refresh relaxation summary: " + str(Error))
        LastSummary = 0
        while any(P.poll() is None for _, P in Processes):
            if time.monotonic() - LastSummary > 60:
                RefreshSummary()
                LastSummary = time.monotonic()
            time.sleep(1)
    except BaseException:
        for _, P in Processes:
            if P.poll() is None:
                if os.name == "nt":
                    P.terminate()
                else:
                    os.killpg(P.pid, signal.SIGTERM)
        raise
    finally:
        signal.signal(signal.SIGTERM, PreviousSignal)
    Eligible = [S for S in Specs if S.Id not in Plans or AcceptedHash(S.Root)]
    RefreshSummary()
    for Spec in Specs:
        if Spec not in Eligible:
            print("[{}] Relaxation not accepted; see {}/Status.json".format(Spec.Id or Spec.Root.name, Spec.Root / NAME))
    return Eligible, len(Eligible) != len(Specs)


def Main():
    Parser = argparse.ArgumentParser(description=__doc__)
    Parser.add_argument("action", choices=("pass", "accept"))
    Parser.add_argument("number", nargs="?", type=int, choices=(1, 2))
    Args = Parser.parse_args()
    Root = Path.cwd().parent  # called by optunitcell from Dir_OptUnitCell
    try:
        if Args.action == "pass":
            if Args.number is None:
                raise ValueError("pass requires 1 or 2")
            RunPass(Root, Args.number)
        else:
            Accept(Root)
    except (OSError, ValueError, KeyError) as Error:
        print("FATAL optunitcell: " + str(Error), flush=True)
        # Do not race another controller's authoritative status when lock acquisition failed.
        if "Another relaxation controller" not in str(Error):
            try:
                with Lock(Root):
                    Existing = ReadJson(Root / "Status.json") if (Root / "Status.json").exists() else {}
                    if Existing.get("state") != "awaiting":
                        SetStatus(Root, "failed", Args.number or 2, str(Error))
            except (OSError, ValueError):
                pass
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())
