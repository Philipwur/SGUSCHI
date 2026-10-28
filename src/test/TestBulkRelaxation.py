"""Bulk preparation tests: no licensed inputs, VASP executable or real scheduler."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import SGUSCHI
from utils import BulkRelaxation as B
from utils.JobSpecs import CheckScientificInputs, JobSpec, LoadJobSpecs, RecordScientificInputs
from utils.SimulationSummary import FormatRelaxations
from workflow import VaspIO
from preprocessing.OxidationPreprocessing import AddVacuum, PreservedSlabCoordinates


def WriteSpec(Root):
    Root.mkdir(parents=True, exist_ok=True)
    (Root / "POSCAR").write_text(
        "bulk\n1\n12 0 0\n0 8 0\n0 0 8\nZr C\n2 2\nDirect\n"
        ".125 0 0\n.625 0 0\n.125 .5 .5\n.625 .5 .5\n")
    (Root / "POTCAR").write_text("".join(
        "TITEL = PAW_PBE {} 01Jan2000\n ENMAX = 300;\nEnd of Dataset\n".format(S)
        for S in ("Zr_sv", "C", "O")))
    (Root / "INCAR").write_text("ENCUT = 400\nIBRION = 0\nISIF = 2\nNSW = 80\n"
                               "TEBEG = 973; TEEND=973\nSMASS=0\nEDIFF=1e-5\n"
                               "SIGMA=.08\nISMEAR=-1\nPREC=Normal\nNELM=80\n"
                               "ISPIN=1\nAMIX=.2\nLREAL=.FALSE.\n")
    (Root / "KPOINTS").write_text("MD mesh\n1\nReciprocal\n.25 .25 .25 1\n")
    (Root / "job.in").write_text("temp=973\npress=0\nvaspcmd=sbatch\n")
    (Root / "jobsub").write_text("#!/bin/bash\n#SBATCH --job-name=old\n#SBATCH --time=03:00:00\n"
                                "#SBATCH --ntasks=48\nmodule load example\nsrun vasp_std\n")
    (Root / "CovalentRadii").write_text("Zr = 1.45\nC = .76\nO = .66\n")
    (Root / "OxParams").write_text("Temperatures=[973,1273]\nNSims=2\nGasRatio=2\nInitO2Count=1\n"
                                   "AtomicRadiusTol=1.5\nO2Tol=.5\nOSmoothing=.001\nRelaxation=bulk\n")
    return VaspIO.ReadKeyValueFile(Root / "OxParams")


@pytest.fixture
def Spec(tmp_path):
    Root = tmp_path / "case"
    Params = WriteSpec(Root)
    return Root, Params


def Outputs(Folder, Force=.01, Pressure=0, Version="6.6.1", Converged=True, Cap=False):
    Data = B.ReadStructure(Folder / "POSCAR")
    (Folder / "CONTCAR").write_text(B.StructureText(Data))
    Text = "vasp.{}\n".format(Version)
    Text += "POSITION                                       TOTAL-FORCE (eV/Angst)\n-----\n"
    Text += "".join("0 0 0 {} 0 0\n".format(Force) for _ in range(sum(Data[1])))
    Text += "external pressure = {} kB\nfree energy TOTEN = -123.45 eV\n".format(Pressure)
    if Converged:
        Text += "reached required accuracy - stopping structural energy minimisation\n"
    Text += "General timing and accounting informations for this job:\nTotal CPU time used (sec): 1\n"
    (Folder / "OUTCAR").write_text(Text)
    (Folder / "OSZICAR").write_text("DAV: {} -100 .0 .0\n 1 F= -100 E0= -100\n".format(200 if Cap else 3))


def Complete(Spec, monkeypatch):
    Root, Params = Spec
    B.Prepare(Root, B.BuildInputs(Root, Params))
    Calls = []
    def Submit(Command, cwd=None, **kwargs):
        assert Path(Command[0]).name == "sbatch"
        Calls.append(Path(cwd))
        Outputs(Path(cwd))
        return SimpleNamespace(returncode=0, stdout="Submitted batch job {}\n".format(100 + len(Calls)), stderr="")
    monkeypatch.setattr(B.subprocess, "run", Submit)
    for Pass in (1, 2):
        B.RunPass(Root / B.NAME, Pass)
    B.Accept(Root / B.NAME)
    return Calls


def TestDerivationKeepsMdPrecisionAndOriginals(Spec):
    Root, Params = Spec
    Before = {P.name: P.read_bytes() for P in Root.iterdir()}
    Plan = B.BuildInputs(Root, Params)
    assert Plan["files"]["KPOINTS"] == (Root / "KPOINTS").read_text()
    assert "job.in" not in Plan["files"] and "KPOINTS_OPT" not in Plan["files"]
    Values = B.IncarValues(Plan["files"]["INCAR"])
    assert Values["ENCUT"] == "400" and Values["ISIF"] == "8"
    assert Values["IBRION"] == "2" and Values["NSW"] == "200"
    assert Values["EDIFFG"] == "-0.02" and Values["AMIX"] == ".2"
    Original = B.IncarValues((Root / "INCAR").read_text())
    for Key in ("ENCUT", "PREC", "EDIFF", "NELM", "ISMEAR", "SIGMA", "ISPIN", "AMIX", "LREAL"):
        assert Values[Key] == Original[Key]
    assert Plan["config"]["nelm"] == 80
    assert not {"TEBEG", "TEEND", "SMASS"} & Values.keys()
    assert "PAW_PBE O " not in Plan["files"]["POTCAR"]
    assert "PAW_PBE Zr_sv" in Plan["files"]["POTCAR"]
    B.Prepare(Root, Plan)
    assert all((Root / N).read_bytes() == V for N, V in Before.items())
    assert not list(Root.glob("*/Dir_VolSearch"))
    assert not (Root / B.NAME / "job.in").exists()
    assert not (Root / B.NAME / "KPOINTS_OPT").exists()


def TestRelaxationPatchesOnlyControlTagsAndKeepsComments(Spec):
    Root, Params = Spec
    Original = "# IBRION=0 is a comment\nencut=400; PREC=Normal ! supplied precision\n"
    Original += "ibrion=0; ISMEAR=-1; TEBEG=973 # shared line\nIBRION=1\n"
    Original += "SMASS=0; SIGMA=.2064\nEDIFF=1e-4\nNCORE=8"
    (Root / "INCAR").write_text(Original)
    Text = B.BuildInputs(Root, Params)["files"]["INCAR"]
    assert "encut=400; PREC=Normal ! supplied precision\n" in Text
    assert "# IBRION=0 is a comment\n" in Text and "# shared line" in Text
    assert "ISMEAR=-1" in Text and "SIGMA=.2064" in Text
    assert "EDIFF=1e-4\nNCORE=8\n" in Text
    assert B.IncarValues(Text)["IBRION"] == "2"
    assert Text.count("IBRION = 2") == 1
    assert not {"TEBEG", "SMASS"} & B.IncarValues(Text).keys()
    assert (Root / "INCAR").read_text() == Original


def TestSchedulerUpdateNeedsNoStagedJobIn(Spec):
    Root, Params = Spec
    Plan = B.BuildInputs(Root, Params)
    B.Prepare(Root, Plan)
    (Root / "job.in").write_text("temp=873\npress=0\nvaspcmd=sbatch --parsable\n")
    Updated = B.BuildInputs(Root, Params)
    assert Updated["fingerprint"] == Plan["fingerprint"]
    B.Prepare(Root, Updated)
    assert B.ReadJson(Root / B.NAME / "Inputs.json")["command"] == ["sbatch", "--parsable"]
    assert not (Root / B.NAME / "job.in").exists()
    (Root / "job.in").write_text("press=3\nvaspcmd=sbatch --parsable\n")
    with pytest.raises(ValueError, match="inputs changed"):
        B.CheckPlan(Root, B.BuildInputs(Root, Params))


def TestWalltimeAndNoTemperatureSpecificRelaxation(Spec):
    Root, Params = Spec
    Params.update(RelaxWalltime="24:00:00")
    Plan = B.BuildInputs(Root, Params)
    assert "#SBATCH --time=24:00:00" in Plan["files"]["jobsub"]
    assert "--time=03" not in Plan["files"]["jobsub"]
    assert "--ntasks=48" in Plan["files"]["jobsub"]
    Params.update(Temperatures="[873]", NSims="8")
    assert B.BuildInputs(Root, Params)["fingerprint"] == Plan["fingerprint"]


def TestOmittedElectronicSettingsRemainOmitted(Spec):
    Root, Params = Spec
    (Root / "INCAR").write_text("IBRION=0\nNSW=80\nPOTIM=1\nPSTRESS=2.5\nNWRITE=3\n")
    Plan = B.BuildInputs(Root, Params)
    Values = B.IncarValues(Plan["files"]["INCAR"])
    assert not {"ENCUT", "PREC", "EDIFF", "NELM", "ISMEAR", "SIGMA"} & Values.keys()
    assert Plan["config"]["nelm"] == 60
    assert Plan["config"]["target_pressure"] == 2.5
    assert Values["NWRITE"] == "3"


@pytest.mark.parametrize("Key,Value", [("RelaxENCUT", "700"), ("RelaxKMesh", "[3,4,4]")])
def TestRemovedPrecisionOverridesAreNotSilentlyUsed(Spec, Key, Value):
    Root, Params = Spec
    Params[Key] = Value
    with pytest.raises(ValueError, match="uses the supplied INCAR and KPOINTS"):
        B.BuildInputs(Root, Params)


def TestElectronicCapUsesSuppliedNelm(Spec, monkeypatch):
    Root, Params = Spec
    B.Prepare(Root, B.BuildInputs(Root, Params))
    def Submit(Command, cwd=None, **kwargs):
        Folder = Path(cwd)
        Outputs(Folder)
        (Folder / "OSZICAR").write_text("DAV: 80 -100 .0 .0\n 1 F= -100 E0= -100\n")
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 123\n", stderr="")
    monkeypatch.setattr(B.subprocess, "run", Submit)
    with pytest.raises(ValueError, match="iteration cap"):
        B.RunPass(Root / B.NAME, 1)
    assert not B.AcceptedHash(Root)


@pytest.mark.parametrize("Key,Value", [("Relaxation", "yes"), ("RelaxForceTol", "nan"),
    ("RelaxPressureTol", "-1"), ("RelaxMaxSteps", "2.5"), ("RelaxWalltime", "bad")])
def TestInvalidSettings(Spec, Key, Value):
    Root, Params = Spec
    Params[Key] = Value
    with pytest.raises(ValueError):
        if B.Enabled(Params):
            B.BuildInputs(Root, Params)


def TestTwoPassesArchivedAcceptedAndNeverRepeated(Spec, monkeypatch):
    Root, Params = Spec
    Calls = Complete(Spec, monkeypatch)
    assert len(Calls) == 2
    assert Calls[0].name == "pass1" and Calls[1].name == "pass2"
    assert B.AcceptedHash(Root)
    B.Prepare(Root, B.BuildInputs(Root, Params))
    for Pass in (1, 2):
        B.RunPass(Root / B.NAME, Pass)
    B.Accept(Root / B.NAME)
    assert len(Calls) == 2
    assert (Root / B.NAME / "Dir_OptUnitCell/pass1/OUTCAR").exists()
    assert "accepted" in FormatRelaxations(Root)
    J = JobSpec("", Root, Params)
    Record = CheckScientificInputs(J)
    assert Record["bulk_relaxation_sha256"] == B.AcceptedHash(Root)
    RecordScientificInputs(J, Record)
    assert CheckScientificInputs(J) == Record


@pytest.mark.parametrize("Change", ["source", "generated", "accepted"])
def TestChangedInputsOrGeometryCannotBeReused(Spec, monkeypatch, Change):
    Root, Params = Spec
    Complete(Spec, monkeypatch)
    if Change == "source":
        (Root / "POSCAR").write_text((Root / "POSCAR").read_text().replace(".125", ".126"))
    elif Change == "generated":
        (Root / B.NAME / "INCAR").write_text("ENCUT = 1\n")
    else:
        (Root / B.NAME / "Accepted.POSCAR").write_text("changed")
    with pytest.raises(ValueError):
        B.CheckPlan(Root, B.BuildInputs(Root, Params))


@pytest.mark.parametrize("Problem", ["force", "pressure", "version", "ionic", "electronic", "empty", "shape", "inventory"])
def TestInvalidFinalResultBlocksHandoff(Spec, monkeypatch, Problem):
    Root, Params = Spec
    B.Prepare(Root, B.BuildInputs(Root, Params))
    def Submit(Command, cwd=None, **kwargs):
        Folder = Path(cwd)
        Options = {}
        if Folder.name == "pass2":
            Options = {"force": {"Force": .1}, "pressure": {"Pressure": 3}, "version": {"Version": "6.3.0"},
                       "ionic": {"Converged": False}, "electronic": {"Cap": True}}.get(Problem, {})
        Outputs(Folder, **Options)
        if Folder.name == "pass2":
            if Problem == "empty":
                (Folder / "CONTCAR").write_text("")
            if Problem in ("shape", "inventory"):
                Text = (Folder / "CONTCAR").read_text()
                Text = Text.replace("12.0000000000000000", "13.0000000000000000") if Problem == "shape" else Text.replace("Zr C", "Zr N")
                (Folder / "CONTCAR").write_text(Text)
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 123\n", stderr="")
    monkeypatch.setattr(B.subprocess, "run", Submit)
    B.RunPass(Root / B.NAME, 1)
    with pytest.raises(ValueError):
        B.RunPass(Root / B.NAME, 2)
    assert not B.AcceptedHash(Root)
    assert not (Root / B.NAME / "Dir_OptUnitCell/optunitcell_is_done").exists()


def TestFirstPassMayReachIonicLimitButNotElectronicLimit(Spec, monkeypatch):
    Root, Params = Spec
    B.Prepare(Root, B.BuildInputs(Root, Params))
    def Submit(Command, cwd=None, **kwargs):
        Outputs(Path(cwd), Converged=False, Force=.1)
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 1\n", stderr="")
    monkeypatch.setattr(B.subprocess, "run", Submit)
    B.RunPass(Root / B.NAME, 1)
    assert B.ReadJson(Root / B.NAME / "Dir_OptUnitCell/pass1/Pass.json")["state"] == "complete"


@pytest.mark.parametrize("Outcome", ["rejected", "ambiguous", "noid", "timeout"])
def TestSubmissionFailureNeverDuplicates(Spec, monkeypatch, Outcome):
    Root, Params = Spec
    B.Prepare(Root, B.BuildInputs(Root, Params))
    Calls = []
    def Submit(*args, **kwargs):
        Calls.append(1)
        if Outcome == "timeout":
            raise subprocess.TimeoutExpired("sbatch", 60)
        return SimpleNamespace(returncode=0 if Outcome == "noid" else 1, stdout="",
                               stderr="Socket timed out" if Outcome == "ambiguous" else "rejected")
    monkeypatch.setattr(B.subprocess, "run", Submit)
    for _ in range(2):
        with pytest.raises(ValueError):
            B.RunPass(Root / B.NAME, 1)
    assert len(Calls) == 1


def TestRestartReattachesToSubmittedJob(Spec, monkeypatch):
    Root, Params = Spec
    B.Prepare(Root, B.BuildInputs(Root, Params))
    def Submit(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 123\n", stderr="")
    monkeypatch.setattr(B.subprocess, "run", Submit)
    monkeypatch.setattr(B, "QueryJob", lambda *a: "unknown")
    with pytest.raises(ValueError, match="Scheduler status unavailable"):
        B.RunPass(Root / B.NAME, 1)
    Folder = Root / B.NAME / "Dir_OptUnitCell/pass1"
    assert B.ReadJson(Folder / "Pass.json")["state"] == "submitted"
    Outputs(Folder)
    monkeypatch.setattr(B.subprocess, "run", lambda *a, **k: pytest.fail("Duplicate submission"))
    B.RunPass(Root / B.NAME, 1)
    assert B.ReadJson(Folder / "Pass.json")["state"] == "complete"


def TestJobGoneWithoutOutputsFails(Spec, monkeypatch):
    Root, Params = Spec
    B.Prepare(Root, B.BuildInputs(Root, Params))
    monkeypatch.setattr(B.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="123\n", stderr=""))
    monkeypatch.setattr(B, "QueryJob", lambda *a: "gone")
    monkeypatch.setattr(B, "GRACE_SECONDS", 0)
    with pytest.raises(ValueError, match="ended without"):
        B.RunPass(Root / B.NAME, 1)


def TestDryRunAndPrepareOnlyNeverSubmit(Spec, monkeypatch):
    Root, Params = Spec
    monkeypatch.setattr(B.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected subprocess"))
    monkeypatch.setattr(B.subprocess, "Popen", lambda *a, **k: pytest.fail("Unexpected subprocess"))
    Before = {P.name: P.read_bytes() for P in Root.iterdir()}
    Args = SimpleNamespace(workdir=str(Root), dry_run=True, prepare_only=False)
    Eligible, Failed = B.Advance([JobSpec("", Root, Params)], Args)
    assert not Eligible and not Failed
    assert Before == {P.name: P.read_bytes() for P in Root.iterdir()}
    Args.dry_run = False
    Args.prepare_only = True
    Eligible, Failed = B.Advance([JobSpec("", Root, Params)], Args)
    assert not Eligible and not Failed
    assert (Root / B.NAME / "INCAR").exists()
    assert not list(Root.glob("*/Dir_VolSearch"))


@pytest.mark.parametrize("Mode", ["bulk", "off"])
@pytest.mark.parametrize("Grouped", [False, True])
def TestStandalonePreparationStopsAtCorrectStage(tmp_path, Mode, Grouped):
    Workspace = tmp_path / "workspace"
    SpecRoot = Workspace / "material" if Grouped else Workspace
    WriteSpec(SpecRoot)
    Config = SpecRoot / "OxParams"
    Config.write_text(Config.read_text().replace("Relaxation=bulk", "Relaxation=" + Mode))
    if Grouped:
        (Workspace / "OxParams").write_text("JobSpecs=[material]\n")
    Result = subprocess.run([sys.executable, str(SGUSCHI.WORKSPACE_SETUP_MODULE)],
                            cwd=Workspace, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert Result.returncode == 0, Result.stdout + Result.stderr
    if Mode == "bulk":
        assert (SpecRoot / B.NAME / "Dir_OptUnitCell").is_dir()
        assert B.ReadJson(SpecRoot / B.NAME / "Status.json")["state"] == "prepared"
        assert not list(SpecRoot.glob("*/Dir_VolSearch"))
        assert not list(SpecRoot.rglob("Pass.json"))
    else:
        assert not (SpecRoot / B.NAME).exists()
        assert len(list(SpecRoot.glob("*/Dir_VolSearch/INCAR"))) == 4
        assert not list(SpecRoot.rglob(".vasp_submitted_step"))


def TestPreflightAllSpecsBeforeAnyMutation(tmp_path):
    A, C = tmp_path / "a", tmp_path / "c"
    Pa, Pc = WriteSpec(A), WriteSpec(C)
    Pc["RelaxMaxSteps"] = "0"
    Args = SimpleNamespace(workdir=str(tmp_path), dry_run=False, prepare_only=True)
    with pytest.raises(ValueError):
        B.Advance([JobSpec("a", A, Pa), JobSpec("c", C, Pc)], Args)
    assert not (A / B.NAME).exists()


def TestExistingTrajectoryNotRetrofitted(Spec):
    Root, Params = Spec
    (Root / "973_1/Dir_VolSearch").mkdir(parents=True)
    with pytest.raises(ValueError, match="retrofit"):
        B.CheckPlan(Root, B.BuildInputs(Root, Params))


def TestCleavagePreservedWhenLargestGapMovesAndPlaneDrifts():
    X = np.array([.05, .30, .55, .80])
    Original = np.column_stack([X, np.zeros(4), np.zeros(4)])
    Relaxed = Original.copy()
    Relaxed[:, 0] = np.array([.05, .28, .60, .80]) + .4
    Position = pd.DataFrame(Relaxed % 1, columns=["x", "y", "z"])
    Position.insert(0, "Element", "Zr")
    Cell = pd.DataFrame(np.diag([12., 8., 8.]), columns=["x", "y", "z"])
    Old, _ = AddVacuum(Position, Cell, 2)
    New, Expanded = AddVacuum(Position, Cell, 2, CleavageReference=Original)
    assert Old.x.idxmin() != 0
    assert New.x.idxmin() == 0 and New.x.idxmax() == 3
    assert Expanded.iloc[0, 0] == 36
    # No deformation: all pairwise unfolded x separations are preserved.
    np.testing.assert_allclose(np.diff(New.x) * 36, np.diff(Relaxed[:, 0]) * 12)


def TestCleavageRejectsAmbiguousReconstruction():
    Original = np.zeros((4, 3))
    Original[:, 0] = [.05, .30, .55, .80]
    Relaxed = Original.copy()
    Relaxed[:, 0] += [0, .25, .5, .75]
    with pytest.raises(ValueError):
        PreservedSlabCoordinates(Original, Relaxed)


def TestAcceptedInputFeedsAllReplicas(Spec, monkeypatch):
    Root, Params = Spec
    Complete(Spec, monkeypatch)
    monkeypatch.setattr(SGUSCHI, "ParseArgs", lambda: SimpleNamespace(
        workdir=str(Root), prepare_only=True, dry_run=False, relax_only=False))
    monkeypatch.setattr(B.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected submit"))
    assert SGUSCHI.main() == 0
    assert len(list(Root.glob("*/Dir_VolSearch"))) == 4
    for PathFile in Root.glob("*/BulkRelaxation.sha256"):
        assert PathFile.read_text().strip() == B.AcceptedHash(Root)


def TestRelaxOnlyDoesNotGenerateOxidation(Spec, monkeypatch):
    Root, Params = Spec
    Complete(Spec, monkeypatch)
    monkeypatch.setattr(SGUSCHI, "ParseArgs", lambda: SimpleNamespace(
        workdir=str(Root), prepare_only=False, dry_run=False, relax_only=True))
    assert SGUSCHI.main() == 0
    assert not list(Root.glob("*/Dir_VolSearch"))


def TestRealCshTwoPassRoutineWithFakeScheduler(Spec, monkeypatch, tmp_path):
    Csh = shutil.which("csh") or shutil.which("tcsh")
    if not Csh or os.name == "nt":
        pytest.skip("C-shell execution needs Linux; all scheduler operations remain mocked")
    Root, Params = Spec
    B.Prepare(Root, B.BuildInputs(Root, Params))
    Bin = tmp_path / "bin"
    Bin.mkdir()
    Fake = Bin / "sbatch"
    Fake.write_text("#!" + sys.executable + "\n" +
        "from pathlib import Path\n"
        "from TestBulkRelaxation import Outputs\n"
        "Outputs(Path.cwd())\n"
        "print('Submitted batch job 123')\n")
    Fake.chmod(0o755)
    Env = os.environ.copy()
    Env.update(sguschipath=str(B.SCRIPT.parent), SGUSCHI_PYTHON=sys.executable,
               PATH=str(Bin) + os.pathsep + Env["PATH"],
               PYTHONPATH=str(Path(__file__).parent) + os.pathsep + str(Path(__file__).parents[1])
               + os.pathsep + Env.get("PYTHONPATH", ""))
    Result = subprocess.run([Csh, "-f", str(B.SCRIPT), "--sguschi"],
                            cwd=Root / B.NAME / "Dir_OptUnitCell", env=Env, capture_output=True, text=True, timeout=30)
    assert Result.returncode == 0, Result.stdout + Result.stderr
    assert B.AcceptedHash(Root)
    assert (Root / B.NAME / "Dir_OptUnitCell/pass1/OUTCAR").exists()
    assert (Root / B.NAME / "Dir_OptUnitCell/pass2/OUTCAR").exists()


def TestAtomicWriteRetriesButNeverDirectlyOverwrites(tmp_path, monkeypatch):
    Target = tmp_path / "status.json"
    Target.write_text("old")
    Original = Path.replace
    Calls = []
    def Replace(Self, Other):
        Calls.append(1)
        if len(Calls) == 1:
            assert Target.read_text() == "old"
            raise PermissionError("temporary reader lock")
        return Original(Self, Other)
    monkeypatch.setattr(Path, "replace", Replace)
    monkeypatch.setattr(B.time, "sleep", lambda *a: None)
    B.AtomicText(Target, "new")
    assert len(Calls) == 2 and Target.read_text() == "new"


@pytest.mark.parametrize("Scheduler,Output,Expected", [
    ("sbatch", "123|TIMEOUT\n", "gone"), ("sbatch", "123|RUNNING\n", "active"),
    ("sbatch", "", "unknown"), ("qsub", "job_state = R\n", "active"),
    ("qsub", "job_state = F\n", "gone")])
def TestSchedulerAccountingFallback(monkeypatch, Scheduler, Output, Expected):
    def Run(Command, **kwargs):
        if Command[0] == "squeue":
            return SimpleNamespace(returncode=1, stdout="", stderr="Invalid job id")
        return SimpleNamespace(returncode=0, stdout=Output, stderr="")
    monkeypatch.setattr(B.subprocess, "run", Run)
    assert B.QueryJob([Scheduler], "123") == Expected


def TestAdvanceLaunchesAllSpecsAndIsolatesFailure(tmp_path, monkeypatch):
    Good, Bad = tmp_path / "good", tmp_path / "bad"
    Pg, Pb = WriteSpec(Good), WriteSpec(Bad)
    from utils import SimulationSummary
    def UnavailableSummary(*args, **kwargs):
        raise PermissionError("summary temporarily locked")
    monkeypatch.setattr(SimulationSummary, "WriteSummaryOnce", UnavailableSummary)
    (tmp_path / "OxParams").write_text("JobSpecs=[good,bad]\n")
    Starts = []
    monkeypatch.setattr(B.shutil, "which", lambda *a: "/usr/bin/csh")
    def Submit(Command, cwd=None, **kwargs):
        Outputs(Path(cwd))
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 1\n", stderr="")
    monkeypatch.setattr(B.subprocess, "run", Submit)
    def Start(Command, cwd=None, **kwargs):
        Starts.append(cwd)
        Root = Path(cwd).parent
        if Root.parent.name == "good":
            B.RunPass(Root, 1)
            B.RunPass(Root, 2)
            B.Accept(Root)
            Code = 0
        else:
            B.SetStatus(Root, "failed", 1, "synthetic failure")
            Code = 1
        def Poll():
            assert len(Starts) == 2  # no waiting for the first before launching the second
            return Code
        return SimpleNamespace(poll=Poll)
    monkeypatch.setattr(B.subprocess, "Popen", Start)
    Args = SimpleNamespace(workdir=str(tmp_path), dry_run=False, prepare_only=False)
    Eligible, Failed = B.Advance([JobSpec("good", Good, Pg), JobSpec("bad", Bad, Pb)], Args)
    assert Failed and [S.Id for S in Eligible] == ["good"]
    assert B.ReadJson(Good / B.NAME / "Status.json")["job_id"] == "1"
    assert not list(Bad.glob("*/Dir_VolSearch"))


@pytest.mark.parametrize("Mode", ["bulk", "off", None])
def TestRelaxationChoiceLaunchesMd(Spec, monkeypatch, Mode):
    Root, Params = Spec
    if Mode == "bulk":
        Complete(Spec, monkeypatch)
    else:
        Config = Root / "OxParams"
        Config.write_text(Config.read_text().replace("Relaxation=bulk\n",
                          "Relaxation=off\n" if Mode == "off" else ""))
        monkeypatch.setattr(B, "BuildInputs", lambda *a: pytest.fail("Disabled relaxation was prepared"))
        monkeypatch.setattr(B.subprocess, "Popen", lambda *a, **k: pytest.fail("Disabled optunitcell was launched"))
    Submitted = []
    def Submit(Command, cwd=None, **kwargs):
        Submitted.append(Path(cwd))
        Values = B.IncarValues((Path(cwd) / "INCAR").read_text())
        assert Values["IBRION"] == "0" and Values["ISIF"] == "2"
        assert Values["ENCUT"] == "400"
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 123\n", stderr="")
    monkeypatch.setattr(B.subprocess, "run", Submit)
    monkeypatch.setattr(SGUSCHI, "ParseArgs", lambda: SimpleNamespace(
        workdir=str(Root), prepare_only=False, dry_run=False, relax_only=False))
    Seen = []
    monkeypatch.setattr(SGUSCHI, "RunOrchestration", lambda Root, Params, Dirs: Seen.extend(Dirs) or 0)
    assert SGUSCHI.main() == 0
    assert len(Submitted) == len(Seen) == 4
    if Mode != "bulk":
        assert not (Root / B.NAME).exists()
        assert not list(Root.glob("*/BulkRelaxation.sha256"))


def TestSubmittedInputModifiedDuringCalculationFails(Spec, monkeypatch):
    Root, Params = Spec
    B.Prepare(Root, B.BuildInputs(Root, Params))
    def Submit(Command, cwd=None, **kwargs):
        Folder = Path(cwd)
        (Folder / "INCAR").write_text("ENCUT = 1\n")
        Outputs(Folder)
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 123\n", stderr="")
    monkeypatch.setattr(B.subprocess, "run", Submit)
    with pytest.raises(ValueError, match="Submitted input changed"):
        B.RunPass(Root / B.NAME, 1)
