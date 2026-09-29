"""First-job initialization, input preservation and shell-entry-point behavior."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

from utils import InitialVasp, StatusLog


def ReadAssignments(Incar):
    Values = {}
    for Line in Incar.read_text().splitlines():
        for Assignment in re.split(r"[#!]", Line, maxsplit=1)[0].split(";"):
            if "=" in Assignment:
                Tag, Value = Assignment.split("=", 1)
                Tag = Tag.strip().upper()
                assert Tag not in Values, "Duplicate active tag: " + Tag
                Values[Tag] = Value.strip()
    return Values


def TestInitializeHandlesCommentsDuplicatesAndSemicolons(tmp_path):
    Incar = tmp_path / "INCAR"
    Incar.write_text(
        "# TEBEG = 300; POTIM = 5 is only a comment\n"
        "! TEEND = 300 is also a comment\n"
        "sigma = 0.2; POTIM = 0.8; nbands = 64 ! keep POTIM\n"
        "SIGMA = 0.1\nTEBEG = 300; TEEND = 400\n"
        "NSW = 10; SMASS = 2; EDIFF = 1e-4\n"
        "ENCUT = 500 # NBANDS is chosen automatically\nNCORE = 8"
    )
    InitialVasp.InitializeIncar(Incar, 873)
    assert ReadAssignments(Incar) == {
        "SIGMA": "0.075078", "TEBEG": "873", "TEEND": "873",
        "NSW": "80", "SMASS": "0", "POTIM": "0.8", "EDIFF": "1e-4",
        "ENCUT": "500", "NCORE": "8",
    }
    assert "# TEBEG = 300; POTIM = 5 is only a comment" in Incar.read_text()
    assert "! TEEND = 300 is also a comment" in Incar.read_text()
    Before = Incar.read_bytes(), Incar.stat().st_mtime_ns
    InitialVasp.InitializeIncar(Incar, 873)
    assert (Incar.read_bytes(), Incar.stat().st_mtime_ns) == Before


@pytest.mark.parametrize("Temp", ["", "invalid", "nan", "inf", 0, -1])
def TestInvalidTemperatureLeavesIncarUntouched(tmp_path, Temp):
    Incar = tmp_path / "INCAR"
    Incar.write_text("POTIM = 1\nNBANDS = 128\n")
    Original = Incar.read_bytes()
    with pytest.raises(ValueError):
        InitialVasp.InitializeIncar(Incar, Temp)
    assert Incar.read_bytes() == Original
    assert list(tmp_path.iterdir()) == [Incar]


def TestFailedAtomicReplaceLeavesOriginalIntact(tmp_path, monkeypatch):
    Incar = tmp_path / "INCAR"
    Incar.write_text("POTIM = 1\nNBANDS = 128\n")
    Original = Incar.read_bytes()
    def FailReplace(*Args):
        raise OSError("Simulated replacement failure")
    monkeypatch.setattr(Path, "replace", FailReplace)
    with pytest.raises(OSError, match="replacement failure"):
        InitialVasp.InitializeIncar(Incar, 873)
    assert Incar.read_bytes() == Original
    assert list(tmp_path.iterdir()) == [Incar]


@pytest.mark.parametrize("Evidence", [
    "OUTCAR", "OSZICAR", "CONTCAR", ".vasp_submitted_step",
    "volsearch_is_done", "step_folder", "submission_log",
])
def TestPreparationNeverRewritesStartedTrajectories(tmp_path, Evidence):
    Incar = tmp_path / "INCAR"
    Incar.write_text("NBANDS = 128\nPOTIM = 1.6\nSIGMA = 0.09\nBMIX = 0.4\n")
    if Evidence == "step_folder":
        (tmp_path / "1").mkdir()
    elif Evidence == "submission_log":
        StatusLog.Append(tmp_path, "test", "submitted", "1 123")
    else:
        (tmp_path / Evidence).touch()  # Even empty output means VASP may have started.
    Before = Incar.read_bytes(), Incar.stat().st_mtime_ns
    # No job.in: the guard must run before reading or validating new settings.
    InitialVasp.PrepareInitialVasp(tmp_path)
    assert (Incar.read_bytes(), Incar.stat().st_mtime_ns) == Before
    assert not InitialVasp.NeedsInitialVaspJob(tmp_path)


def TestStandaloneInitializerUsesJobTemperatureAndReportsFailure(tmp_path):
    Incar = tmp_path / "INCAR"
    Incar.write_text("POTIM = 1\n")
    JobIn = tmp_path / "job.in"
    JobIn.write_text("temp = 1073\n")
    Command = [sys.executable, str(Path(InitialVasp.__file__).resolve())]
    Result = subprocess.run(Command, cwd=tmp_path, capture_output=True, text=True)
    assert Result.returncode == 0, Result.stderr
    assert ReadAssignments(Incar)["SIGMA"] == "0.092278"
    Original = Incar.read_bytes()
    JobIn.write_text("# temp = 1073\n")
    Result = subprocess.run(Command, cwd=tmp_path, capture_output=True, text=True)
    assert Result.returncode != 0
    assert "FATAL initial VASP preparation" in Result.stderr
    assert Incar.read_bytes() == Original
