"""Runtime checks for the PressureAvg Fortran helper."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

import pytest


RootDir = Path(__file__).resolve().parents[2]
SourcePath = RootDir / "src" / "dependencies" / "SLUSCHI_mod" / "PressureAvg.f90"


def FindFortranCompiler() -> Optional[str]:
    """Return an available Fortran compiler executable."""
    for Name in ("ifort", "ifx", "gfortran"):
        PathToCompiler = shutil.which(Name)
        if PathToCompiler:
            return PathToCompiler
    return None


@pytest.fixture(scope="session")
def PressureAvgExe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile PressureAvg.f90 once when a compiler is available."""
    Compiler = FindFortranCompiler()
    if Compiler is None:
        pytest.skip("No Fortran compiler found for PressureAvg runtime tests.")

    BuildDir = tmp_path_factory.mktemp("pressure_avg_build")
    ExePath = BuildDir / ("PressureAvg.exe" if os.name == "nt" else "PressureAvg.x")
    Proc = subprocess.run(
        [Compiler, "-o", str(ExePath), str(SourcePath)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert Proc.returncode == 0, Proc.stdout + Proc.stderr
    return ExePath


def WritePressureInputs(
    WorkDir: Path,
    Pressure: str = "1 2 3 4 5 6\n7 8 9 10 11 12\n",
    Volume: Optional[str] = "100\n",
    Kinetic: str = "1\n",
    Pulay: str = "2\n",
    Target: str = "0.5\n",
) -> None:
    """Write the helper files consumed by PressureAvg.x."""
    (WorkDir / "pressure3.out").write_text(Pressure, encoding="utf-8")
    if Volume is not None:
        (WorkDir / "volume.out").write_text(Volume, encoding="utf-8")
    (WorkDir / "pressure_kinetic.out").write_text(Kinetic, encoding="utf-8")
    (WorkDir / "pressure_Pulay.out").write_text(Pulay, encoding="utf-8")
    (WorkDir / "pressure_target.out").write_text(Target, encoding="utf-8")


def RunPressureAvg(ExePath: Path, WorkDir: Path) -> subprocess.CompletedProcess[str]:
    """Run PressureAvg.x in a prepared work directory."""
    return subprocess.run(
        [str(ExePath)],
        cwd=str(WorkDir),
        capture_output=True,
        text=True,
        check=False,
    )


def TestPressureAvgComputesExpectedAverage(
    PressureAvgExe: Path,
    tmp_path: Path,
) -> None:
    """Valid helper files should produce six total pressure components."""
    WritePressureInputs(tmp_path)

    Proc = RunPressureAvg(PressureAvgExe, tmp_path)

    assert Proc.returncode == 0, Proc.stdout + Proc.stderr
    Values = [float(Token) for Token in (tmp_path / "pressure3_total.out").read_text().split()]
    ExpectedBase = [Value * 1602.177 / 100.0 for Value in (4, 5, 6, 7, 8, 9)]
    Expected = [ExpectedBase[0] + 2.5, ExpectedBase[1] + 2.5, ExpectedBase[2] + 2.5]
    Expected.extend(ExpectedBase[3:])

    assert Values == pytest.approx(Expected, abs=1e-6)


@pytest.mark.parametrize(
    ("Overrides", "ExpectedMessage"),
    [
        ({"Pressure": ""}, "pressure3.out contains no valid rows"),
        ({"Pressure": "1 2 bad 4 5 6\n"}, "invalid numeric row in pressure3.out"),
        ({"Volume": "0\n"}, "last volume.out value must be positive"),
        ({"Volume": None}, "could not open volume.out"),
        ({"Pulay": "abc\n"}, "invalid or missing value in pressure_Pulay.out"),
    ],
)
def TestPressureAvgRejectsBadInputs(
    PressureAvgExe: Path,
    tmp_path: Path,
    Overrides: Dict[str, Optional[str]],
    ExpectedMessage: str,
) -> None:
    """Bad helper files should stop cleanly with a useful fatal message."""
    WritePressureInputs(tmp_path, **Overrides)

    Proc = RunPressureAvg(PressureAvgExe, tmp_path)

    assert Proc.returncode != 0
    assert ExpectedMessage in (Proc.stdout + Proc.stderr)
