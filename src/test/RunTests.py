#%%
"""Convenience runner for the SGUSCHI test suite."""

from __future__ import annotations

import sys
from pathlib import Path


def EnsureSrcOnPath() -> None:
    """Insert the project src directory into sys.path if missing."""
    RootDir = Path(__file__).resolve().parents[2]
    SrcDir = RootDir / "src"
    if str(SrcDir) not in sys.path:
        sys.path.insert(0, str(SrcDir))


def RunPytest(Args: list[str] | None = None) -> int:
    """Execute pytest with defaults for PascalCase test names."""
    RootDir = Path(__file__).resolve().parents[2]
    TestDir = RootDir / "src" / "test"

    EnsureSrcOnPath()

    try:
        import pytest
    except Exception as Exc:
        print("pytest import failed:", Exc)
        return 1

    if Args is None:
        Args = [
            "-q",
            str(TestDir),
            "-o",
            "python_files=Test*.py",
            "-o",
            "python_functions=Test*",
        ]

    ExitCode = pytest.main(Args)

    if ExitCode == 0:
        print("\nAll tests passed.\n")
        print("Summary of coverage:")
        print("- Tested:")
        print("  - VaspIO: ReadKeyValueFile, Read/Write POSCAR, Read/Write XYZ (metadata + coords), ReadRateAnalysis")
        print("  - OxidationStep: TestCase mode and file-writing outputs (XYZ, RateAnalysis)")
        print("  - Gas identification: FindGases + CreateGassesRemovedStr on real frames")
        print("  - Metadata parsing: Time field from ZrCN XYZ")
        print("- Not tested:")
        print("  - OutcarParser parsing of real OUTCAR files")
        print("  - Full OxidationStep physics with real OUTCAR + POSCAR pipeline")
        print("  - FixXYZ / FixRateAnalysis utilities")
        print("  - Performance/stress cases and long trajectories")
        print("\nProxy statement:")
        print("These tests exercise core I/O and gas-identification steps used by OxidationStep,")
        print("but they do not execute a full real-data OxidationStep run end-to-end.")

    return ExitCode


if __name__ == "__main__":
    RunPytest()

# %%
