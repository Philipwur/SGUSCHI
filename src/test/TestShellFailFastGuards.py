"""Static checks for fail-fast guards in SLUSCHI shell controllers."""

from __future__ import annotations

from pathlib import Path


RootDir = Path(__file__).resolve().parents[2]
SluschiDir = RootDir / "src" / "dependencies" / "SLUSCHI_mod"


def ReadScript(Name: str) -> str:
    """Read a SLUSCHI shell helper."""
    return (SluschiDir / Name).read_text(encoding="utf-8")


def TestUpdateIncarValidatesBeforeDeletingTag() -> None:
    """Blank updates should fail before the old INCAR tag is removed."""
    Text = ReadScript("UpdateINCAR")

    assert "FATAL UpdateINCAR" in Text
    assert '"$tag" == "" || "$value" == ""' in Text
    assert Text.index('if ( "$tag" == ""') < Text.index("sed -i")


def TestAdjustBmixRejectsMissingOrInvalidInputs() -> None:
    """BMIX adjustment should fail on missing BMIX/GAMMA instead of writing blanks."""
    Text = ReadScript("AdjustBMIX")

    assert "FATAL AdjustBMIX: could not extract BMIX" in Text
    assert "FATAL AdjustBMIX: could not extract a positive GAMMA" in Text
    assert "g+0 <= 0" in Text
    assert "UpdateINCAR BMIX $bmix" in Text


def TestVolsearchContStopsBeforeSubmittingAfterCriticalFailures() -> None:
    """The controller should exit before job submission on critical helper failures."""
    Text = ReadScript("volsearch_cont")

    assert "DetermineSize.x failed" in Text
    assert "OxidationStep.py failed; no new job submitted" in Text
    assert Text.index("OxidationStep.py failed; no new job submitted") < Text.index("$vaspcmd jobsub")


def TestVolsearchContRecoversFromPressureUpdateFailures() -> None:
    """Malformed pressure helper files should skip lattice updates, not stop jobs."""
    Text = ReadScript("volsearch_cont")

    assert "grep -v ' 0.00 '" not in Text
    assert "has no valid numeric stress rows" in Text
    assert "six-column numeric stress rows" in Text
    assert "pressure_kinetic.out is missing or non-numeric" in Text
    assert "pressure_Pulay.out is missing or non-numeric" in Text
    assert "pressure_target.out is missing or non-numeric" in Text
    assert "volume.out is missing or its last value is invalid" in Text
    assert "PressureAvg.x failed or did not create pressure3_total.out; skipping pressure-controlled lattice update" in Text
    assert "DetermineSize.x skipped" in Text
    assert "VolSearchStop.x skipped" in Text
    assert "current lattice will be reused" in Text
    assert Text.index("PressureAvg.x failed or did not create pressure3_total.out") < Text.index("$vaspcmd jobsub")


def TestVolsearchContWarnsOnIncarAdjustmentFailures() -> None:
    """INCAR adjustment helpers should not stop an otherwise usable job."""
    Text = ReadScript("volsearch_cont")

    assert "WARNING volsearch_cont: AdjustPOTIM failed; keeping current POTIM." in Text
    assert "WARNING volsearch_cont: AdjustNBANDS failed; keeping current NBANDS/default." in Text
    assert "WARNING volsearch_cont: AdjustBMIX failed; keeping current BMIX." in Text
    assert "FATAL volsearch_cont: AdjustPOTIM failed" not in Text
    assert "FATAL volsearch_cont: AdjustNBANDS failed" not in Text
    assert "FATAL volsearch_cont: AdjustBMIX failed" not in Text


def TestPressureAvgFortranUsesCheckedReads() -> None:
    """PressureAvg.x should fail cleanly on bad input files."""
    Text = ReadScript("PressureAvg.f90")

    assert "iostat=ios" in Text
    assert "FATAL PressureAvg.x" in Text
    assert "pressure3.out contains no valid rows" in Text
    assert "last volume.out value must be positive" in Text
    assert "invalid or missing value in" in Text
