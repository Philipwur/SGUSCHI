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

    assert "PressureAvg.x failed" in Text
    assert "DetermineSize.x failed" in Text
    assert "AdjustBMIX failed" in Text
    assert "OxidationStep.py failed; no new job submitted" in Text
    assert Text.index("OxidationStep.py failed; no new job submitted") < Text.index("$vaspcmd jobsub")
