"""Static checks for fail-fast guards in SLUSCHI shell controllers."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


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


def TestVolsearchStartupUsesGuardedPreparation() -> None:
    """Startup must not reset an INCAR already handed to VASP, including NBANDS."""
    Startup = ReadScript("volsearch_cont").split("# submit the first job", 1)[0]
    assert 'python "$sluschipath/../../utils/InitialVasp.py"' in Startup
    assert "FATAL volsearch_cont: initial VASP preparation failed." in Startup
    assert "UpdateINCAR" not in Startup
    assert "sed -i" not in Startup


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
    # rindex targets the main-loop submission; the first "$vaspcmd jobsub" is now
    # the startup-guard resubmission, which sits earlier in the file by design.
    assert Text.index("OxidationStep.py failed; no new job submitted") < Text.rindex("$vaspcmd jobsub")


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
    assert Text.index("PressureAvg.x failed or did not create pressure3_total.out") < Text.rindex("$vaspcmd jobsub")


def TestVolsearchContWarnsOnIncarAdjustmentFailures() -> None:
    """INCAR adjustment helpers should not stop an otherwise usable job."""
    Text = ReadScript("volsearch_cont")

    assert "WARNING volsearch_cont: AdjustPOTIM failed; keeping current POTIM." in Text
    assert "WARNING volsearch_cont: AdjustNBANDS failed; keeping current NBANDS/default." in Text
    assert "WARNING volsearch_cont: AdjustBMIX failed; keeping current BMIX." in Text
    assert "FATAL volsearch_cont: AdjustPOTIM failed" not in Text
    assert "FATAL volsearch_cont: AdjustNBANDS failed" not in Text
    assert "FATAL volsearch_cont: AdjustBMIX failed" not in Text


def TestVolsearchContArchivesCompletedInputsBeforeAdjustment(tmp_path):
    """Run the archive/adjustment block with helpers that change the next INCAR."""
    Csh = shutil.which("csh") or shutil.which("tcsh")
    if Csh is None:
        pytest.skip("C-shell interpreter is not available")

    Originals = {
        "INCAR": "POTIM = 1.0\nNBANDS = 96\nBMIX = 0.63\n",
        "KPOINTS": "original k-points\n",
        "POSCAR": "original structure\n",
        "OSZICAR": "completed segment energies\n",
    }
    for Name, Content in Originals.items():
        (tmp_path / Name).write_text(Content, encoding="utf-8")
    (tmp_path / "OUTCAR").write_text("completed segment output\n", encoding="utf-8")
    Helpers = tmp_path / "helpers"
    Helpers.mkdir()
    for Tag, Value in (("POTIM", "1.2"), ("NBANDS", "112"), ("BMIX", "0.4")):
        Helper = Helpers / ("Adjust" + Tag)
        Helper.write_text(
            f'#!{Csh} -f\n'
            'if ( ! -s OUTCAR || ! -s OSZICAR ) exit 1\n'
            f"sed -i '/^{Tag} /d' INCAR\n"
            f'echo "{Tag} = {Value}" >> INCAR\n',
            encoding="utf-8",
        )
        Helper.chmod(0o755)

    Text = ReadScript("volsearch_cont")
    Start = min(Text.index("    mkdir $nstep"), Text.index("    echo --- ADJUST INCAR TAGS ---"))
    Block = Text[Start:Text.index("    set nkpts_value", Start)]
    Script = tmp_path / "archive.csh"
    Script.write_text('set nstep = 12\nset failure_marker = sguschi_failed\n'
                      'set sluschipath = "$cwd/helpers"\n' + Block, encoding="utf-8")
    Result = subprocess.run([Csh, "-f", str(Script)], cwd=tmp_path, capture_output=True, text=True)
    assert Result.returncode == 0, Result.stdout + Result.stderr
    assert not (tmp_path / "sguschi_failed").exists()
    for Name, Content in Originals.items():
        assert (tmp_path / "12" / Name).read_text(encoding="utf-8") == Content
    assert (tmp_path / "INCAR").read_text(encoding="utf-8") == "POTIM = 1.2\nNBANDS = 112\nBMIX = 0.4\n"


def TestAdjustNbandsFallsBackWithoutCshExpressionErrors() -> None:
    """NBANDS adjustment should warn and keep current defaults on bad OUTCAR data."""
    Text = ReadScript("AdjustNBANDS")

    assert "@ nbands = `grep" not in Text
    assert "if ( ! ($a == 0.00000) )" not in Text
    assert "could not extract NBANDS from OUTCAR; keeping current NBANDS/default." in Text
    assert "could not determine occupied band from OUTCAR; keeping current NBANDS/default." in Text
    assert "missing occupancy data for band $nbands; keeping current NBANDS/default." in Text
    assert "non-numeric occupancy value $a; keeping current NBANDS/default." in Text
    assert "invalid add_nbands value $add_nbands_value; using default add_nbands 10." in Text
    assert "UpdateINCAR failed for NBANDS $nbands; keeping current NBANDS/default." in Text


def TestPressureAvgFortranUsesCheckedReads() -> None:
    """PressureAvg.x should fail cleanly on bad input files."""
    Text = ReadScript("PressureAvg.f90")

    assert "iostat=ios" in Text
    assert "FATAL PressureAvg.x" in Text
    assert "pressure3.out contains no valid rows" in Text
    assert "last volume.out value must be positive" in Text
    assert "invalid or missing value in" in Text
