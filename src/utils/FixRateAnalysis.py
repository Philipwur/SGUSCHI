import os
import sys
from pathlib import Path
from typing import Union, Dict

import pandas as pd

try:
    from tqdm import tqdm as Tqdm
except ImportError:
    Tqdm = None

sys.path.append(str(Path(__file__).resolve().parents[1]))

from workflow import VaspIO as vio
from workflow import OxidationAnalysis as an
from workflow.OxidationStep import ExponentialSmoothing, CreateGassesRemovedStr


'''
-------------------------------------------------------------------------------
How to use this script:

From any given directory you can run:
> python path/to/SGUSCHI/src/utils/FixRateAnalysis.py /path/to/Dir_VolSearch

Or if you’re already in Dir_VolSearch:
> python src/utils/FixRateAnalysis.py

There's a lot of IO required for this script, so expect about 1 minute per 50
VolSearch folders.

tqdm functionality is optional, other imports arent.
This script drops the final row of rateanalysis so that oxidation step can work
(it needs to append latest simulation to rateanalysis).
-------------------------------------------------------------------------------
'''


def FixRateAnalysis(WorkDir: Union[str, Path] = None) -> pd.DataFrame:
    """
    Rebuild RateAnalysis for a Dir_VolSearch-like working directory.

    - Reads OxParams and CovalentRadii from the project RootDir.
    - Loops over all numbered subfolders in WorkDir (1, 2, 3, ...).
    - For each folder, reads OUTCAR data via vio.OutcarParser.
    - Applies frame-wise exponential smoothing.
    - Recomputes O2 totals, cumulative O2 added, gas removal, gas fraction.
    - Writes the repaired RateAnalysis.csv:
        - in WorkDir
        - in RootDir / 'xyz_files' / f'RateAnalysis_{TrajectoryName}.csv'
    """
    if WorkDir is None:
        WorkDir = os.getcwd()

    WorkDir = Path(WorkDir).resolve()
    RootDir = WorkDir.parents[1]
    TrajectoryName = WorkDir.parent.name

    # ------------------------ Read hyperparameters ------------------------

    OxParamsPath = RootDir / "OxParams"
    if not OxParamsPath.exists():
        raise FileNotFoundError("OxParams file not found in %r." % OxParamsPath)

    OxParams = vio.ReadKeyValueFile(
        OxParamsPath,
        RequiredKeys=[
            "AtomicRadiusTol",
            "O2Tol",
            "OSmoothing",
            "GasRatio",
            "InitO2Count",
        ],
    )

    AtomicRadiusTol = float(OxParams["AtomicRadiusTol"])
    O2TolBase = float(OxParams["O2Tol"])
    OxygenSmoothing = float(OxParams["OSmoothing"])
    GasRatio = float(OxParams["GasRatio"])
    InitO2Count = int(OxParams["InitO2Count"])

    CovalentRadiiPath = RootDir / "CovalentRadii"
    if not CovalentRadiiPath.exists():
        raise FileNotFoundError("CovalentRadii file not found in %r." % CovalentRadiiPath)

    CovalentRadiiRaw = vio.ReadKeyValueFile(CovalentRadiiPath)
    CovalentRadii: Dict[str, float] = {
        Key: float(Value) for Key, Value in CovalentRadiiRaw.items()
    }

    # CellDim is assumed constant across the simulation, take from current POSCAR
    PositionInitial, CellDim, _ = vio.ReadPoscar(WorkDir, GiveVelocities=True)

    # -------------------- Initialize RateAnalysis DataFrame --------------------

    Columns = [
        "Time (fs)",
        "O2 Count",
        "Smoothed O2 Count",
        "O2 Added",
        "Gas Removed",
        "Free Gas Fraction",
    ]

    RateAnalysis = pd.DataFrame(
        [
            {
                "Time (fs)": 0.0,
                "O2 Count": InitO2Count,
                "Smoothed O2 Count": InitO2Count,
                "O2 Added": InitO2Count,
                "Gas Removed": "[]",
                "Free Gas Fraction": 1.0,
            }
        ],
        columns=Columns,
    )

    CurrentTimeFs = 0.0
    CurrentSmoothedO2Count = InitO2Count
    CurrentO2AddedCumulative = InitO2Count

    # --------------------- Get all numbered step folders ---------------------

    StepFolders = sorted(
        int(Directory.name)
        for Directory in WorkDir.iterdir()
        if Directory.is_dir() and Directory.name.isdigit()
    )

    if not StepFolders:
        RateAnalysisPathWorkDir = WorkDir / "RateAnalysis.csv"
        RateAnalysisPathRoot = RootDir / "xyz_files" / ("RateAnalysis_%s.csv" % TrajectoryName)

        RateAnalysisPathWorkDir.parent.mkdir(parents=True, exist_ok=True)
        RateAnalysis.to_csv(RateAnalysisPathWorkDir, index=False)

        RateAnalysisPathRoot.parent.mkdir(parents=True, exist_ok=True)
        RateAnalysis.to_csv(RateAnalysisPathRoot, index=False)

        return RateAnalysis

    # Decide whether to wrap with tqdm
    StepIterable = StepFolders
    if Tqdm is not None:
        StepIterable = Tqdm(StepFolders, desc="FixRateAnalysis", unit="step")

    # -------------------------- Rebuild step by step --------------------------

    for StepIndex in StepIterable:
        OutcarPath = WorkDir / str(StepIndex)
        OutcarData = vio.OutcarParser(OutcarPath)

        TimesFs = OutcarData["TimesFs"]
        if len(TimesFs) == 0:
            continue

        StepDurationFs = float(TimesFs[-1])
        CurrentTimeFs += StepDurationFs

        # ----------------- Frame-wise smoothing over this OUTCAR -----------------

        O2CountLastFrame = 0

        for OutcarPosition in OutcarData["Positions"]:
            FrameGasses = an.FindGases(
                OutcarPosition,
                CellDim,
                CovalentRadii=CovalentRadii,
                AtomicRadiusTol=AtomicRadiusTol,
                MinimumComplexity=2,
                MaximumComplexity=3,
                ReturnBondMatrix=False,
            )

            if (
                FrameGasses is not None
                and not FrameGasses.empty
                and "Molecule" in FrameGasses.columns
            ):
                FrameGasses = FrameGasses.copy()
                FrameGasses["Molecule"] = FrameGasses["Molecule"].apply(
                    lambda Molecule: tuple(Molecule)
                    if not isinstance(Molecule, tuple)
                    else Molecule
                )
                O2CountLastFrame = int(
                    (FrameGasses["Molecule"] == ("O", "O")).sum()
                )
            else:
                O2CountLastFrame = 0

            CurrentSmoothedO2Count = ExponentialSmoothing(
                O2CountLastFrame,
                CurrentSmoothedO2Count,
                alpha=OxygenSmoothing,
            )

        # ------------------- Gas fraction / removal for this step -------------------

        PositionLast = OutcarData["Positions"][-1]
        PositionLast = vio.FixElementFormatting(PositionLast)

        GasFraction = an.CalculateGasFraction(PositionLast, GasRatio)

        GassesLast = an.FindGases(
            PositionLast,
            CellDim,
            CovalentRadii=CovalentRadii,
            AtomicRadiusTol=AtomicRadiusTol,
            MinimumComplexity=2,
            MaximumComplexity=3,
            ReturnBondMatrix=False,
        )

        if (
            GassesLast is not None
            and not GassesLast.empty
            and "Molecule" in GassesLast.columns
        ):
            GassesLast = GassesLast.copy()
            GassesLast["Molecule"] = GassesLast["Molecule"].apply(
                lambda Molecule: tuple(Molecule)
                if not isinstance(Molecule, tuple)
                else Molecule
            )
            O2Count = int((GassesLast["Molecule"] == ("O", "O")).sum())
        else:
            O2Count = 0

        GasRemovedStr = CreateGassesRemovedStr(GassesLast)

        # ------------------------ Decide if O2 was added ------------------------

        O2TolEffective = O2TolBase * GasFraction
        O2AddedThisStep = 1 if (
            CurrentSmoothedO2Count <= O2TolEffective
            and O2Count < O2TolEffective
        ) else 0

        CurrentO2AddedCumulative += O2AddedThisStep

        NewRateRow = pd.DataFrame(
            [
                {
                    "Time (fs)": CurrentTimeFs,
                    "O2 Count": O2Count,
                    "Smoothed O2 Count": CurrentSmoothedO2Count,
                    "O2 Added": CurrentO2AddedCumulative,
                    "Gas Removed": GasRemovedStr,
                    "Free Gas Fraction": GasFraction,
                }
            ],
            columns=Columns,
        )

        RateAnalysis = pd.concat(
            [RateAnalysis, NewRateRow],
            ignore_index=True,
        )
        
    # Drop the final row to allow OxidationStep to append latest simulation when run from folder (this is incorrect)
    #RateAnalysis.drop(index=RateAnalysis.index[-1],axis=0,inplace=True)
    
    # ---------------------------- Write results out ----------------------------

    RateAnalysisPathWorkDir = WorkDir / "RateAnalysis.csv"
    RateAnalysisPathRoot = RootDir / "xyz_files" / ("RateAnalysis_%s.csv" % TrajectoryName)

    RateAnalysisPathWorkDir.parent.mkdir(parents=True, exist_ok=True)
    RateAnalysis.to_csv(RateAnalysisPathWorkDir, index=False)

    RateAnalysisPathRoot.parent.mkdir(parents=True, exist_ok=True)
    RateAnalysis.to_csv(RateAnalysisPathRoot, index=False)

    return RateAnalysis


if __name__ == "__main__":
    if len(sys.argv) > 1:
        WorkDirArgument = sys.argv[1]
    else:
        WorkDirArgument = os.getcwd()

    FixRateAnalysis(WorkDirArgument)
