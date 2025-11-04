#%%
# src/utils/TestOxidationHarness.py

from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path
import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from unittest import mock
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from workflow import VaspIO as vio
from workflow import OxidationAnalysis as an
# Adjust these imports to your package layout
# Assumes your runtime is in src/workflow/OxidationStep.py and importable as workflow.OxidationStep
import importlib

def WriteKeyValueFile(FilePath: Union[str, Path], Mapping: Dict[str, Union[str, float, int]]) -> None:
    FilePath = Path(FilePath)
    Lines = [f"{k}={v}\n" for k, v in Mapping.items()]
    FilePath.write_text("".join(Lines))

def BuildSandbox(TemplatePoscarPath: Union[str, Path]) -> Tuple[Path, Path, str]:
    """
    Create an on-disk sandbox with the exact structure your runtime expects:
    RootDir/
      xyz_files/
      OxParams
      CovalentRadii
      Jobs/
        TrajectoryName/
          POSCAR   <-- WorkDir (two levels below RootDir)

    Returns:
      (RootDir, WorkDir, TrajectoryName)
    """
    TmpRoot = Path(tempfile.mkdtemp(prefix="oxidation_sandbox_")).resolve()
    RootDir = TmpRoot / "RootDir"
    JobsDir = RootDir / "Jobs"
    TrajectoryName = "Trajectory_Test"
    WorkDir = JobsDir / TrajectoryName

    # Folders
    (RootDir / "xyz_files").mkdir(parents=True, exist_ok=True)
    WorkDir.mkdir(parents=True, exist_ok=True)

    # Files: POSCAR (copy from template)
    TemplatePoscarPath = Path(TemplatePoscarPath)
    shutil.copy2(str(TemplatePoscarPath), str(WorkDir / "POSCAR"))

    # Files: OxParams (choose values that can trigger O2 addition if needed)
    OxParams = {
        "AtomicRadiusTol": 1.05,
        "O2Tol": 2.0,          # target O2 count; adjust in scenarios to test add/no-add
        "OSmoothing": 0.5,     # exponential smoothing alpha in (0,1]
        "GasRatio": 1.0,       # used by CalculateGasFraction
        "InitO2Count": 0,
    }
    WriteKeyValueFile(RootDir / "OxParams", OxParams)

    # Files: CovalentRadii (include everything that appears in bulk + gases)
    # Values are typical covalent radii in Å; tweak to your dataset if needed.
    Radii = {
        "O": 0.66,
        "C": 0.76,
        "N": 0.71,
        "Zr": 1.45,
        # add any other bulk species present in your POSCAR, e.g., "H": 0.31
    }
    WriteKeyValueFile(RootDir / "CovalentRadii", Radii)

    return RootDir, WorkDir, TrajectoryName

def SynthesizeDiatomic(
    Position: pd.DataFrame,
    CellDim: Union[pd.DataFrame, np.ndarray],
    Elements: Tuple[str, str],
    FracCenter: Tuple[float, float, float] = (0.80, 0.80, 0.80),
    BondLength: float = 1.15,
    Axis: int = 0
) -> pd.DataFrame:
    """
    Append a diatomic molecule (in fractional coords) into a vacuum pocket so your
    real FindGases can detect it. Uses lattice lengths to convert a small fractional
    offset corresponding to the BondLength.
    """
    if isinstance(CellDim, pd.DataFrame):
        Lattice = CellDim.to_numpy()
    else:
        Lattice = np.asarray(CellDim, dtype=float)
    AxisVec = Lattice[:, Axis]
    AxisLen = float(np.linalg.norm(AxisVec))
    if AxisLen == 0.0:
        raise ValueError("CellDim axis length is zero; invalid lattice.")

    # Convert desired BondLength along a single axis into fractional offset
    DeltaFrac = BondLength / AxisLen
    p = np.array(FracCenter, dtype=float)
    p1 = (p - np.eye(3)[Axis] * (DeltaFrac / 2.0)) % 1.0
    p2 = (p + np.eye(3)[Axis] * (DeltaFrac / 2.0)) % 1.0

    NewRows = pd.DataFrame({
        "Element": [Elements[0], Elements[1]],
        "x": [p1[0], p2[0]],
        "y": [p1[1], p2[1]],
        "z": [p1[2], p2[2]],
    })
    # Concatenate and reset a clean RangeIndex (your downstream code resets anyway)
    PositionOut = pd.concat([Position.reset_index(drop=True), NewRows], ignore_index=True)
    return PositionOut

def MakeZeroVelocity(Position: pd.DataFrame, Existing: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if Existing is not None and len(Existing) == len(Position):
        return Existing.copy()
    return pd.DataFrame(index=Position.index).assign(vx=0.0, vy=0.0, vz=0.0)

def RunScenario(
    TemplatePoscarPath: Union[str, Path],
    ModifyPositionFn=None,
    ExpectedAdds: Optional[int] = None,
    ForceTol: Optional[float] = None
) -> pd.DataFrame:
    """
    Runs workflow.OxidationStep.main once in a sandbox.
    - ModifyPositionFn: a callable(Position, CellDim) -> (PositionMod, VelocityMod)
      used to inject gases (e.g., CO, N2) before the main logic runs.
    - ExpectedAdds: if provided, assert that the new RateAnalysis row shows the expected
      cumulative 'O2 Added' delta (0 or 1 are common in a single step).
    - ForceTol: if provided, override O2Tol in OxParams to force add/no-add behavior.
    Returns the resulting RateAnalysis dataframe.
    """
    RootDir, WorkDir, TrajectoryName = BuildSandbox(TemplatePoscarPath)

    # Import (or reload) your runtime after sandbox exists
    Spec = importlib.util.find_spec("workflow.OxidationStep")
    if Spec is None:
        raise ImportError("Could not import workflow.OxidationStep. Check PYTHONPATH.")
    Ox = importlib.import_module("workflow.OxidationStep")

    # Optionally adjust O2Tol to force behavior in this scenario
    if ForceTol is not None:
        OxParamsPath = RootDir / "OxParams"
        Lines = (RootDir / "OxParams").read_text().splitlines()
        NewLines: List[str] = []
        for L in Lines:
            if L.startswith("O2Tol="):
                NewLines.append(f"O2Tol={ForceTol}")
            else:
                NewLines.append(L)
        OxParamsPath.write_text("\n".join(NewLines) + "\n")

    # We will patch vio.OutcarParser so you don't need a real OUTCAR;
    # and patch vio.WriteXYZ / vio.WritePoscar to no-ops for safety.
    # Also patch InsertNewVelocities to a simple append that keeps shapes aligned.
    def FakeOutcarParser(_WorkDir: Union[str, Path]) -> Dict[str, Union[float, List[float]]]:
        return {"Temperature": 300.0, "TimesFs": [0.0, 80.0]}

    def NoOp(*args, **kwargs) -> None:
        return None

    def SimpleInsertNewVelocities(Velocity: pd.DataFrame, NewVelocity: pd.DataFrame, ElementSymbol: str) -> pd.DataFrame:
        # Expect NewVelocity to have 2 rows for O2; if not, coerce.
        if isinstance(NewVelocity, pd.DataFrame):
            ToAppend = NewVelocity.copy()
        else:
            # If a different type is returned by your generator, make zeros
            Count = 2
            ToAppend = pd.DataFrame({"vx": [0.0]*Count, "vy": [0.0]*Count, "vz": [0.0]*Count})
        VelocityOut = pd.concat([Velocity.reset_index(drop=True), ToAppend.reset_index(drop=True)], ignore_index=True)
        return VelocityOut

    # We also intercept vio.ReadPoscar to optionally inject molecules into the Position
    RealReadPoscar = Ox.vio.ReadPoscar

    def PatchedReadPoscar(WorkDirIn: Union[str, Path], GiveVelocities: bool = True):
        Position, CellDim, Velocity = RealReadPoscar(WorkDirIn, GiveVelocities=True)
        # Basic sanitation for tests: ensure Velocity is present
        Velocity = MakeZeroVelocity(Position, Velocity)
        if ModifyPositionFn is not None:
            PositionMod, VelocityMod = ModifyPositionFn(Position, CellDim, Velocity)
            Position = PositionMod
            Velocity = MakeZeroVelocity(Position, VelocityMod)
        return Position, CellDim, Velocity

    with mock.patch.object(Ox.vio, "OutcarParser", side_effect=FakeOutcarParser), \
         mock.patch.object(Ox.vio, "WriteXYZ", side_effect=NoOp), \
         mock.patch.object(Ox.vio, "WritePoscar", side_effect=NoOp), \
         mock.patch.object(Ox, "InsertNewVelocities", side_effect=SimpleInsertNewVelocities), \
         mock.patch.object(Ox.vio, "ReadPoscar", side_effect=PatchedReadPoscar):

        # Run one step in "test mode" to avoid unintended writes
        Ox.main(str(WorkDir), TestCase=True)

    # Load the produced (or updated) RateAnalysis
    # In TestCase=True branch you still update the in-memory RateAnalysis,
    # but not write; for assertion we can re-run main with not TestCase,
    # or directly construct expected outputs. To keep it concrete, run once
    # with writes enabled but still patched to no-ops (safe).
    with mock.patch.object(Ox.vio, "OutcarParser", side_effect=FakeOutcarParser), \
         mock.patch.object(Ox.vio, "WriteXYZ", side_effect=NoOp), \
         mock.patch.object(Ox.vio, "WritePoscar", side_effect=NoOp), \
         mock.patch.object(Ox, "InsertNewVelocities", side_effect=SimpleInsertNewVelocities), \
         mock.patch.object(Ox.vio, "ReadPoscar", side_effect=PatchedReadPoscar):

        Ox.main(str(WorkDir), TestCase=False)

    RateCsv = Path(WorkDir) / "RateAnalysis.csv"
    if not RateCsv.exists():
        raise FileNotFoundError(f"Expected RateAnalysis.csv at {RateCsv}")
    RateAnalysis = pd.read_csv(str(RateCsv))

    if ExpectedAdds is not None:
        # Compare the last delta in 'O2 Added' vs the previous row
        if len(RateAnalysis) >= 2:
            Delta = float(RateAnalysis["O2 Added"].iloc[-1]) - float(RateAnalysis["O2 Added"].iloc[-2])
        else:
            Delta = float(RateAnalysis["O2 Added"].iloc[-1])
        assert int(round(Delta)) == int(ExpectedAdds), f"Expected O2 add delta {ExpectedAdds}, got {Delta}"

    return RateAnalysis

# -------------------
# Ready-made scenarios
# -------------------

def ScenarioSmokeNoAdd(TemplatePoscarPath: Union[str, Path]) -> pd.DataFrame:
    """
    Baseline: ensure main runs, writes RateAnalysis, and does NOT add O2.
    We force O2Tol=0 so the add condition is false.
    """
    def Modify(Position: pd.DataFrame, CellDim: Union[pd.DataFrame, np.ndarray], Velocity: pd.DataFrame):
        # No modifications
        return Position, Velocity
    return RunScenario(TemplatePoscarPath, ModifyPositionFn=Modify, ExpectedAdds=0, ForceTol=0.0)

def ScenarioRemoveCOThenAddO2(TemplatePoscarPath: Union[str, Path]) -> pd.DataFrame:
    """
    Inject a CO molecule (non-O2) into vacuum so RemoveNonO2Gasses strips it.
    Then set a generous O2Tol so one O2 gets added.
    """
    def Modify(Position: pd.DataFrame, CellDim: Union[pd.DataFrame, np.ndarray], Velocity: pd.DataFrame):
        # Add CO into vacuum pocket; your FindGases should pick it up and removal should delete it
        Position1 = SynthesizeDiatomic(Position, CellDim, Elements=("C", "O"), FracCenter=(0.77, 0.79, 0.81), BondLength=1.15, Axis=0)
        return Position1, Velocity
    # Force a threshold that almost surely triggers an addition after smoothing
    return RunScenario(TemplatePoscarPath, ModifyPositionFn=Modify, ExpectedAdds=1, ForceTol=10.0)

if __name__ == "__main__":
    
    
    # Example manual runs:
    Workdir = os.getcwd()
    Template = Path(f"{Workdir}/../Test/POSCAR_ZrCN_Vacuum")
    
    print("Running smoke (no add)...")
    RA1 = ScenarioSmokeNoAdd(Template)
    print(RA1.tail(3))

    print("Running remove CO then add O2...")
    RA2 = ScenarioRemoveCOThenAddO2(Template)
    print(RA2.tail(3))

# %%
