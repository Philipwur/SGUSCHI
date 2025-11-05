# %%
# src/utils/TestOxidationHarness.py

from typing import Tuple, List, Dict, Optional, Union, Callable
from pathlib import Path
import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from unittest import mock
import sys
import importlib
import importlib.util

# Make `src` importable (so `workflow.*` works)
SysSrcRoot = Path(__file__).resolve().parents[1]
if str(SysSrcRoot) not in sys.path:
    sys.path.insert(0, str(SysSrcRoot))

from workflow import VaspIO as vio   # noqa: E402
from workflow import OxidationAnalysis as an  # noqa: E402


def WriteKeyValueFile(FilePath: Union[str, Path], Mapping: Dict[str, Union[str, float, int]]) -> None:
    FilePath = Path(FilePath)
    Lines = [f"{K}={V}\n" for K, V in Mapping.items()]
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

    (RootDir / "xyz_files").mkdir(parents=True, exist_ok=True)
    WorkDir.mkdir(parents=True, exist_ok=True)

    TemplatePoscarPath = Path(TemplatePoscarPath)
    shutil.copy2(str(TemplatePoscarPath), str(WorkDir / "POSCAR"))

    OxParams: Dict[str, Union[float, int]] = {
        "AtomicRadiusTol": 1.05,
        "O2Tol": 2.0,          # target O2 count; tweak in scenarios
        "OSmoothing": 0.5,     # (0,1]
        "GasRatio": 1.0,
        "InitO2Count": 0,
    }
    WriteKeyValueFile(RootDir / "OxParams", OxParams)

    Radii: Dict[str, float] = {
        "O": 0.66,
        "C": 0.76,
        "N": 0.71,
        "Zr": 1.45,
        # add any bulk species present in your POSCAR (e.g., "H": 0.31)
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

    DeltaFrac = BondLength / AxisLen
    Center = np.array(FracCenter, dtype=float)
    P1 = (Center - np.eye(3)[Axis] * (DeltaFrac / 2.0)) % 1.0
    P2 = (Center + np.eye(3)[Axis] * (DeltaFrac / 2.0)) % 1.0

    NewRows = pd.DataFrame({
        "Element": [Elements[0], Elements[1]],
        "x": [P1[0], P2[0]],
        "y": [P1[1], P2[1]],
        "z": [P1[2], P2[2]],
    })

    PositionOut = pd.concat([Position.reset_index(drop=True), NewRows], ignore_index=True)
    return PositionOut


def MakeZeroVelocity(Position: pd.DataFrame, Existing: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if Existing is not None and len(Existing) == len(Position):
        return Existing.copy()
    return pd.DataFrame(index=Position.index).assign(vx=0.0, vy=0.0, vz=0.0)


def RunScenario(
    TemplatePoscarPath: Union[str, Path],
    ModifyPositionFn: Optional[Callable[[pd.DataFrame, Union[pd.DataFrame, np.ndarray], pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]] = None,
    ExpectedAdds: Optional[int] = None,
    ForceTol: Optional[float] = None
) -> pd.DataFrame:
    """
    Runs workflow.OxidationStep.main once in a sandbox.
      - ModifyPositionFn(Position, CellDim, Velocity) -> (PositionMod, VelocityMod)
      - ExpectedAdds: if provided, assert that the new RateAnalysis row shows the expected
        cumulative 'O2 Added' delta (0 or 1 are common in a single step).
      - ForceTol: override O2Tol in OxParams for this run.
    Returns the resulting RateAnalysis dataframe.
    """
    RootDir, WorkDir, TrajectoryName = BuildSandbox(TemplatePoscarPath)

    Spec = importlib.util.find_spec("workflow.OxidationStep")
    if Spec is None:
        raise ImportError("Could not import workflow.OxidationStep. Check PYTHONPATH.")
    Ox = importlib.import_module("workflow.OxidationStep")

    if ForceTol is not None:
        OxParamsPath = RootDir / "OxParams"
        Lines: List[str] = OxParamsPath.read_text().splitlines()
        NewLines: List[str] = []
        for L in Lines:
            if L.startswith("O2Tol="):
                NewLines.append(f"O2Tol={ForceTol}")
            else:
                NewLines.append(L)
        OxParamsPath.write_text("\n".join(NewLines) + "\n")

    def FakeOutcarParser(_WorkDir: Union[str, Path]) -> Dict[str, Union[float, List[float]]]:
        return {"Temperature": 300.0, "TimesFs": [0.0, 80.0]}

    def NoOp(*args, **kwargs) -> None:
        return None

    def SimpleInsertNewVelocities(Velocity: pd.DataFrame, NewVelocity: pd.DataFrame, ElementSymbol: str) -> pd.DataFrame:
        if isinstance(NewVelocity, pd.DataFrame):
            ToAppend = NewVelocity.copy()
        else:
            Count = 2
            ToAppend = pd.DataFrame({"vx": [0.0] * Count, "vy": [0.0] * Count, "vz": [0.0] * Count})
        VelocityOut = pd.concat([Velocity.reset_index(drop=True), ToAppend.reset_index(drop=True)], ignore_index=True)
        return VelocityOut

    RealReadPoscar = Ox.vio.ReadPoscar

    def PatchedReadPoscar(WorkDirIn: Union[str, Path], GiveVelocities: bool = True):
        Result = RealReadPoscar(WorkDirIn, GiveVelocities=True)
        if not isinstance(Result, tuple):
            raise TypeError("vio.ReadPoscar returned non-tuple result: {}".format(type(Result).__name__))

        if len(Result) == 3:
            Position, CellDim, VelocityIn = Result
        elif len(Result) == 2:
            Position, CellDim = Result
            VelocityIn = None
        else:
            raise TypeError("vio.ReadPoscar returned unexpected tuple length: {}".format(len(Result)))

        Velocity = MakeZeroVelocity(Position, VelocityIn)

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

        Ox.main(str(WorkDir), TestCase=True)

    with mock.patch.object(Ox.vio, "OutcarParser", side_effect=FakeOutcarParser), \
         mock.patch.object(Ox.vio, "WriteXYZ", side_effect=NoOp), \
         mock.patch.object(Ox.vio, "WritePoscar", side_effect=NoOp), \
         mock.patch.object(Ox, "InsertNewVelocities", side_effect=SimpleInsertNewVelocities), \
         mock.patch.object(Ox.vio, "ReadPoscar", side_effect=PatchedReadPoscar):

        Ox.main(str(WorkDir), TestCase=False)

    RateCsv = Path(WorkDir) / "RateAnalysis.csv"
    if not RateCsv.exists():
        raise FileNotFoundError("Expected RateAnalysis.csv at {}".format(RateCsv))
    RateAnalysis = pd.read_csv(str(RateCsv))

    if ExpectedAdds is not None:
        if len(RateAnalysis) >= 2:
            Delta = float(RateAnalysis["O2 Added"].iloc[-1]) - float(RateAnalysis["O2 Added"].iloc[-2])
        else:
            Delta = float(RateAnalysis["O2 Added"].iloc[-1])
        assert int(round(Delta)) == int(ExpectedAdds), "Expected O2 add delta {}, got {}".format(ExpectedAdds, Delta)

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
        return Position, Velocity
    return RunScenario(TemplatePoscarPath, ModifyPositionFn=Modify, ExpectedAdds=0, ForceTol=0.0)


def ScenarioRemoveCOThenAddO2(TemplatePoscarPath: Union[str, Path]) -> pd.DataFrame:
    """
    Inject a CO molecule (non-O2) into vacuum so RemoveNonO2Gasses strips it.
    Then set a generous O2Tol so one O2 gets added.
    """
    def Modify(Position: pd.DataFrame, CellDim: Union[pd.DataFrame, np.ndarray], Velocity: pd.DataFrame):
        Position1 = SynthesizeDiatomic(Position, CellDim, Elements=("C", "O"),
                                       FracCenter=(0.77, 0.79, 0.81), BondLength=1.15, Axis=0)
        return Position1, Velocity
    return RunScenario(TemplatePoscarPath, ModifyPositionFn=Modify, ExpectedAdds=1, ForceTol=10.0)


if __name__ == "__main__":
    WorkDir = os.getcwd()
    Template = Path(WorkDir) / ".." / "Test" / "POSCAR_ZrCN_Vacuum"

    print("Running smoke (no add)...")
    RA1 = ScenarioSmokeNoAdd(Template)
    print(RA1.tail(3))

    print("Running remove CO then add O2...")
    RA2 = ScenarioRemoveCOThenAddO2(Template)
    print(RA2.tail(3))
# %%
