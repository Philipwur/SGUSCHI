"""
CompareMethodsCumulative.py

Compares partial pressure (bar) and impingement flux (m⁻² s⁻¹) from three
counting/volume methods:
  - Old   : FastCountO2 + ZrDistance  (src/DevArea/Merged_XYZ_ZrC/PartialPressure)
  - New-Zr: XYZFindGases + ZrDistance (Results/ZrC/PP_ZrDistance_XYZFindGases_*.csv)
  - New-RC: XYZFindGases + RayCast    (Results/ZrC/PP_RayCast_XYZFindGases_*.csv)

Each method is compared using the cumulative average (running mean from t=0,
final converged value).

Note: ZrDistance was used as the volume estimator in the pressure control
algorithm during the AIMD simulations. New-Zr is therefore the most consistent
method for normalising rate values derived from those same simulations.

Outputs:
  - Comparison_PP_Methods.csv           — table of all per-run values
  - Comparison_PP_Methods.html          — 1x2 per-run grouped bar chart (PP + flux)
  - Comparison_TemperatureAverages.html — 1x2 temperature-mean bar chart with error bars
"""

import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────

OLD_PP_DIR = Path(r"c:\Users\pdwurzner\OneDrive - Delft University of Technology\Research\Shared_workspace\SGUSCHI\SGUSCHI\src\DevArea\Merged_XYZ_ZrC\PartialPressure")
NEW_DIR    = Path(r"c:\Users\pdwurzner\OneDrive - Delft University of Technology\Research\Shared_workspace\SGUSCHI\SGUSCHI\src\DevArea\Data\ZrC_Paper\PartialPressureAnalysis\Results\ZrC")
OUTPUT_DIR = NEW_DIR

ATM_TO_BAR = 1.01325

# ── Physical constants (Hertz-Knudsen flux) ──────────────────────────────────

_O2_MASS_KG = 32.0 * 1.66054e-27   # kg
_KB          = 1.380649e-23         # J K⁻¹


def ComputeImpingementFlux(PressureBar: np.ndarray, TempK: float) -> np.ndarray:
    """Hertz-Knudsen flux: Z = P / sqrt(2 pi m kB T)  [molecules m⁻² s⁻¹]"""
    P_Pa = np.asarray(PressureBar, dtype=float) * 1.0e5
    return P_Pa / np.sqrt(2.0 * np.pi * _O2_MASS_KG * _KB * TempK)


# ── Helpers ──────────────────────────────────────────────────────────────────

def ParseTempRun(Stem: str):
    """Return (temperature, run) from a stem like '973_2_Pressure' or 'PP_ZrDistance_XYZFindGases_973_2'."""
    Match = re.search(r'(\d{3,4})_(\d)', Stem)
    return (int(Match.group(1)), int(Match.group(2))) if Match else (None, None)


# ── Loaders ──────────────────────────────────────────────────────────────────

def LoadOldMetrics(OldDir: Path) -> pd.DataFrame:
    """
    Read each *_Pressure.csv (old FastCountO2 + ZrDistance method) and compute
    cumulative and sliding-window metrics for both pressure and flux.

    Old CSV columns: Step, Time_fs, O2_Count, Gas_Volume_A3, Void_Fraction, Pressure_atm.
    """
    Records = []
    for CsvPath in sorted(OldDir.glob("*_Pressure.csv")):
        Temp, Run = ParseTempRun(CsvPath.stem)
        if Temp is None:
            continue

        Df          = pd.read_csv(CsvPath)
        PressureBar = Df['Pressure_atm'].to_numpy() * ATM_TO_BAR
        FluxArr     = ComputeImpingementFlux(PressureBar, float(Temp))

        CumPressure = np.cumsum(PressureBar) / np.arange(1, len(PressureBar) + 1)
        CumFlux     = np.cumsum(FluxArr)     / np.arange(1, len(FluxArr)     + 1)

        Records.append({
            'Temperature (K)':                 Temp,
            'Run Number':                       Run,
            'Old Cumulative PP (bar)':          float(CumPressure[-1]),
            'Old Cumulative Flux (m^-2 s^-1)': float(CumFlux[-1]),
        })

    return (pd.DataFrame(Records)
            .sort_values(['Temperature (K)', 'Run Number'])
            .reset_index(drop=True))


def LoadNewMetrics(NewDir: Path, VolMethod: str) -> pd.DataFrame:
    """
    Read each PP_{VolMethod}_XYZFindGases_T_R.csv (new method) and extract
    cumulative and sliding-window metrics for both pressure and flux.

    New CSV columns include pre-computed: Cumulative Average (bar),
    Smoothed Pressure (bar), Cumulative Flux (m^-2 s^-1), Smoothed Flux (m^-2 s^-1).
    """
    Records = []
    for CsvPath in sorted(NewDir.glob(f"PP_{VolMethod}_XYZFindGases_*.csv")):
        Temp, Run = ParseTempRun(CsvPath.stem)
        if Temp is None:
            continue

        Df = pd.read_csv(CsvPath)
        Records.append({
            'Temperature (K)':                               Temp,
            'Run Number':                                     Run,
            f'New-{VolMethod} Cumulative PP (bar)':           float(Df['Cumulative Average (bar)'].iloc[-1]),
            f'New-{VolMethod} Cumulative Flux (m^-2 s^-1)':  float(Df['Cumulative Flux (m^-2 s^-1)'].iloc[-1]),
        })

    return (pd.DataFrame(Records)
            .sort_values(['Temperature (K)', 'Run Number'])
            .reset_index(drop=True))


# ── Figures ──────────────────────────────────────────────────────────────────

def PlotPerRunComparison(Df: pd.DataFrame, RunLabels: list, BarColors: dict, OutputDir: Path) -> None:
    """
    1×2 grouped bar chart comparing all three methods per run:
      Col 1 — Partial Pressure (bar), cumulative average (final converged value)
      Col 2 — Impingement Flux (m⁻² s⁻¹), cumulative average (final converged value)
    """
    Figure = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.08,
        subplot_titles=(
            "Cumulative Average PP  (final value)",
            "Cumulative Average Flux  (final value)",
        ),
    )

    Specs = [
        # (col, df_col_old, df_col_zr, df_col_rc)
        (1, 'Old Cumulative PP (bar)',          'New-ZrDistance Cumulative PP (bar)',          'New-RayCast Cumulative PP (bar)'),
        (2, 'Old Cumulative Flux (m^-2 s^-1)',  'New-ZrDistance Cumulative Flux (m^-2 s^-1)', 'New-RayCast Cumulative Flux (m^-2 s^-1)'),
    ]

    for PanelIndex, (Col, OldCol, ZrCol, RcCol) in enumerate(Specs):
        ShowLegend = PanelIndex == 0
        Figure.add_trace(go.Bar(
            name='Old (FastCountO2 + ZrDist)', x=RunLabels, y=Df[OldCol],
            marker_color=BarColors['Old'], legendgroup='Old', showlegend=ShowLegend,
        ), row=1, col=Col)
        Figure.add_trace(go.Bar(
            name='New: ZrDistance + XYZFindGases', x=RunLabels, y=Df[ZrCol],
            marker_color=BarColors['Zr'], legendgroup='Zr', showlegend=ShowLegend,
        ), row=1, col=Col)
        Figure.add_trace(go.Bar(
            name='New: RayCast + XYZFindGases', x=RunLabels, y=Df[RcCol],
            marker_color=BarColors['RC'], legendgroup='RC', showlegend=ShowLegend,
        ), row=1, col=Col)

    Figure.update_layout(
        barmode='group',
        title='Partial Pressure & Flux Method Comparison — Cumulative Average (per run)',
        template='plotly_white',
        height=500,
        legend=dict(x=1.01, y=0.99),
        margin=dict(r=280),
        hovermode='x unified',
    )
    Figure.update_yaxes(title_text='Pressure (bar)', row=1, col=1)
    Figure.update_yaxes(title_text='Flux (m-2 s-1)', row=1, col=2)
    Figure.update_xaxes(title_text='Run',            row=1, col=1)
    Figure.update_xaxes(title_text='Run',            row=1, col=2)

    Figure.write_html(str(OutputDir / "Comparison_PP_Methods.html"))


def PlotTemperatureAverages(Df: pd.DataFrame, BarColors: dict, OutputDir: Path) -> None:
    """
    1×2 bar chart of temperature-averaged cumulative metrics with error bars
    (std across runs):
      Col 1 — Partial Pressure (bar)
      Col 2 — Impingement Flux (m⁻² s⁻¹)
    """
    Temps      = sorted(Df['Temperature (K)'].dropna().unique())
    TempLabels = [f"{int(T)} K" for T in Temps]

    GroupMean = Df.groupby('Temperature (K)').mean(numeric_only=True)
    GroupStd  = Df.groupby('Temperature (K)').std(numeric_only=True)

    Figure = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.08,
        subplot_titles=(
            "Cumulative Average PP  (mean ± std across runs)",
            "Cumulative Average Flux  (mean ± std across runs)",
        ),
    )

    Specs = [
        (1, 'Old Cumulative PP (bar)',          'New-ZrDistance Cumulative PP (bar)',          'New-RayCast Cumulative PP (bar)'),
        (2, 'Old Cumulative Flux (m^-2 s^-1)',  'New-ZrDistance Cumulative Flux (m^-2 s^-1)', 'New-RayCast Cumulative Flux (m^-2 s^-1)'),
    ]

    MethodSpecs = [
        ('Old (FastCountO2 + ZrDist)',     'Old', BarColors['Old']),
        ('New: ZrDistance + XYZFindGases', 'Zr',  BarColors['Zr']),
        ('New: RayCast + XYZFindGases',    'RC',  BarColors['RC']),
    ]

    for PanelIndex, (Col, OldCol, ZrCol, RcCol) in enumerate(Specs):
        ShowLegend = PanelIndex == 0
        for (MethodName, MethodKey, Color), DataCol in zip(
            MethodSpecs, [OldCol, ZrCol, RcCol]
        ):
            MeanVals = [GroupMean.loc[T, DataCol] if T in GroupMean.index and DataCol in GroupMean.columns else np.nan for T in Temps]
            StdVals  = [GroupStd.loc[T,  DataCol] if T in GroupStd.index  and DataCol in GroupStd.columns  else np.nan for T in Temps]

            Figure.add_trace(go.Bar(
                name=MethodName,
                x=TempLabels,
                y=MeanVals,
                error_y=dict(type='data', array=StdVals, visible=True),
                marker_color=Color,
                legendgroup=MethodKey,
                showlegend=ShowLegend,
            ), row=1, col=Col)

    Figure.update_layout(
        barmode='group',
        title='Partial Pressure & Flux: Temperature Averages — Cumulative (mean ± std)',
        template='plotly_white',
        height=500,
        legend=dict(x=1.01, y=0.99),
        margin=dict(r=280),
        hovermode='x unified',
    )
    Figure.update_yaxes(title_text='Pressure (bar)', row=1, col=1)
    Figure.update_yaxes(title_text='Flux (m-2 s-1)', row=1, col=2)
    Figure.update_xaxes(title_text='Temperature',    row=1, col=1)
    Figure.update_xaxes(title_text='Temperature',    row=1, col=2)

    Figure.write_html(str(OutputDir / "Comparison_TemperatureAverages.html"))


# ── Main ─────────────────────────────────────────────────────────────────────

def BuildInterTemperatureVariability(
    TempMeansDf: pd.DataFrame,
    MethodColumns: list,
) -> pd.DataFrame:
    """
    Compute inter-temperature variability from temperature-mean values.

    MethodColumns: list of tuples (MethodLabel, ColumnName)
    """
    Records = []
    for MethodLabel, ColumnName in MethodColumns:
        Values = TempMeansDf[ColumnName].dropna().to_numpy(dtype=float)
        if Values.size == 0:
            Records.append({
                'Method': MethodLabel,
                'N Temperatures': 0,
                'Mean Across Temperatures': np.nan,
                'Std Across Temperatures': np.nan,
                'Min Temperature Mean': np.nan,
                'Max Temperature Mean': np.nan,
                'Range (Max-Min)': np.nan,
                'CV (%)': np.nan,
            })
            continue

        MeanVal  = float(np.mean(Values))
        StdVal   = float(np.std(Values, ddof=0))
        MinVal   = float(np.min(Values))
        MaxVal   = float(np.max(Values))
        RangeVal = MaxVal - MinVal
        CvPct    = (StdVal / MeanVal * 100.0) if MeanVal != 0.0 else np.nan

        Records.append({
            'Method': MethodLabel,
            'N Temperatures': int(Values.size),
            'Mean Across Temperatures': MeanVal,
            'Std Across Temperatures': StdVal,
            'Min Temperature Mean': MinVal,
            'Max Temperature Mean': MaxVal,
            'Range (Max-Min)': RangeVal,
            'CV (%)': CvPct,
        })

    return pd.DataFrame(Records)


def main():
    DfOld = LoadOldMetrics(OLD_PP_DIR)
    DfZr  = LoadNewMetrics(NEW_DIR, 'ZrDistance')
    DfRc  = LoadNewMetrics(NEW_DIR, 'RayCast')

    Df = (DfOld
          .merge(DfZr, on=['Temperature (K)', 'Run Number'], how='outer')
          .merge(DfRc,  on=['Temperature (K)', 'Run Number'], how='outer')
          .sort_values(['Temperature (K)', 'Run Number'])
          .reset_index(drop=True))

    # Ratio columns (pressure only — flux ratios are identical by linearity)
    Df['Ratio New-Zr/Old (Cumulative PP)'] = Df['New-ZrDistance Cumulative PP (bar)'] / Df['Old Cumulative PP (bar)']
    Df['Ratio New-RC/Old (Cumulative PP)'] = Df['New-RayCast Cumulative PP (bar)']    / Df['Old Cumulative PP (bar)']

    OutCsv = OUTPUT_DIR / "Comparison_PP_Methods.csv"
    Df.to_csv(OutCsv, index=False)
    print(f"Table saved to {OutCsv}")
    print(Df.to_string(index=False))

    PpCols = [
        'Old Cumulative PP (bar)',
        'New-ZrDistance Cumulative PP (bar)',
        'New-RayCast Cumulative PP (bar)',
    ]
    FluxCols = [
        'Old Cumulative Flux (m^-2 s^-1)',
        'New-ZrDistance Cumulative Flux (m^-2 s^-1)',
        'New-RayCast Cumulative Flux (m^-2 s^-1)',
    ]
    BaseCols = ['Temperature (K)', 'Run Number']

    PpTable = Df[BaseCols + PpCols].copy()
    FluxTable = Df[BaseCols + FluxCols].copy()

    TempMeanPp = (PpTable
                  .groupby('Temperature (K)', as_index=False)[PpCols]
                  .mean()
                  .sort_values('Temperature (K)')
                  .reset_index(drop=True))
    TempMeanFlux = (FluxTable
                    .groupby('Temperature (K)', as_index=False)[FluxCols]
                    .mean()
                    .sort_values('Temperature (K)')
                    .reset_index(drop=True))

    print("\n=== Per-run cumulative partial pressure (bar) ===")
    print(PpTable.to_string(index=False, float_format=lambda X: f"{X:.6f}"))

    print("\n=== Per-run cumulative impingement flux (m^-2 s^-1) ===")
    print(FluxTable.to_string(index=False, float_format=lambda X: f"{X:.4e}"))

    print("\n=== Temperature-mean cumulative partial pressure (bar) ===")
    print(TempMeanPp.to_string(index=False, float_format=lambda X: f"{X:.6f}"))

    print("\n=== Temperature-mean cumulative impingement flux (m^-2 s^-1) ===")
    print(TempMeanFlux.to_string(index=False, float_format=lambda X: f"{X:.4e}"))

    PpInterTempVar = BuildInterTemperatureVariability(
        TempMeanPp,
        [
            ('Old (FastCountO2 + ZrDist)', 'Old Cumulative PP (bar)'),
            ('New: ZrDistance + XYZFindGases', 'New-ZrDistance Cumulative PP (bar)'),
            ('New: RayCast + XYZFindGases', 'New-RayCast Cumulative PP (bar)'),
        ],
    )
    FluxInterTempVar = BuildInterTemperatureVariability(
        TempMeanFlux,
        [
            ('Old (FastCountO2 + ZrDist)', 'Old Cumulative Flux (m^-2 s^-1)'),
            ('New: ZrDistance + XYZFindGases', 'New-ZrDistance Cumulative Flux (m^-2 s^-1)'),
            ('New: RayCast + XYZFindGases', 'New-RayCast Cumulative Flux (m^-2 s^-1)'),
        ],
    )

    print("\n=== Inter-temperature variability: partial pressure (bar) ===")
    print(PpInterTempVar.to_string(index=False, float_format=lambda X: f"{X:.6f}"))

    print("\n=== Inter-temperature variability: impingement flux (m^-2 s^-1) ===")
    print(FluxInterTempVar.to_string(index=False, float_format=lambda X: f"{X:.4e}"))

    RunLabels = [
        f"{int(Row['Temperature (K)'])}K R{int(Row['Run Number'])}"
        for _, Row in Df.iterrows()
    ]
    BarColors = {'Old': 'steelblue', 'Zr': 'darkorange', 'RC': 'forestgreen'}

    PlotPerRunComparison(Df, RunLabels, BarColors, OUTPUT_DIR)
    print(f"Per-run plot saved to  {OUTPUT_DIR / 'Comparison_PP_Methods.html'}")

    PlotTemperatureAverages(Df, BarColors, OUTPUT_DIR)
    print(f"Temperature averages plot saved to  {OUTPUT_DIR / 'Comparison_TemperatureAverages.html'}")


if __name__ == "__main__":
    main()
