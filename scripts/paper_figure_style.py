"""Shared figure style for the manuscript.

Imported by scripts/paper_figures.py (model-based figures) and scripts/paper_summary_figures.py
(figures from saved JSON artifacts). No torch and no project-model import here.

Conventions
- Final-size figures: build each figure at the width it prints (FULL_W or HALF_W) with constrained
  layout and save without a tight bounding box, so LaTeX includes it at natural size (no scaling).
  JPF_FIGURE_TARGET=spe switches to two-column widths (7.0 in double, 3.5 in single column).
- Text: 8 pt labels, 7 pt ticks and legends, 9 pt bold panel letters; nothing below 6.5 pt.
  TrueType fonts (pdf.fonttype 42), sans serif.
- Phase colors (Okabe-Ito, color-blind safe): oil #009E73, gas #D55E00, water #0072B2; when phases
  share an axis also oil solid, gas dashed, water dash-dot. Phase colors are used only for phases.
- Model styles: Bayesian family black (v1 solid, v2 dotted, width-diverse dashed); positive-basis
  network #E69F00; conditional DCA network #56B4E9; contextual neural #CC79A7; probabilistic forest
  #999999; per-well Arps #8C510A dashed (other per-well DCA baselines the same brown, other markers).
- Field units: bbl/D, Mcf/D, Mbbl, MMcf, Mcf/bbl, bbl/bbl; time in days (years beyond 5 years).
  No STB or scf: the export does not document standard conditions.
- Gray (0.93) marks days after the last report (day 1,080); on cumulative axes it marks volumes
  beyond the well's reported day-1,080 cumulative.
- P-notation is the exceedance convention: P90 = low estimate = 10th percentile of the samples.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

ROOT = Path(__file__).resolve().parents[1]

# ---- widths (inches) -------------------------------------------------------------------------
# draft: single-column pages (\textwidth = 6.5 in). spe: two-column pages.
_WIDTHS = {"draft": (6.5, 3.15), "spe": (7.0, 3.5)}
TARGET = os.environ.get("JPF_FIGURE_TARGET", "draft")
if TARGET not in _WIDTHS:
    raise ValueError(f"JPF_FIGURE_TARGET must be one of {sorted(_WIDTHS)}")
FULL_W, HALF_W = _WIDTHS[TARGET]

# ---- text sizes (points, at final size) ------------------------------------------------------
LABEL_PT, TICK_PT, LEGEND_PT, PANEL_PT, NOTE_PT = 8.0, 7.0, 7.0, 9.0, 6.5

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": LABEL_PT,
    "axes.labelsize": LABEL_PT,
    "axes.titlesize": LABEL_PT,
    "xtick.labelsize": TICK_PT,
    "ytick.labelsize": TICK_PT,
    "legend.fontsize": LEGEND_PT,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.borderaxespad": 0.3,
    "legend.labelspacing": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.titlepad": 3.0,
    "axes.formatter.useoffset": False,
    "axes.formatter.limits": (-7, 7),
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "lines.linewidth": 1.0,
    "lines.markersize": 4.0,
    "patch.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "stixsans",
    "mathtext.default": "regular",
    "figure.dpi": 100,
    "savefig.dpi": 600,
    "savefig.bbox": "standard",
    "savefig.pad_inches": 0.02,
}

# ---- colors, line styles, labels -----------------------------------------------------------
PHASES = ("oil", "gas", "water")
PHASE_COLOR = {"oil": "#009E73", "gas": "#D55E00", "water": "#0072B2"}
PHASE_LS = {"oil": "-", "gas": "--", "water": "-."}
PHASE_LABEL = {"oil": "Oil", "gas": "Gas", "water": "Water"}
RATE_UNIT = {"oil": "bbl/D", "gas": "Mcf/D", "water": "bbl/D"}
# Native volumes are bbl (oil, water) and Mcf (gas); dividing by 1,000 gives Mbbl and MMcf.
VOLUME_UNIT = {"oil": "Mbbl", "gas": "MMcf", "water": "Mbbl"}
VOLUME_DIVISOR = 1e3
SHADE = "0.93"
LAST_REPORT_DAY = 1080.0
DAYS_PER_YEAR = 365.25

# Model names follow the naming table of the writing brief (Stage B directive, B2): the width-diverse ensemble is
# the recommended forecaster and is drawn as the primary (solid black) series.
MODEL_LABEL = {
    "v1": "Frozen version",
    "v1_raw": "Frozen version, before widening",
    "v2": "Intermediate version",
    "arch": "Refined version",
    "basis": "Basis network",
    "dca": "DCA network",
    "neural": "Contextual network",
    "forest": "Forest",
    "arps": "Per-well Arps",
    "duong": "Per-well Duong",
    "sedm": "Per-well SEDM",
    "slab": "Per-well slab scaling",
}
MODEL_STYLE = {
    "v1": dict(color="0.4", ls="--", marker="o", lw=1.1, mfc="white"),
    "v1_raw": dict(color="0.4", ls=":", marker="o", lw=0.9, mfc="white"),
    "v2": dict(color="black", ls=":", marker="D", lw=1.0),
    "arch": dict(color="black", ls="-", marker="o", lw=1.6),
    "basis": dict(color="#E69F00", ls="-", marker="s", lw=1.0),
    "dca": dict(color="#56B4E9", ls="-", marker="^", lw=1.0),
    "neural": dict(color="#CC79A7", ls="-", marker="v", lw=1.0),
    "forest": dict(color="#999999", ls="-", marker="X", lw=1.0),
    "arps": dict(color="#8C510A", ls="--", marker="o", lw=1.0),
    "duong": dict(color="#8C510A", ls=":", marker="^", lw=0.8, mfc="white"),
    "sedm": dict(color="#8C510A", ls=":", marker="v", lw=0.8, mfc="white"),
    "slab": dict(color="#8C510A", ls=":", marker="s", lw=0.8, mfc="white"),
}

ATTRIBUTE_LABEL = {
    "LateralLength_FT": "Lateral length, ft",
    "ProppantIntensity_LBSPerFT": "Proppant intensity, lbm/ft",
    "FluidIntensity_BBLPerFT": "Fluid intensity, bbl/ft",
    "TrueVerticalDepth_FT": "TVD, ft",
}


def apply_style() -> None:
    plt.rcParams.update(RC)


def new_figure(width: float, height: float, nrows: int = 1, ncols: int = 1, **kwargs):
    """plt.subplots with constrained layout (falls back to the pre-3.5 keyword)."""
    kwargs.setdefault("squeeze", False)
    try:
        return plt.subplots(nrows, ncols, figsize=(width, height), layout="constrained", **kwargs)
    except TypeError:
        return plt.subplots(nrows, ncols, figsize=(width, height), constrained_layout=True, **kwargs)


def panel_label(ax, letter: str) -> None:
    """Bold "(a)" at the upper left of the panel (a left title, so constrained layout keeps room)."""
    ax.set_title(f"({letter})", loc="left", fontweight="bold", fontsize=PANEL_PT, pad=3.0)


def label_panels(axes, start: str = "a") -> None:
    for i, ax in enumerate(axes):
        panel_label(ax, chr(ord(start) + i))


def _plain(value, _pos=None) -> str:
    """Plain tick text: thousands separators, no offsets, no scientific notation, true minus sign."""
    if value is None or not np.isfinite(value):
        return ""
    if abs(value) >= 1000:
        text = f"{value:,.0f}"
    elif abs(value - round(value)) < 1e-9 * max(1.0, abs(value)):
        text = f"{value:.0f}"
    else:
        text = f"{value:.6g}"
    return text.replace("-", "\N{MINUS SIGN}")


PLAIN = ticker.FuncFormatter(_plain)


def plain_ticks(ax, axis: str = "both") -> None:
    """Thousands separators and no offset text; on log axes only major ticks are labeled."""
    for name in ("x", "y") if axis == "both" else (axis,):
        target = ax.xaxis if name == "x" else ax.yaxis
        target.set_major_formatter(PLAIN)
        scale = ax.get_xscale() if name == "x" else ax.get_yscale()
        if scale in ("log", "symlog", "functionlog"):
            target.set_minor_formatter(ticker.NullFormatter())


def shade_beyond(ax, start: float = LAST_REPORT_DAY, label: str | None = None, end: float | None = None) -> None:
    """Gray from `start` to the right edge (call after the x limits are final)."""
    x0, x1 = ax.get_xlim()
    stop = x1 if end is None else end
    lo = max(start, x0)
    if stop > lo:
        ax.axvspan(lo, stop, color=SHADE, zorder=0, lw=0)
        if label and start >= x0:
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(start, 0.995, " " + label, transform=trans, ha="left", va="top", fontsize=NOTE_PT,
                    color="0.35", zorder=1)
    ax.set_xlim(x0, x1)


def note(ax, x, y, text, **kwargs):
    """Small annotation text (6.5 pt, dark gray) in axes coordinates unless a transform is given."""
    kwargs.setdefault("transform", ax.transAxes)
    kwargs.setdefault("fontsize", NOTE_PT)
    kwargs.setdefault("color", "0.3")
    return ax.text(x, y, text, **kwargs)


def effective_decline_pct(nominal_per_year):
    """Effective annual decline (%) of a nominal (continuous) decline D in 1/yr: 100 (1 - e^-D)."""
    return 100.0 * (1.0 - np.exp(-np.asarray(nominal_per_year, dtype=float)))


def nominal_from_effective(effective_pct):
    e = np.clip(np.asarray(effective_pct, dtype=float), 0.0, 99.999999) / 100.0
    return -np.log1p(-e)


def terminal_decline(tau_max_days: float) -> dict:
    """Terminal nominal decline 1/tau_max (1/yr) and its effective equivalent.

    The manuscript's derived D_min equivalents use effective = 1 - exp(-365/tau_max)
    1,825 d -> 18.1, 3,650 d -> 9.5, 7,300 d -> 4.9 %/yr.
    """
    return dict(tau_max_days=float(tau_max_days), nominal_per_year=DAYS_PER_YEAR / float(tau_max_days),
                effective_pct_per_year=100.0 * (1.0 - math.exp(-365.0 / float(tau_max_days))))


def legend_handles(entries):
    """entries: list of (kind, label, style dict); kind is 'line', 'patch' or 'marker'."""
    handles = []
    for kind, label, style in entries:
        style = dict(style)
        if kind == "patch":
            handles.append(Patch(label=label, **style))
        elif kind == "marker":
            style.setdefault("ls", "none")
            handles.append(Line2D([], [], label=label, **style))
        else:
            handles.append(Line2D([], [], label=label, **style))
    return handles


def figure_legend(fig, handles, labels, ncol: int, fallback_ax=None, fallback_loc: str = "best"):
    """Legend above the panels, outside the axes (constrained layout reserves room; matplotlib >= 3.7).

    Older matplotlib has no "outside" location; the legend then goes inside `fallback_ax`.
    """
    try:
        return fig.legend(handles, labels, loc="outside upper center", ncol=ncol)
    except (ValueError, TypeError):
        target = fallback_ax if fallback_ax is not None else fig.axes[0]
        return target.legend(handles, labels, loc=fallback_loc)


def save(fig, output: Path, name: str) -> list[str]:
    """PDF (vector, rasterized heavy fills) for LaTeX and a 600-dpi PNG for previews/inspection."""
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        path = output / f"{name}.{ext}"
        fig.savefig(path, dpi=600)
        paths.append(str(path))
    plt.close(fig)
    return paths


def sha256(path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_record(path) -> dict:
    path = Path(path)
    try:
        shown = str(path.resolve().relative_to(ROOT))
    except ValueError:
        shown = str(path)
    return dict(path=shown, sha256=sha256(path))


def run_manifest(script: Path, extra: dict | None = None) -> dict:
    """Commit/dirty state, command, host, versions and script hash for figures.json files."""
    def git(*args):
        try:
            result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False)
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    versions = {}
    for package in ("numpy", "matplotlib", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    status = git("status", "--porcelain=v1")
    commit = git("rev-parse", "--verify", "HEAD")
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), command=[sys.executable, *sys.argv],
                    script=source_record(script), style=source_record(Path(__file__)), git_commit=commit,
                    git_dirty=bool(status) or commit is None, host=socket.gethostname(),
                    platform=platform.platform(), python=sys.version.split()[0], packages=versions,
                    figure_target=TARGET, widths_in=dict(full=FULL_W, half=HALF_W))
    if extra:
        manifest.update(extra)
    return manifest


def json_ready(value):
    """Recursively convert numpy types; non-finite floats become None (strict JSON)."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=1, allow_nan=False) + "\n")
