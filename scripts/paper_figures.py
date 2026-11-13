"""Model-based figures for the manuscript (Section 5, reading a forecast in reservoir-engineering terms).

Uses the fitted Bayesian decline model (by default "the model": the five v1 fits pooled with equal
samples per fit and the calibrated settings used for scoring), the matching positive-basis and
conditional DCA network fits for the comparison panels, and VALIDATION wells only (no test record).
Example wells A, B and C are chosen by fixed rules on observed reports to day 1,080 (pick_wells,
unchanged). Every number a caption or the text quotes is written to figures.json together with the
model label, checkpoints and their SHA-256, the calibration settings, seeds and sample counts.

Figures (files are named by content, PDF + 600-dpi PNG):
  fig_updating          well A at forecast dates 0, 90 and 360 d: latent rates, cumulative oil,
                                EUR by phase
  fig_decline           well A at 360 d: rate-cumulative (+ zoom), D(t) and b(t), model vs
                                positive-basis network
  fig_modes             decay-time spectrum (A, B, C), mode decomposition and EUR timing (A)
  fig_phase_coupling    well B: GOR and WOR against cumulative oil, joint three-year oil-gas sample
  fig_attributes        one-at-a-time attribute effects on the prior (forecast date 0)
  supp_fig_correlation  cross-phase correlation of log rates (200 wells)
Data blocks without a figure:
  worked_example        forecast card for well A (written with fig_updating)
  property_audit        latent property shares (96 wells, origins 0/90/360); also written under
                        the legacy key fig8_audit, which scripts/paper_results_bundle.py reads
  fig5_correlation      legacy alias of the supp_fig_correlation numbers (read by the bundle)

Pooling: a family's samples are split equally over its checkpoints and member i uses seed + i, so a
single checkpoint reproduces the earlier single-fit numbers wherever seeds and sample counts are
unchanged. All Bayesian members must share the latent law (decay grid, ratio index) and phase scales.

    python scripts/paper_figures.py \
        --bayes models/ensemble/bayes_arch_seed42_model.pt ... --bayes models/ensemble/bayes_arch_seed46_model.pt \
        --basis models/comparators/basis_secondary_seed42_model.pt ... (seeds 42-46) \
        --dca models/comparators/dca_secondary_seed42_model.pt ... (seeds 42-46) \
        --model-label arch --report-inflation 1.0 --prior-inflation 1.0 [--device cuda] [--only fig_decline]

For --model-label v1 the calibrated settings default to results/v1/cal/calibration/calibration.json
(report_inflation 1.15, prior_inflation 1.0); pass --calibration none or explicit --report-inflation /
--prior-inflation for other models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import torch
from matplotlib import ticker

from paper_figure_style import (FULL_W, HALF_W, MODEL_LABEL, figure_legend, MODEL_STYLE, NOTE_PT, PHASE_COLOR, PHASE_LABEL,
                                PHASE_LS, PHASES, PLAIN, RATE_UNIT, VOLUME_DIVISOR, VOLUME_UNIT, ATTRIBUTE_LABEL,
                                LAST_REPORT_DAY, DAYS_PER_YEAR, apply_style, effective_decline_pct, label_panels,
                                legend_handles, new_figure, nominal_from_effective, note, plain_ticks, run_manifest,
                                save, sha256, shade_beyond, source_record, terminal_decline, write_json)
from depletion_dev import DAYS, load_split
from depletion_properties import rate_checks
from joint_production.depletion_bayes import DepletionBayesForecaster

FAST_TAU_DAYS = 100.0
PD_WELLS, PD_SAMPLES = 200, 50  # partial dependence of the attribute figure: training wells, samples per well
RATE_FLOOR = 10.0  # lower limit of the log rate axes, bbl/D or Mcf/D (confirm after rendering)
FIGURES = ("fig_updating", "fig_decline", "fig_modes", "fig_phase_coupling", "fig_attributes",
           "supp_fig_correlation", "property_audit")
DEFAULT_CALIBRATION = ROOT / "results/v1/cal/calibration/calibration.json"
BASIS_COLOR = MODEL_STYLE["basis"]["color"]
LEGACY_KEYS = {"fig1_updating": "fig_updating", "fig2_rate_cumulative": "fig_decline.rate_cumulative",
               "fig3_gas_oil": "fig_phase_coupling.legacy_gor_at_midpoint",
               "fig4_drainage": "fig_modes.drainage_summary", "fig5_correlation": "supp_fig_correlation (alias kept)",
               "fig6_decline": "fig_decline.diagnostics", "fig7_spectrum": "fig_modes.spectrum",
               "fig8_audit": "property_audit.results (alias kept)", "fig9_attributes": "fig_attributes"}


# ---- sampling helpers -------------------------------------------------------------------------------------
class Family:
    """Equal-weight ensemble of fitted forecasters of one kind (the Bayesian model or one comparator)."""

    def __init__(self, name: str, paths: list[str], members: list):
        self.name, self.paths, self.members = name, list(paths), list(members)


def split_counts(n: int, members: int) -> list[int]:
    return [n // members + (1 if i < n % members else 0) for i in range(members)]


def law_of(member):
    """(latent law, native phase scales, device) of a Bayesian or comparator forecaster."""
    if isinstance(member, DepletionBayesForecaster):
        return member.model.rate, member.model.phase_scales.double().cpu().numpy(), torch.device("cpu")
    return member.rate_model.double(), member.volume_scale.double().cpu().numpy(), member.device


def as_days(t) -> torch.Tensor:
    return torch.as_tensor(np.asarray(t, dtype=np.float64), dtype=torch.float64)


class Draws:
    """Pooled posterior (or prior) samples; every accessor returns native units, samples on axis 1."""

    def __init__(self, parts):
        self.parts = parts  # list of (member, z [N, S_i, D] float64 CPU tensor, predict_distribution output)

    def _evaluate(self, what, t=None, parts=None):
        values = []
        for member, z, _ in parts or self.parts:
            law, scale, device = law_of(member)
            zz = z.to(device)
            with torch.no_grad():
                if what == "rate":
                    v = law.rate(zz, t.to(device))
                elif what == "cumulative":
                    v = law.cumulative(zz, t.to(device))
                else:
                    v = law.total_volume(zz)
            values.append(v.detach().cpu().numpy() * scale)
        return np.concatenate(values, axis=1)

    def rate(self, t):
        return self._evaluate("rate", as_days(t))

    def cumulative(self, t):
        return self._evaluate("cumulative", as_days(t))

    def total(self):
        return self._evaluate("total")

    def reports(self):
        return np.concatenate([np.asarray(out["report_cumulative"]) for _, _, out in self.parts], axis=1)

    def unpack(self, key):
        """Bayesian latent-law parameters [N, S, ...] in normalized units (weights: multiply by the oil scale)."""
        return np.concatenate([member.model.rate.unpack(z)[key].detach().cpu().numpy() for member, z, _ in self.parts],
                              axis=1)

    def members(self):
        return [Draws([part]) for part in self.parts]

    @property
    def size(self) -> int:
        return int(sum(z.shape[1] for _, z, _ in self.parts))


def truncate(reports, origin):
    history = np.array(reports, dtype=np.float64, copy=True)
    history[:, DAYS > origin] = np.nan
    return history


def sample(family: Family, reports, x, origin, n, seed) -> Draws:
    """n samples split equally over the family's members; member i draws with seed + i."""
    history = truncate(reports, origin)
    parts = []
    for i, (member, count) in enumerate(zip(family.members, split_counts(n, len(family.members)))):
        if count == 0:
            continue
        out = member.predict_distribution(history, origin, count, np.random.default_rng(seed + i), x)
        parts.append((member, torch.as_tensor(np.asarray(out["latent_parameters"]), dtype=torch.float64), out))
    return Draws(parts)


def draw_record(family: Family, n: int, seed: int, **extra) -> dict:
    counts = split_counts(n, len(family.members))
    return dict(family=family.name, samples=n, members=len(family.members),
                samples_per_member=counts, seeds=[seed + i for i in range(len(family.members))], **extra)


# ---- data helpers (unchanged rules) ---------------------------------------------------------------------
def interval_rates(reports):
    """Observed period-average rates (volume per calendar day) for the report periods DAYS[j-1]..DAYS[j]."""
    return np.diff(reports, axis=-2) / np.diff(DAYS)[:, None]


def pick_wells(reports):
    """Fixed rules on observed reports: A = median 1080-day oil among complete wells; B = the median rise of
    interval GOR from 180-360 to 720-1080 days among wells with oil above the median and a 1.3-2.5x rise
    (a typical rising-GOR well); C = 75th-percentile oil."""
    complete = np.isfinite(reports).all(axis=(1, 2)) & (np.diff(reports, axis=1) > 0).all(axis=(1, 2))
    idx = np.where(complete)[0]
    oil = reports[idx, -1, 0]
    a = idx[np.argsort(oil)[len(idx) // 2]]
    c = idx[np.argsort(oil)[int(0.75 * len(idx))]]
    late = np.diff(reports, axis=1)
    gor_early = late[idx, 4, 1] / late[idx, 4, 0]
    gor_late = late[idx, 6, 1] / late[idx, 6, 0]
    ratio = gor_late / gor_early
    eligible = np.where((oil > np.median(oil)) & (ratio >= 1.3) & (ratio <= 2.5))[0]
    b = idx[eligible[np.argsort(ratio[eligible])[len(eligible) // 2]]]
    return {"A": int(a), "B": int(b), "C": int(c)}


def reports_known(well_reports, origin) -> int:
    return int(np.sum((DAYS[1:] <= origin) & np.isfinite(well_reports[1:, 0])))


def date_title(well_reports, origin) -> str:
    n = reports_known(well_reports, origin)
    if n == 0:
        return f"Day {origin:,} (attributes only)"
    return f"Day {origin:,} ({n} report{'s' if n != 1 else ''})"


def log_axis(ax, axis: str, ticks, limits) -> None:
    if axis == "x":
        ax.set_xscale("log")
        ax.set_xlim(*limits)
        ax.set_xticks(ticks)
    else:
        ax.set_yscale("log")
        ax.set_ylim(*limits)
        ax.set_yticks(ticks)
    plain_ticks(ax, axis)


def pctl(values, q, axis=0):
    return np.percentile(values, q, axis=axis)


def p_notation(values):
    """[P90, P50, P10] in the exceedance convention = the 10th, 50th and 90th percentiles."""
    return np.percentile(values, [10, 50, 90]).tolist()


# ---- Fig. 2: forecasting one well as data arrive (+ Table 2 worked example) ---------------------------
def fig_updating(ctx):
    a = ctx.wells["A"]
    well = ctx.reports[a]
    grid = np.linspace(0.0, 1800.0, 1801)
    origins = (0, 90, 360)
    obs_rate = interval_rates(well)  # [7, 3]
    fig, axes = new_figure(FULL_W, 6.2, 4, 4, gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.0, 0.62]))
    record, card, eurs = {}, {}, {}
    row_max = np.zeros(3)
    cum_max = 0.0
    for j, origin in enumerate(origins):
        n, seed = 400, 11 + origin
        d = sample(ctx.bayes, ctx.reports[a:a + 1], ctx.features[a:a + 1], origin, n, seed)
        r = d.rate(grid)[0]  # [S, G, 3]
        cum = d.cumulative(grid)[0]
        total = d.total()[0]  # [S, 3]
        rep = d.reports()[0]  # [S, 8, 3] reported samples (calibrated report settings)
        eurs[origin] = total
        for k, phase in enumerate(PHASES):
            ax = axes[k, j]
            color = PHASE_COLOR[phase]
            if origin > 0:
                ax.axvspan(0, origin, color=color, alpha=0.06, lw=0, zorder=0)
            q = pctl(r[..., k], [2.5, 10, 50, 90, 97.5])
            ax.fill_between(grid, q[0], q[4], color=color, alpha=0.15, lw=0, rasterized=True)
            ax.fill_between(grid, q[1], q[3], color=color, alpha=0.35, lw=0, rasterized=True)
            ax.plot(grid, q[2], color=color, lw=1.1)
            for p in range(1, len(DAYS)):
                v = obs_rate[p - 1, k]
                if not np.isfinite(v) or v <= 0:
                    continue
                known = DAYS[p] <= origin
                ax.hlines(v, DAYS[p - 1], DAYS[p], color="black" if known else "0.45", lw=1.5 if known else 0.9,
                          linestyles="solid" if known else "dashed", zorder=4)
            row_max[k] = max(row_max[k], float(np.nanmax(q[4])), float(np.nanmax(obs_rate[:, k])))
            if k == 0:
                ax.set_title(date_title(well, origin), loc="right")  # panel letter sits at the left
        ax = axes[3, j]
        q = pctl(cum[..., 0] / VOLUME_DIVISOR, [10, 50, 90])
        ax.fill_between(grid, q[0], q[2], color=PHASE_COLOR["oil"], alpha=0.35, lw=0, rasterized=True)
        ax.plot(grid, q[1], color=PHASE_COLOR["oil"], lw=1.1)
        if origin > 0:
            ax.axvspan(0, origin, color=PHASE_COLOR["oil"], alpha=0.06, lw=0, zorder=0)
        for p in range(1, len(DAYS)):
            v = well[p, 0] / VOLUME_DIVISOR
            if np.isfinite(v):
                known = DAYS[p] <= origin
                ax.plot(DAYS[p], v, ls="none", marker="o", ms=3.2, mec="black", mfc="black" if known else "white",
                        mew=0.8, zorder=4)
        cum_max = max(cum_max, float(np.nanmax(q[2])), float(np.nanmax(well[:, 0])) / VOLUME_DIVISOR)
        # Table 2 (forecast card): reported samples at day 1,080 and the latent oil EUR.
        d_oil = rep[:, -1, 0] - rep[:, -2, 0]
        d_gas = rep[:, -1, 1] - rep[:, -2, 1]
        gor = np.where(d_oil > 0, d_gas / np.where(d_oil > 0, d_oil, 1.0), np.nan)
        observed = well[-1]
        card[str(origin)] = dict(
            reported_cum1080_p90_p50_p10={ph: (np.asarray(p_notation(rep[:, -1, k])) / VOLUME_DIVISOR).tolist()
                                          for k, ph in enumerate(PHASES)},
            period_gor_720_1080_p90_p50_p10_mcf_per_bbl=np.nanpercentile(gor, [10, 50, 90]).tolist(),
            period_gor_undefined_share=float(np.mean(~np.isfinite(gor))),
            latent_oil_eur_p90_p50_p10_mbbl=(np.asarray(p_notation(total[:, 0])) / VOLUME_DIVISOR).tolist(),
            observed_cum1080_percentile_rank={ph: float(100 * np.mean(rep[:, -1, k] < observed[k]))
                                              for k, ph in enumerate(PHASES)},
            reports_known=reports_known(well, origin), **draw_record(ctx.bayes, n, seed))
        record[str(origin)] = dict(
            eur_oil_p10_p90=pctl(total[:, 0], [10, 90]).tolist(), eur_oil_median=float(np.median(total[:, 0])),
            eur_percentiles_10_50_90_native={ph: pctl(total[:, k], [10, 50, 90]).tolist() for k, ph in enumerate(PHASES)},
            eur_p90_p50_p10={ph: (np.asarray(p_notation(total[:, k])) / VOLUME_DIVISOR).tolist() for k, ph in enumerate(PHASES)},
            eur_p90_p10_width={ph: float((pctl(total[:, k], 90) - pctl(total[:, k], 10)) / VOLUME_DIVISOR)
                               for k, ph in enumerate(PHASES)},
            reports_known=reports_known(well, origin), **draw_record(ctx.bayes, n, seed))
    d_oil, d_gas = well[-1, 0] - well[-2, 0], well[-1, 1] - well[-2, 1]
    card["observed"] = dict(cum1080={ph: float(well[-1, k] / VOLUME_DIVISOR) for k, ph in enumerate(PHASES)},
                            period_gor_720_1080_mcf_per_bbl=float(d_gas / d_oil) if d_oil > 0 else None)
    card["units"] = dict(volumes="Mbbl (oil, water), MMcf (gas)", gor="Mcf/bbl",
                         p_notation="[P90, P50, P10] = 10th, 50th, 90th percentiles")
    # axes, limits, labels
    for k, phase in enumerate(PHASES):
        for j in range(3):
            ax = axes[k, j]
            ax.set_yscale("log")
            ax.set_ylim(RATE_FLOOR, row_max[k] * 1.6)
            plain_ticks(ax, "y")
            ax.set_xlim(0, 1800)
            ax.set_xticks([0, 360, 720, 1080, 1800])
            ax.tick_params(labelbottom=False)
            if j > 0:
                ax.tick_params(labelleft=False)
            shade_beyond(ax, LAST_REPORT_DAY)  # the legend panel explains the shading
        axes[k, 0].set_ylabel(f"{PHASE_LABEL[phase]} rate, {RATE_UNIT[phase]}")
    for j in range(3):
        ax = axes[3, j]
        ax.set_xlim(0, 1800)
        ax.set_ylim(0, cum_max * 1.08)
        ax.set_xticks([0, 360, 720, 1080, 1800])
        plain_ticks(ax)
        ax.set_xlabel("Time, days")
        if j > 0:
            ax.tick_params(labelleft=False)
        shade_beyond(ax, LAST_REPORT_DAY)
    axes[3, 0].set_ylabel("Cumulative oil, Mbbl")
    # EUR column
    for k, phase in enumerate(PHASES):
        ax = axes[k, 3]
        color = PHASE_COLOR[phase]
        for j, origin in enumerate(origins):
            q = pctl(eurs[origin][:, k] / VOLUME_DIVISOR, [2.5, 10, 50, 90, 97.5])
            ax.vlines(j, q[0], q[4], color=color, lw=0.8, alpha=0.6)
            ax.vlines(j, q[1], q[3], color=color, lw=3.5, alpha=0.6)
            ax.plot(j, q[2], ls="none", marker="o", ms=4, mfc="white", mec=color, mew=1.0, zorder=3)
        ax.set_xlim(-0.5, 2.5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([f"{o:,}" for o in origins])
        ax.set_ylim(bottom=0)
        plain_ticks(ax, "y")
        ax.set_ylabel(f"{PHASE_LABEL[phase]} EUR, {VOLUME_UNIT[phase]}")
        if k == 0:
            ax.set_title("EUR")
    axes[2, 3].set_xlabel("Forecast date, days")
    leg = axes[3, 3]
    leg.axis("off")
    leg.legend(handles=legend_handles([
        ("line", "P50", dict(color="0.25", lw=1.1)),
        ("patch", "P90–P10", dict(color="0.25", alpha=0.35, lw=0)),
        ("patch", "P97.5–P2.5", dict(color="0.25", alpha=0.15, lw=0)),
        ("line", "report known at\nforecast date", dict(color="black", lw=1.5)),
        ("line", "later report", dict(color="0.45", lw=0.9, ls="--")),
        ("patch", "beyond last\nreport", dict(color="0.93", lw=0))]),
        loc="center left", bbox_to_anchor=(-0.05, 0.5), fontsize=NOTE_PT, handlelength=1.4)
    label_panels([axes[k, j] for k in range(4) for j in range(4) if not (k == 3 and j == 3)])
    files = save(fig, ctx.output, "fig_updating")
    record.update(files=files, well="A", tau_max_days=ctx.tau_max, rate_floor=RATE_FLOOR,
                  note="latent rates and cumulative oil; reports drawn as period averages over each report period")
    ctx.record["worked_example"] = dict(card, well="A", provenance=ctx.provenance)
    return record


# ---- Fig. 3: decline in petroleum terms ---------------------------------------------------------------
def _rate_cum_panel(ax, curves, obs, eur_ticks, thin, clip_ticks=False):
    """curves: list of (q [S, T], Q [S, T] in bbl, eur [S], color, label); obs: (mid-cumulative Mbbl, rate)."""
    out = {}
    for q, Q, eur, color, label, i1080, i3650 in curves:
        for s in thin(len(q)):
            ax.plot(Q[s] / VOLUME_DIVISOR, q[s], color=color, lw=0.4, alpha=0.25, rasterized=True)
        m = int(np.argmin(np.abs(eur - np.median(eur))))
        ax.plot(Q[m] / VOLUME_DIVISOR, q[m], color=color, lw=1.5, label=label, zorder=4)
        ax.plot(Q[m, i1080] / VOLUME_DIVISOR, q[m, i1080], ls="none", marker="o", ms=4.5, mfc="white", mec=color,
                mew=1.0, zorder=5)
        ax.plot(Q[m, i3650] / VOLUME_DIVISOR, q[m, i3650], ls="none", marker="^", ms=4.5, color=color, zorder=5)
        out[label] = dict(sample=m, q1080=float(q[m, i1080]), Q1080=float(Q[m, i1080]), q3650=float(q[m, i3650]),
                          Q3650=float(Q[m, i3650]))
    ax.plot(obs[0], obs[1], ls="none", marker="s", ms=3.0, color="black", zorder=6)
    ax.plot(eur_ticks, np.zeros(3), ls="none", marker="|", ms=9, mew=1.2, color="black", clip_on=clip_ticks, zorder=6)
    return out


def fig_decline(ctx):
    a = ctx.wells["A"]
    origin = 360
    well = ctx.reports[a]
    long = np.unique(np.concatenate((np.linspace(0.0, 3650.0, 3651), np.logspace(3.6, 5.0, 200))))
    i1080, i3650 = int(np.searchsorted(long, 1080.0)), int(np.searchsorted(long, 3650.0))
    db = sample(ctx.bayes, ctx.reports[a:a + 1], ctx.features[a:a + 1], origin, 60, 21)
    dc = sample(ctx.basis, ctx.reports[a:a + 1], ctx.features[a:a + 1], origin, 60, 21)
    qb, Qb, eb = db.rate(long)[0][..., 0], db.cumulative(long)[0][..., 0], db.total()[0][:, 0]
    qc, Qc, ec = dc.rate(long)[0][..., 0], dc.cumulative(long)[0][..., 0], dc.total()[0][:, 0]
    rates = interval_rates(well)[:, 0]
    mids = 0.5 * (well[1:, 0] + well[:-1, 0]) / VOLUME_DIVISOR
    ok = np.isfinite(rates) & np.isfinite(mids) & (rates > 0)
    obs = (mids[ok], rates[ok])
    eur_ticks = np.asarray(p_notation(eb)) / VOLUME_DIVISOR  # P90, P50, P10
    model_label = MODEL_LABEL.get(ctx.model_label, ctx.model_label)
    curves = [(qb, Qb, eb, "black", model_label, i1080, i3650), (qc, Qc, ec, BASIS_COLOR, MODEL_LABEL["basis"], i1080, i3650)]
    thin = lambda n: np.unique(np.linspace(0, n - 1, min(40, n)).astype(int))
    last_cum = well[-1, 0] / VOLUME_DIVISOR

    fig, axes = new_figure(FULL_W, 4.4, 2, 2)
    # (a) rate-cumulative
    ax = axes[0, 0]
    picked = _rate_cum_panel(ax, curves, obs, eur_ticks, thin)
    ymax = 1.05 * max(np.percentile(qb.max(1), 97.5), np.percentile(qc.max(1), 97.5), float(np.max(obs[1])) if len(obs[1]) else 0)
    ax.set_xlim(0, eur_ticks[2] * 1.03)
    ax.set_ylim(0, ymax)
    plain_ticks(ax)
    ax.set_xlabel("Cumulative oil, Mbbl")
    ax.set_ylabel("Oil rate, bbl/D")
    if np.isfinite(last_cum):
        shade_beyond(ax, last_cum)
    ax.legend(handles=legend_handles([
        ("line", model_label, dict(color="black", lw=1.5)),
        ("line", MODEL_LABEL["basis"], dict(color=BASIS_COLOR, lw=1.5)),
        ("marker", "day 1,080 (last report)", dict(marker="o", ms=4.5, mfc="white", mec="0.2")),
        ("marker", "day 3,650 (10 years)", dict(marker="^", ms=4.5, color="0.2")),
        ("marker", "reported period average", dict(marker="s", ms=3.0, color="black")),
        ("marker", "model EUR P90, P50, P10", dict(marker="|", ms=8, mew=1.2, color="black")),
        ("patch", "beyond last report", dict(facecolor="0.93", edgecolor="0.6", lw=0.4))]),
        loc="upper right", fontsize=NOTE_PT)
    # (b) zoom on the tail
    ax = axes[0, 1]
    _rate_cum_panel(ax, curves, obs, eur_ticks, thin, clip_ticks=True)
    pb, pc = picked[model_label], picked[MODEL_LABEL["basis"]]
    x0 = 0.9 * min(pb["Q1080"], pc["Q1080"]) / VOLUME_DIVISOR
    ytop = 1.15 * max(pb["q1080"], pc["q1080"])
    x_end = (pb["Q3650"] + pb["q3650"] * ctx.tau_max) / VOLUME_DIVISOR
    ax.plot([pb["Q3650"] / VOLUME_DIVISOR, x_end], [pb["q3650"], 0.0], color="0.35", lw=0.9, ls=(0, (4, 2)), zorder=5)
    ax.annotate(r"terminal decline, slope $-1/\tau_{\max}$", xy=(0.5 * (pb["Q3650"] / VOLUME_DIVISOR + x_end), 0.5 * pb["q3650"]),
                xytext=(0.985, 0.33), textcoords="axes fraction", ha="right", va="bottom", fontsize=NOTE_PT, color="0.15",
                arrowprops=dict(arrowstyle="-", color="0.35", lw=0.6, shrinkB=1),
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=6)
    for tick, text in zip(eur_ticks, ("P90", "P50", "P10")):
        ax.text(tick, 0.07 * ytop, text, ha="center", va="bottom", fontsize=NOTE_PT, color="0.2", clip_on=True)
    ax.set_xlim(x0, max(eur_ticks[2], x_end) * 1.03)
    ax.set_ylim(0, ytop)
    plain_ticks(ax)
    ax.set_xlabel("Cumulative oil, Mbbl")
    ax.set_ylabel("Oil rate, bbl/D")
    if np.isfinite(last_cum):
        shade_beyond(ax, last_cum)
    # (c) nominal decline D(t) and (d) Arps b(t), 200 samples
    daily = np.arange(180.0, 7301.0, 1.0)
    window = daily <= 3650.0
    db2 = sample(ctx.bayes, ctx.reports[a:a + 1], ctx.features[a:a + 1], origin, 200, 81)
    dc2 = sample(ctx.basis, ctx.reports[a:a + 1], ctx.features[a:a + 1], origin, 200, 81)
    term = terminal_decline(ctx.tau_max)
    diag, bands = {}, {}
    for name, d in (("Bayes", db2), ("basis", dc2)):
        q = d.rate(daily)[0][..., 0]
        D = -np.gradient(np.log(np.maximum(q, 1e-300)), axis=1)  # 1/day on the daily grid
        bfac = np.gradient(1.0 / np.maximum(D, 1e-12), axis=1)
        diag[name] = dict(share_with_increasing_decline=float(np.mean((np.diff(D[:, window], axis=1) > 1e-10).any(1))),
                          share_with_negative_b=float(np.mean((bfac[:, window] < -1e-6).any(1))))
        b_plot = np.where(D > 0, bfac, np.nan)
        bands[name] = dict(D=np.nanpercentile(D * DAYS_PER_YEAR, [10, 50, 90], axis=0),
                           b=np.nanpercentile(b_plot, [10, 50, 90], axis=0))
    ax = axes[1, 0]
    lo = 0.5 * term["nominal_per_year"]
    hi = 1.3 * max(np.nanmax(bands["Bayes"]["D"][2]), np.nanmax(np.where(np.isfinite(bands["basis"]["D"][2]), bands["basis"]["D"][2], 0)))
    for name, color, alpha in (("basis", BASIS_COLOR, 0.25), ("Bayes", "black", 0.18)):
        band = np.clip(bands[name]["D"], 0.5 * lo, None)
        ax.fill_between(daily, band[0], band[2], color=color, alpha=alpha, lw=0, rasterized=True)
        ax.plot(daily, band[1], color=color, lw=1.2, label=model_label if name == "Bayes" else MODEL_LABEL["basis"])
    ax.axhline(term["nominal_per_year"], color="0.35", lw=0.8, ls=(0, (4, 2)))
    ax.text(200, term["nominal_per_year"] * 1.08,
            rf"$1/\tau_{{\max}}$ (like $D_{{\min}}$), {term['effective_pct_per_year']:.1f}%/yr effective",
            ha="left", va="bottom", fontsize=NOTE_PT, color="0.25")
    log_axis(ax, "x", [180, 365, 1080, 3650, 7300], (180, 7300))
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    plain_ticks(ax, "y")
    ax.set_xlabel("Time, days")
    ax.set_ylabel("Nominal decline D(t), 1/yr")
    sec = ax.secondary_yaxis("right", functions=(effective_decline_pct, nominal_from_effective))
    sec.set_ylabel("Effective decline, %/yr")
    sec.set_yticks([5, 10, 20, 50, 90])
    sec.yaxis.set_major_formatter(PLAIN)
    sec.yaxis.set_minor_formatter(ticker.NullFormatter())
    shade_beyond(ax, LAST_REPORT_DAY)
    ax.legend(loc="lower left", fontsize=NOTE_PT)
    ax = axes[1, 1]
    for level in (0.0, 1.0, 2.0):
        ax.axhline(level, color="0.55", lw=0.6, ls=":", zorder=1)
    bb = bands["Bayes"]["b"]
    ax.fill_between(daily, bb[0], bb[2], color="black", alpha=0.18, lw=0, rasterized=True)
    ax.plot(daily, bb[1], color="black", lw=1.2, label=model_label)
    ax.plot(daily, bands["basis"]["b"][1], color=BASIS_COLOR, lw=1.0, label=f"{MODEL_LABEL['basis']} (P50)")
    log_axis(ax, "x", [180, 365, 1080, 3650, 7300], (180, 7300))
    ax.set_ylim(-0.5, 3.0)
    plain_ticks(ax, "y")
    peak = int(np.nanargmax(bb[1]))
    ax.annotate(f"P50 peak\nnear day {daily[peak]:,.0f}", xy=(daily[peak], min(bb[1][peak], 2.9)), xytext=(0.04, 0.96),
                textcoords="axes fraction", ha="left", va="top", fontsize=NOTE_PT, color="0.25",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="0.35"))
    ax.set_xlabel("Time, days")
    ax.set_ylabel(r"Arps $b(t) = d(1/D)/dt$")
    shade_beyond(ax, LAST_REPORT_DAY)  # colors as in (c); the network's line is its P50
    label_panels([axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]])
    files = save(fig, ctx.output, "fig_decline")
    at = {day: int(np.argmin(np.abs(daily - day))) for day in (365.0, 1080.0, 3650.0, 7300.0)}
    return dict(
        files=files, well="A", origin=origin,
        rate_cumulative=dict(origin=origin, eur_oil_p10_p50_p90=pctl(eb, [10, 50, 90]).tolist(),
                             basis_eur_oil_p10_p50_p90=pctl(ec, [10, 50, 90]).tolist(),
                             eur_over_cum3650_median=dict(Bayes=float(np.median(eb / Qb[:, i3650])),
                                                          basis=float(np.median(ec / Qc[:, i3650]))),
                             thick_line_samples=picked, **draw_record(ctx.bayes, 60, 21)),
        zoom=dict(x_mbbl=[x0, max(eur_ticks[2], x_end) * 1.03], y_bbl_per_day=[0.0, ytop]),
        diagnostics=diag, diagnostics_window_days=[180, 3650],
        decline_p50_per_year={name: {str(int(day)): float(bands[name]["D"][1][i]) for day, i in at.items()} for name in bands},
        b_p50_peak=dict(day=float(daily[peak]), value=float(bb[1][peak])),
        terminal_decline=term, decline_samples=draw_record(ctx.bayes, 200, 81),
        note="D = -d ln q/dt and b = d(1/D)/dt by finite differences on a daily grid (180-7,300 d)")


# ---- Fig. 4: decay modes and EUR timing ---------------------------------------------------------------
def fig_modes(ctx):
    origin = 360
    tau = np.asarray(ctx.decay_times, dtype=float)
    lam = 1.0 / tau
    fast = tau < FAST_TAU_DAYS
    fig, axes = new_figure(FULL_W, 2.6, 1, 3)
    # (a) decay-time spectrum of wells A, B and C
    ax = axes[0, 0]
    styles = {"A": dict(marker="o", color="black"), "B": dict(marker="s", color="#E69F00"),
              "C": dict(marker="^", color="#56B4E9")}
    spectrum, draws_a = {}, None
    for offset, (label, idx) in zip((-0.04, 0.0, 0.04), ctx.wells.items()):
        d = sample(ctx.bayes, ctx.reports[idx:idx + 1], ctx.features[idx:idx + 1], origin, 400, 91)
        w = d.unpack("weights")[0]
        share = w / w.sum(1, keepdims=True)
        q = pctl(share, [25, 50, 75])
        ax.errorbar(np.log10(tau) + offset, 100 * q[1], yerr=[100 * (q[1] - q[0]), 100 * (q[2] - q[1])], fmt=styles[label]["marker"],
                    color=styles[label]["color"], ms=3.5, elinewidth=0.8, capsize=0, label=f"Well {label}")
        spectrum[label] = dict(median=q[1].round(4).tolist(), p25=q[0].round(4).tolist(), p75=q[2].round(4).tolist(),
                               fast_mode_share_median=float(np.median(w[:, fast].sum(1) / w.sum(1))))
        if label == "A":
            draws_a = d
    ax.axvline(np.log10(tau[-1]), color="0.35", lw=0.8, ls=(0, (4, 2)))
    ax.text(np.log10(tau[-1]) - 0.06, 0.45, rf"$\tau_{{\max}}$ = {tau[-1]:,.0f} d" + "\n(grid limit)", ha="right", va="center",
            fontsize=NOTE_PT, color="0.25")
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.6)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 1, 10, 100])
    plain_ticks(ax, "y")
    ax.set_xlim(np.log10(tau[0]) - 0.15, np.log10(tau[-1]) + 0.15)
    ax.set_xticks(np.log10(tau))
    ax.set_xticklabels([f"{v:,.0f}" if i in (0, 2, 4, 7) else "" for i, v in enumerate(tau)])
    ax.set_xlabel(r"Decay time $\tau_k$, days")
    ax.set_ylabel("Share of post-ramp oil, %")
    ax.legend(loc="upper left", fontsize=NOTE_PT)
    # (b) one sample of well A split into its modes
    ax = axes[0, 1]
    total = draws_a.total()[0][:, 0]
    pick = int(np.argmin(np.abs(total - np.median(total))))
    t = np.unique(np.concatenate((np.geomspace(1.0, 50 * DAYS_PER_YEAR, 600), [LAST_REPORT_DAY, 3650.0])))
    r = draws_a.rate(t)[0][pick, :, 0]
    weights = draws_a.unpack("weights")[0][pick] * ctx.scale[0]  # bbl remaining in each mode at the end of the ramp
    peak = float(draws_a.unpack("peak")[0][pick])
    after = np.clip(t - peak, 0.0, None)
    contrib = weights[None, :] * lam[None, :] * np.exp(-lam[None, :] * after[:, None])
    contrib[t < peak] = np.nan
    for k in range(len(tau)):
        slowest = k == len(tau) - 1
        ax.plot(t, contrib[:, k], color="black" if slowest else str(0.8 - 0.5 * k / (len(tau) - 1)),
                lw=1.0 if slowest else 0.6, ls=(0, (4, 2)) if slowest else "-", zorder=3)
    ax.plot(t, r, color=PHASE_COLOR["oil"], lw=1.4, zorder=4)
    ax.axvline(peak, color="0.5", lw=0.6, ls=":")
    log_axis(ax, "x", [1, 10, 100, 1000, 10000], (1.0, 50 * DAYS_PER_YEAR))
    ax.set_yscale("log")
    ax.set_ylim(0.1, 60.0 * float(np.nanmax(r)))  # headroom above the curves for the legend
    plain_ticks(ax, "y")
    ax.set_xlabel("Time, days")
    ax.set_ylabel("Oil rate, bbl/D")
    shade_beyond(ax, LAST_REPORT_DAY)
    entries = [("line", "total", dict(color=PHASE_COLOR["oil"], lw=1.4)),
               ("line", "slowest mode", dict(color="black", lw=1.0, ls=(0, (4, 2)))),
               ("line", "faster modes", dict(color="0.6", lw=0.6))]
    if peak >= 1.0:  # the time axis starts at day 1
        entries.append(("line", "end of ramp", dict(color="0.5", lw=0.6, ls=":")))
    entries.append(("patch", "beyond last report", dict(facecolor="0.93", edgecolor="0.6", lw=0.4)))
    ax.legend(handles=legend_handles(entries), loc="upper right", fontsize=NOTE_PT)
    # (c) share of EUR produced against time, by phase (well A)
    ax = axes[0, 2]
    tc = np.unique(np.concatenate((np.geomspace(30.0, 50 * DAYS_PER_YEAR, 300), [LAST_REPORT_DAY, 3650.0])))
    cum = draws_a.cumulative(tc)[0]
    tot = draws_a.total()[0]
    frac = 100 * cum / tot[:, None, :]
    years = tc / DAYS_PER_YEAR
    timing = {}
    for k, phase in enumerate(PHASES):
        band = pctl(frac[..., k], [10, 50, 90])
        ax.fill_between(years, band[0], band[2], color=PHASE_COLOR[phase], alpha=0.15, lw=0, rasterized=True)
        ax.plot(years, band[1], color=PHASE_COLOR[phase], ls=PHASE_LS[phase], lw=1.2, label=PHASE_LABEL[phase])
        timing[phase] = {str(int(day)): dict(p50=float(band[1][np.searchsorted(tc, day)]),
                                             p10_p90=[float(band[0][np.searchsorted(tc, day)]), float(band[2][np.searchsorted(tc, day)])])
                         for day in (LAST_REPORT_DAY, 3650.0)}
    ax.axvline(10.0, color="0.35", lw=0.7, ls=":")
    ax.text(10.0 * 1.08, 4, "10 years", ha="left", va="bottom", fontsize=NOTE_PT, color="0.25")
    log_axis(ax, "x", [0.1, 0.5, 1, 3, 10, 50], (30.0 / DAYS_PER_YEAR, 50.0))
    ax.set_ylim(0, 100)
    plain_ticks(ax, "y")
    shade_beyond(ax, LAST_REPORT_DAY / DAYS_PER_YEAR)
    ax.set_xlabel("Time, years")
    ax.set_ylabel("Share of EUR produced, %")
    ax.legend(loc="upper left", fontsize=NOTE_PT)
    label_panels([axes[0, 0], axes[0, 1], axes[0, 2]])
    files = save(fig, ctx.output, "fig_modes")
    # Legacy drainage numbers (old fig4, 40 samples, seed 51) kept for the sentence that replaces the portrait.
    d51 = sample(ctx.bayes, ctx.reports[ctx.wells["A"]:ctx.wells["A"] + 1], ctx.features[ctx.wells["A"]:ctx.wells["A"] + 1],
                 origin, 40, 51)
    w = d51.unpack("weights")[0] * ctx.scale[0]
    drainage = dict(fast_rate_per_day_median=float(np.median((w[:, fast] * lam[fast]).sum(1) / w[:, fast].sum(1))),
                    slow_rate_per_day_median=float(np.median((w[:, ~fast] * lam[~fast]).sum(1) / w[:, ~fast].sum(1))),
                    fast_share_of_decline_volume_median=float(np.median(w[:, fast].sum(1) / w.sum(1))),
                    fast_tau_days=FAST_TAU_DAYS, **draw_record(ctx.bayes, 40, 51))
    return dict(files=files, origin=origin, decay_times_days=tau.tolist(),
                spectrum={k: v["median"] for k, v in spectrum.items()}, spectrum_detail=spectrum,
                mode_example=dict(well="A", sample=pick, eur_oil_bbl=float(total[pick]), peak_day=peak,
                                  mode_volume_share=(weights / weights.sum()).round(4).tolist()),
                eur_timing_share_pct=timing, drainage_summary=drainage, **draw_record(ctx.bayes, 400, 91))


# ---- Fig. 6: gas and water follow depletion -----------------------------------------------------------
def ratio_band(cum_oil, ratio, oil_axis, min_defined=0.8):
    """Percentiles of a ratio on a cumulative-oil grid; a sample is undefined beyond its own
    20,000-day cumulative, and percentiles are kept only where at least `min_defined` of samples exist."""
    values = np.full((len(cum_oil), len(oil_axis)), np.nan)
    ramp = 1e-9 * np.arange(cum_oil.shape[1])
    for s in range(len(cum_oil)):
        xp = cum_oil[s] + ramp * max(1.0, float(cum_oil[s, -1]))  # strictly increasing for np.interp
        values[s] = np.interp(oil_axis, xp, ratio[s], left=np.nan, right=np.nan)
    defined = np.isfinite(values).mean(0) >= min_defined
    band = np.full((3, len(oil_axis)), np.nan)
    if defined.any():
        band[:, defined] = np.nanpercentile(values[:, defined], [10, 50, 90], axis=0)
    return band, defined


def spearman(x, y) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def fig_phase_coupling(ctx):
    b = ctx.wells["B"]
    well = ctx.reports[b]
    dense = np.linspace(0.0, 20000.0, 4001)
    store, legacy = {}, {}
    for origin in (0, 90, 360):
        d = sample(ctx.bayes, ctx.reports[b:b + 1], ctx.features[b:b + 1], origin, 300, 31 + origin)
        r = d.rate(dense)[0]
        cum = d.cumulative(dense)[0]
        gor = r[..., 1] / r[..., 0]
        store[origin] = dict(gor=gor, wor=r[..., 2] / r[..., 0], cum_oil=cum[..., 0], total=d.total()[0],
                             reports=d.reports()[0])
        # Legacy number of the old fig3 (holds a sample's final GOR beyond its EUR): kept for traceability only.
        axis_old = np.linspace(0, np.percentile(cum[:, -1, 0], 90), 120)
        interp_old = np.stack([np.interp(axis_old, cum[s, 1:, 0], gor[s, 1:]) for s in range(len(gor))])
        legacy[str(origin)] = dict(gor_at_median_eur_p10_p50_p90=np.percentile(interp_old[:, len(axis_old) // 2], [10, 50, 90]).tolist())
    xmax = float(np.median(store[360]["total"][:, 0]))
    oil_axis = np.linspace(0.0, xmax, 200)
    x_mbbl = oil_axis / VOLUME_DIVISOR
    last_cum = well[-1, 0] / VOLUME_DIVISOR
    d_oil = np.diff(well[:, 0])
    periods = [(p, well[p - 1, 0] / VOLUME_DIVISOR, well[p, 0] / VOLUME_DIVISOR) for p in range(1, len(DAYS))
               if np.isfinite(d_oil[p - 1]) and d_oil[p - 1] > 0]
    fig, axes = new_figure(FULL_W, 2.6, 1, 3)
    summary = {}
    for col, (key, phase_k, ylabel) in enumerate((("gor", 1, "GOR, Mcf/bbl"), ("wor", 2, "WOR, bbl/bbl"))):
        ax = axes[0, col]
        summary[key] = {}
        tops = []
        for origin in (0, 90, 360):
            band, defined = ratio_band(store[origin]["cum_oil"], store[origin][key], oil_axis)
            label = date_title(well, origin)
            if origin == 0:
                ax.fill_between(x_mbbl, band[0], band[2], color="0.85", lw=0, rasterized=True)
                for edge in (band[0], band[2]):  # outline: the band lies on the gray shading beyond the last report
                    ax.plot(x_mbbl, edge, color="0.55", lw=0.4)
                ax.plot(x_mbbl, band[1], color="0.5", lw=1.0, label=label)
            elif origin == 360:
                ax.fill_between(x_mbbl, band[0], band[2], color=PHASE_COLOR["gas"], alpha=0.3, lw=0, rasterized=True)
                ax.plot(x_mbbl, band[1], color=PHASE_COLOR["gas"], lw=1.3, label=label)
            else:
                ax.plot(x_mbbl, band[1], color=PHASE_COLOR["gas"], lw=0.9, ls=(0, (4, 2)), label=label)
            late = x_mbbl >= 0.1 * x_mbbl[-1]
            if origin != 90 and np.isfinite(band[2][late]).any():
                tops.append(float(np.nanmax(band[2][late])))
            fractions = (0.25, 0.5, 0.75, 1.0)
            summary[key][str(origin)] = dict(
                defined_to_mbbl=float(x_mbbl[defined][-1]) if defined.any() else None,
                p10_p50_p90_at_fraction_of_x={str(f): band[:, min(len(oil_axis) - 1, int(round(f * (len(oil_axis) - 1))))].tolist()
                                             for f in fractions})
        reported = []
        for p, x0, x1 in periods:
            v = (well[p, phase_k] - well[p - 1, phase_k]) / (well[p, 0] - well[p - 1, 0])
            if np.isfinite(v):
                ax.hlines(v, x0, x1, color="black", lw=1.5, zorder=5)
                reported.append(float(v))
        tops += reported
        ax.axvline(last_cum, color="0.35", lw=0.7, ls=":")
        ax.set_xlim(0, x_mbbl[-1])
        ax.set_ylim(0, 1.1 * max(tops) if tops else None)
        plain_ticks(ax)
        ax.set_xlabel("Cumulative oil, Mbbl")
        ax.set_ylabel(ylabel)
        shade_beyond(ax, last_cum)
        summary[key]["reported_periods"] = reported
    handles, labels = axes[0, 0].get_legend_handles_labels()
    extra = legend_handles([("line", "reported period ratio", dict(color="black", lw=1.5)),
                            ("patch", "beyond last report", dict(facecolor="0.93", edgecolor="0.6", lw=0.4))])
    figure_legend(fig, handles + extra, labels + [h.get_label() for h in extra], ncol=5, fallback_ax=axes[0, 0],
                  fallback_loc="upper left")
    # (c) joint three-year oil and gas from reported samples at forecast dates 90 and 360
    ax = axes[0, 2]
    obs_o, obs_g = well[-1, 0] / VOLUME_DIVISOR, well[-1, 1] / VOLUME_DIVISOR
    joint, tops_c = {}, [obs_g]
    looks = {90: (dict(facecolors="none", edgecolors="0.45", linewidths=0.4, alpha=0.7), dict(color="0.35", lw=0.9, ls=(0, (4, 2)))),
             360: (dict(color=PHASE_COLOR["gas"], alpha=0.45, linewidths=0), dict(color="black", lw=1.3))}
    for origin, (points, cross) in looks.items():
        rep = store[origin]["reports"]
        xo, yg = rep[:, -1, 0] / VOLUME_DIVISOR, rep[:, -1, 1] / VOLUME_DIVISOR
        ax.scatter(xo, yg, s=5, rasterized=True, zorder=2 if origin == 90 else 3, **points)
        px, py = p_notation(xo), p_notation(yg)
        # marginal P90-P10 bars crossing at the two P50 values
        ax.plot([px[0], px[2]], [py[1], py[1]], zorder=4, **cross)
        ax.plot([px[1], px[1]], [py[0], py[2]], zorder=4, **cross)
        tops_c.append(float(np.percentile(yg, 99.5)))
        joint[str(origin)] = dict(origin=origin, spearman_oil_gas=spearman(xo, yg), samples=int(len(xo)),
                                  oil_p90_p50_p10_mbbl=px, gas_p90_p50_p10_mmcf=py,
                                  observed_oil_mbbl=float(obs_o), observed_gas_mmcf=float(obs_g),
                                  observed_rank_pct=dict(oil=float(100 * np.mean(xo < obs_o)), gas=float(100 * np.mean(yg < obs_g))))
    ax.plot(obs_o, obs_g, ls="none", marker="*", ms=9, color="black", mec="white", mew=0.6, zorder=6)
    ax.annotate("reported", xy=(obs_o, obs_g), xytext=(7, 0), textcoords="offset points", ha="left", va="center",
                fontsize=NOTE_PT, color="black", zorder=7, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
    plain_ticks(ax)
    low, high = ax.get_ylim()
    ax.set_ylim(low, low + 1.32 * (max(tops_c) - low))  # headroom above the samples for the legend
    ax.set_xlabel("Oil to day 1,080, Mbbl")
    ax.set_ylabel("Gas to day 1,080, MMcf")
    ax.legend(handles=legend_handles([
        ("marker", date_title(well, 90), dict(marker="o", ms=3, mfc="none", mec="0.45", mew=0.5)),
        ("marker", date_title(well, 360), dict(marker="o", ms=3, color=PHASE_COLOR["gas"], alpha=0.7, mew=0)),
        ("line", "P90 to P10, crossing at P50", dict(color="black", lw=1.3))]), loc="upper left", fontsize=NOTE_PT)
    label_panels([axes[0, 0], axes[0, 1], axes[0, 2]])
    files = save(fig, ctx.output, "fig_phase_coupling")
    total360 = store[360]["total"]
    return dict(files=files, well="B", x_range_mbbl=[0.0, float(x_mbbl[-1])], last_report_cum_oil_mbbl=float(last_cum),
                ratio_summary=summary, min_defined_share=0.8,
                joint_1080=joint["90"], joint_1080_by_origin=joint,
                eur_over_reported_cum1080_median={ph: float(np.median(total360[:, k] / well[-1, k])) for k, ph in enumerate(PHASES)},
                legacy_gor_at_midpoint=legacy, samples_per_origin=300, seeds_by_origin={str(o): 31 + o for o in (0, 90, 360)},
                members=len(ctx.bayes.members))


# ---- Fig. 7: attribute effects on the prior -----------------------------------------------------------
def fig_attributes(ctx):
    from joint_production.data import load_dataset
    from joint_production.static_features import STATIC_FEATURE_NAMES, StaticFeaturePreprocessor
    dataset = load_dataset(ROOT / "data/processed/bakken_v1")
    train = [r for r in dataset["records"] if r["split"] == "train"]
    proc = StaticFeaturePreprocessor.fit(train)
    del dataset, train
    fit_x = ctx.data["fit_features"]
    fit_oil = ctx.data["fit_reports"][:, -1, 0]
    # The model's attribute-only (forecast date 0) P50 of oil to day 1,080 for every training well, with its
    # own attributes: comparable with the observed bin medians, unlike the one-attribute-at-a-time curves.
    fit_pred = np.full(len(fit_x), np.nan)
    blank = np.full((1, len(DAYS), 3), np.nan)
    for start in range(0, len(fit_x), 512):
        part = fit_x[start:start + 512]
        d_fit = sample(ctx.bayes, np.repeat(blank, len(part), axis=0), part, 0, 50, 303)
        fit_pred[start:start + 512] = np.median(d_fit.cumulative(np.array([1.0, LAST_REPORT_DAY]))[:, :, 1, 0], axis=1)
    k = len(STATIC_FEATURE_NAMES)
    # Partial dependence: PD_WELLS training wells keep their own other attributes and missing flags while one
    # attribute is set to each grid value; the samples of all wells are pooled.
    pd_idx = np.sort(np.random.default_rng(2026).choice(len(fit_x), size=min(PD_WELLS, len(fit_x)), replace=False))
    pd_x = fit_x[pd_idx]
    pd_blank = np.repeat(blank, len(pd_x), axis=0)
    chosen = ["LateralLength_FT", "ProppantIntensity_LBSPerFT", "FluidIntensity_BBLPerFT", "TrueVerticalDepth_FT"]
    times = np.array([1.0, LAST_REPORT_DAY])
    rows = (("cum1080", "Oil to day 1,080,\nMbbl", "oil"), ("eur", "Oil EUR,\nMbbl", "oil"),
            ("gor0", "Initial GOR,\nMcf/bbl", "gas"))
    fig, axes = new_figure(FULL_W, 4.6, 3, 4, sharex="col", sharey="row")
    record = {}
    for col, feat in enumerate(chosen):
        j = STATIC_FEATURE_NAMES.index(feat)
        grid_x = np.linspace(*np.percentile(fit_x[:, j], [5, 95]), 17)  # even indices are the earlier 9-point grid
        raw = proc.center[j] + proc.scale[j] * grid_x
        values = {key: [] for key, _, _ in rows}
        for g in grid_x:
            x = pd_x.copy()
            x[:, j] = g
            x[:, k + j] = 1.0  # the varied attribute is present
            d = sample(ctx.bayes, pd_blank, x, 0, PD_SAMPLES, 101)  # fresh generators per grid value: common random numbers
            cum = d.cumulative(times)
            rate = d.rate(times)
            values["cum1080"].append((cum[:, :, 1, 0] / VOLUME_DIVISOR).ravel())
            values["eur"].append((d.total()[:, :, 0] / VOLUME_DIVISOR).ravel())
            values["gor0"].append((rate[:, :, 0, 1] / rate[:, :, 0, 0]).ravel())
        present = fit_x[:, k + j] > 0.5
        raw_train = proc.center[j] + proc.scale[j] * fit_x[:, j]
        inside = present & (raw_train >= raw[0]) & (raw_train <= raw[-1])
        rec = dict(raw_grid=raw.tolist(), legacy_grid_index=list(range(0, 17, 2)))
        for row, (key, ylabel, phase) in enumerate(rows):
            ax = axes[row, col]
            v = np.asarray(values[key])  # [grid, samples]
            band = pctl(v, [25, 50, 75], axis=1)
            ax.fill_between(raw, band[0], band[2], color=PHASE_COLOR[phase], alpha=0.25, lw=0)
            ax.plot(raw, band[1], color=PHASE_COLOR[phase], lw=1.2)
            rec[f"{key}_median"] = np.round(band[1], 3).tolist()
            rec[f"{key}_iqr"] = [np.round(band[0], 3).tolist(), np.round(band[2], 3).tolist()]
            if col == 0:
                ax.set_ylabel(ylabel)
        # observed training outcomes in 10 quantile bins of this attribute (other attributes vary)
        sel = inside & np.isfinite(fit_oil)
        binned = []
        if sel.sum() >= 50:
            edges = np.quantile(raw_train[sel], np.linspace(0, 1, 11))
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = sel & (raw_train >= lo) & (raw_train <= hi)
                if m.sum():
                    binned.append(dict(x=float(np.median(raw_train[m])), oil_mbbl=float(np.median(fit_oil[m]) / VOLUME_DIVISOR),
                                       model_p50_mbbl=float(np.median(fit_pred[m]) / VOLUME_DIVISOR), wells=int(m.sum())))
            axes[0, col].plot([p["x"] for p in binned], [p["oil_mbbl"] for p in binned], ls="none", marker="s", ms=3,
                              color="0.25", zorder=4)
            axes[0, col].plot([p["x"] for p in binned], [p["model_p50_mbbl"] for p in binned], ls="none", marker="o", ms=3.4,
                              mfc="white", mec=PHASE_COLOR["oil"], mew=0.9, zorder=5)
        ax = axes[2, col]
        rug = np.percentile(raw_train[inside], np.arange(2, 100, 2))  # every 2nd percentile of the training wells
        ax.plot(rug, np.full(len(rug), 0.025), ls="none", marker="|", ms=3, color="0.35", mew=0.5,
                transform=ax.get_xaxis_transform())
        ax.set_xlabel(ATTRIBUTE_LABEL.get(feat, feat))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        for row in range(3):
            plain_ticks(axes[row, col])
        slope = np.polyfit(raw, band_median(values["cum1080"]), 1)[0]
        rec.update(eur_median_mbbl=rec["eur_median"], cum1080_median_mbbl=rec["cum1080_median"],
                   training_binned_cum1080=binned,
                   cum1080_slope_mbbl_per_unit=float(slope),
                   training_wells_present=int(present.sum()))
        if feat == "LateralLength_FT":
            rec["cum1080_slope_mbbl_per_1000_ft"] = float(1000 * slope)
        record[feat] = rec
    axes[0, 0].legend(handles=legend_handles([
        ("line", "one attribute changed, P50", dict(color=PHASE_COLOR["oil"], lw=1.2)),
        ("patch", "P75–P25", dict(color=PHASE_COLOR["oil"], alpha=0.25, lw=0)),
        ("marker", "wells grouped by attribute, reported", dict(marker="s", ms=3, color="0.25")),
        ("marker", "same wells, forecast P50", dict(marker="o", ms=3.4, mfc="white", mec=PHASE_COLOR["oil"], mew=0.9))]),
        loc="upper left", fontsize=NOTE_PT)
    label_panels([axes[r, c] for r in range(3) for c in range(4)])
    files = save(fig, ctx.output, "fig_attributes")
    record.update(files=files, origin=0, grid_points=17, partial_dependence_wells=int(len(pd_idx)), partial_dependence_seed=2026,
                  other_attributes="each of the partial-dependence training wells keeps its own values and missing flags; "
                                   "samples pooled over wells",
                  rug="every 2nd percentile of the training wells with the attribute present, inside the grid range",
                  **draw_record(ctx.bayes, PD_SAMPLES, 101, note="per well; fresh generators per grid value (common random numbers)"))
    return record


def band_median(values):
    return np.median(np.asarray(values), axis=1)


# ---- Supplementary: cross-phase correlation --------------------------------------------------------------
def supp_fig_correlation(ctx):
    order = np.argsort(ctx.row_ids)[:ctx.args.corr_wells]
    t = np.unique(np.round(np.geomspace(30, 3650, 40)))
    d = sample(ctx.bayes, ctx.reports[order], ctx.features[order], 90, 256, 61)
    lr = np.log(d.rate(t))  # [N, S, T, 3]
    pairs = {"oil-gas": (0, 1), "oil-water": (0, 2), "gas-water": (1, 2)}
    styles = {"oil-gas": dict(color="black", ls="-"), "oil-water": dict(color="#E69F00", ls="--"),
              "gas-water": dict(color="#56B4E9", ls="-.")}
    corr = {}
    for name, (i, j) in pairs.items():
        x0 = lr[..., i] - lr[..., i].mean(1, keepdims=True)
        x1 = lr[..., j] - lr[..., j].mean(1, keepdims=True)
        corr[name] = (x0 * x1).mean(1) / (x0.std(1) * x1.std(1) + 1e-12)  # [N, T]
    a = ctx.wells["A"]
    da = sample(ctx.bayes, ctx.reports[a:a + 1], ctx.features[a:a + 1], 90, 1000, 71)
    well_a = {}
    for day in (180.0, 720.0, 1800.0):
        v = np.log(da.rate([day])[0][:, 0, :])
        well_a[str(int(day))] = np.corrcoef(v.T).round(3).tolist()
    fig, axes = new_figure(HALF_W, 2.4)
    ax = axes[0, 0]
    ax.axhline(0.0, color="0.6", lw=0.5)
    bands = {name: pctl(c, [25, 50, 75]) for name, c in corr.items()}
    for name, band in bands.items():
        st = styles[name]
        ax.fill_between(t, band[0], band[2], color=st["color"], alpha=0.15, lw=0, rasterized=True)
        ax.plot(t, band[1], color=st["color"], ls=st["ls"], lw=1.1)
        i, j = pairs[name]
        days = [180.0, 720.0, 1800.0]
        ax.plot(days, [well_a[str(int(dd))][i][j] for dd in days], ls="none", marker="o", ms=3.2, mfc="white",
                mec=st["color"], mew=0.8, zorder=4)
    # direct labels at the left end, spread so they do not overlap
    starts = sorted(((float(band[1][0]), name) for name, band in bands.items()))
    placed = []
    for value, name in starts:
        y = value if not placed else max(value, placed[-1] + 0.09)
        placed.append(y)
        ax.text(t[0] * 1.06, y + 0.02, name.replace("-", "–"), color=styles[name]["color"], fontsize=NOTE_PT,
                ha="left", va="bottom", clip_on=True)
    log_axis(ax, "x", [30, 100, 365, 1080, 3650], (30, 3650))
    lo = min(-0.1, float(np.nanmin([b[0].min() for b in bands.values()])) - 0.05)
    ax.set_ylim(lo, 1.08)  # headroom for the shading label
    ax.set_yticks([v for v in (-0.5, 0.0, 0.5, 1.0) if v >= lo])
    plain_ticks(ax, "y")
    shade_beyond(ax, LAST_REPORT_DAY, label="beyond last report")
    ax.set_xlabel("Time, days")
    ax.set_ylabel("Correlation of log rates")
    ax.legend(handles=legend_handles([("marker", "well A (days 180, 720, 1,800)",
                                       dict(marker="o", ms=3.2, mfc="white", mec="0.2"))]), loc="lower right", fontsize=NOTE_PT)
    files = save(fig, ctx.output, "supp_fig_correlation")
    return dict(files=files, wells=int(len(order)), origin=90, times=t.tolist(),
                median_by_time={k: np.median(v, 0).round(3).tolist() for k, v in corr.items()},
                iqr_by_time={k: [np.percentile(v, 25, 0).round(3).tolist(), np.percentile(v, 75, 0).round(3).tolist()]
                             for k, v in corr.items()},
                well_A=well_a, samples=draw_record(ctx.bayes, 256, 61), well_A_samples=draw_record(ctx.bayes, 1000, 71))


# ---- Table 1 data: latent property audit -------------------------------------------------------------
def property_audit(ctx):
    order = np.argsort(ctx.row_ids)[:ctx.args.audit_wells]
    full = np.arange(0.0, 3651.0, 1.0)
    audit, families = {}, (("Bayes", ctx.bayes), ("basis", ctx.basis), ("DCA", ctx.dca))
    for name, family in families:
        if family is None:
            continue
        audit[name] = {}
        for origin in (0, 90, 360):
            d = sample(family, ctx.reports[order], ctx.features[order], origin, 64, 99 + origin)
            # Per member to bound memory; shares combine as sample-weighted means, maxima as maxima.
            parts = [(m.size, rate_checks(m.rate(full), m.cumulative(full))) for m in d.members()]
            total = sum(size for size, _ in parts)
            combined = {}
            for key in parts[0][1]:
                if key.startswith("max_"):
                    combined[key] = max(c[key] for _, c in parts)
                else:
                    combined[key] = sum(size * c[key] for size, c in parts) / total
            audit[name][str(origin)] = combined
    return dict(results=audit, wells=int(len(order)), samples=64, seeds_by_origin={str(o): 99 + o for o in (0, 90, 360)},
                members={name: len(f.members) for name, f in families if f is not None}, grid_days=[0, 3650])


FUNCTIONS = {"fig_updating": fig_updating, "fig_decline": fig_decline, "fig_modes": fig_modes,
             "fig_phase_coupling": fig_phase_coupling, "fig_attributes": fig_attributes,
             "supp_fig_correlation": supp_fig_correlation, "property_audit": property_audit}


# ---- loading -------------------------------------------------------------------------------------------------
def parse_paths(values) -> list[str]:
    paths = []
    for value in values or []:
        paths += [p.strip() for p in value.split(",") if p.strip()]
    return paths


def resolve_calibration(args) -> dict:
    path = args.calibration
    if path is None and args.model_label == "v1" and DEFAULT_CALIBRATION.is_file():
        path = str(DEFAULT_CALIBRATION)
    selected, source = {}, None
    if path and path.lower() != "none":
        if not Path(path).is_file():
            raise SystemExit(f"calibration file not found: {path}")
        payload = json.loads(Path(path).read_text())
        selected = payload.get("selected", {})
        source = source_record(path)
    report = args.report_inflation if args.report_inflation is not None else selected.get("report_inflation", 1.0)
    prior = args.prior_inflation if args.prior_inflation is not None else selected.get("prior_inflation", 1.0)
    return dict(report_inflation=report, prior_inflation=prior, source=source,
                overridden=dict(report=args.report_inflation is not None, prior=args.prior_inflation is not None))


def load_bayes(paths, device, calibration) -> Family:
    members = [DepletionBayesForecaster.load(p, device=device) for p in paths]
    for m in members:
        m.config.update(report_inflation=calibration["report_inflation"], prior_inflation=calibration["prior_inflation"])
    ref = members[0].model
    for p, m in zip(paths[1:], members[1:]):
        if not np.allclose(m.model.phase_scales.double().cpu().numpy(), ref.phase_scales.double().cpu().numpy()):
            raise SystemExit(f"{p}: phase scales differ from {paths[0]}; members must be fitted on the same wells")
        if tuple(m.model.rate.decay_times) != tuple(ref.rate.decay_times) or m.model.rate.ratio_index != ref.rate.ratio_index:
            raise SystemExit(f"{p}: latent law differs from {paths[0]} (decay grid or ratio index)")
    return Family("bayes", paths, members)


def load_comparator(name, paths, device) -> Family | None:
    if not paths:
        return None
    from joint_production.conditional_rates import ConditionalRateForecaster
    return Family(name, paths, [ConditionalRateForecaster.load(p, device=device) for p in paths])


class Context:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bayes", action="append", required=True, help="Bayesian checkpoint(s); repeat or comma-separate")
    parser.add_argument("--basis", action="append", help="positive-basis network checkpoint(s)")
    parser.add_argument("--dca", action="append", help="conditional DCA network checkpoint(s)")
    parser.add_argument("--model-label", default="v1", help="v1 (default), v2 or arch; sets legend names")
    parser.add_argument("--calibration", default=None,
                        help="calibration.json whose 'selected' block sets report/prior inflation; 'none' disables "
                             "(default for v1: build/v1/cal/calibration/calibration.json)")
    parser.add_argument("--report-inflation", type=float, default=None)
    parser.add_argument("--prior-inflation", type=float, default=None)
    parser.add_argument("--output", default=str(ROOT / "outputs/figures"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--audit-wells", type=int, default=96)
    parser.add_argument("--corr-wells", type=int, default=200)
    parser.add_argument("--only", help="comma list of " + ",".join(FIGURES))
    args = parser.parse_args(argv)
    wanted = [f for f in FIGURES if not args.only or f in set(args.only.split(","))]
    unknown = set(args.only.split(",")) - set(FIGURES) if args.only else set()
    if unknown:
        raise SystemExit(f"unknown figure(s): {sorted(unknown)}")
    apply_style()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    calibration = resolve_calibration(args)
    bayes_paths, basis_paths, dca_paths = parse_paths(args.bayes), parse_paths(args.basis), parse_paths(args.dca)
    if "fig_decline" in wanted and not basis_paths:
        raise SystemExit("fig_decline needs --basis")
    data = load_split("validation")
    reports, features, row_ids = data["holdout_reports"], data["holdout_features"], data["holdout_row_ids"]
    bayes = load_bayes(bayes_paths, args.device, calibration)
    basis = load_comparator("basis", basis_paths, args.device) if {"fig_decline", "property_audit"} & set(wanted) else None
    dca = load_comparator("dca", dca_paths, args.device) if "property_audit" in wanted else None
    wells = pick_wells(reports)
    law = bayes.members[0].model.rate
    figures_json = output / "figures.json"
    record = json.loads(figures_json.read_text()) if (args.only and figures_json.is_file()) else {}
    checkpoints = {p: sha256(p) for p in bayes_paths + basis_paths + dca_paths}
    provenance = dict(split="validation", model_label=args.model_label,
                      model_name=MODEL_LABEL.get(args.model_label, args.model_label),
                      bayes_models=bayes_paths, basis_models=basis_paths, dca_models=dca_paths,
                      calibration=calibration, pooling="equal samples per member; member i uses seed + i")
    record.update(provenance)
    record.update(bayes_model=bayes_paths[0], basis_model=basis_paths[0] if basis_paths else None,
                  dca_model=dca_paths[0] if dca_paths else None, checkpoints_sha256=checkpoints,
                  example_wells={k: dict(index=v, row_id=str(row_ids[v])) for k, v in wells.items()},
                  well_rules=pick_wells.__doc__, decay_times_days=list(law.decay_times),
                  tau_max_days=float(law.decay_times[-1]), phase_order=list(PHASES),
                  phase_scales_native=bayes.members[0].model.phase_scales.double().cpu().numpy().tolist(),
                  units=dict(rate="bbl/D (oil, water), Mcf/D (gas)", volume="Mbbl (oil, water), MMcf (gas)",
                             gor="Mcf/bbl", wor="bbl/bbl", native="eur_oil_* legacy fields and *_native fields are bbl/Mcf"),
                  legacy_key_map=LEGACY_KEYS)
    ctx = Context(args=args, data=data, reports=reports, features=features, row_ids=row_ids, bayes=bayes, basis=basis,
                  dca=dca, wells=wells, output=output, record=record, model_label=args.model_label,
                  decay_times=list(law.decay_times), tau_max=float(law.decay_times[-1]),
                  scale=bayes.members[0].model.phase_scales.double().cpu().numpy(), provenance=provenance)
    run = run_manifest(Path(__file__), dict(rendered=wanted))
    for name in wanted:
        record[name] = FUNCTIONS[name](ctx)
        # Per-figure provenance, so a partial re-render (--only) never inherits another run's labels.
        record[name]["provenance"] = dict(model_label=args.model_label, bayes_models=bayes_paths,
                                          basis_models=basis_paths, dca_models=dca_paths, calibration=calibration,
                                          run_created_at_utc=run["created_at_utc"])
        print(json.dumps(dict(event="figure", name=name, files=record[name].get("files"))), flush=True)
    if "property_audit" in record:
        record["fig8_audit"] = record["property_audit"]["results"]  # read by scripts/paper_results_bundle.py
    if "supp_fig_correlation" in record:
        corr = record["supp_fig_correlation"]
        record["fig5_correlation"] = dict(wells=corr["wells"], median_by_time=corr["median_by_time"], times=corr["times"],
                                          well_A=corr["well_A"])
    for old in ("fig1_updating", "fig2_rate_cumulative", "fig3_gas_oil", "fig4_drainage", "fig6_decline", "fig7_spectrum",
                "fig9_attributes"):
        record.pop(old, None)  # superseded keys from the v2 single-fit render (see legacy_key_map)
    record["run"] = run
    write_json(figures_json, record)
    print(json.dumps(dict(event="done", output=str(output), wells=record["example_wells"])), flush=True)


if __name__ == "__main__":
    main()
