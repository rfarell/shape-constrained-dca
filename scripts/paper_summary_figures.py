"""Summary figures for the manuscript, built from saved JSON score files only.

No model is loaded and nothing is re-scored: json, numpy and matplotlib only (no torch, no
joint_production import). Every plotted number is written to summary_figures.json in the output
directory together with the SHA-256 of each source file.

  fig_taumax       EUR and three-year cumulative against tau_max (v1 configuration, seed 42,
                   200 validation wells; post-test analysis)
  fig_accuracy     (a) scaled WIS by forecast date on test; (b) 80% coverage against 80% width
  fig_reliability  share of three-year outcomes above P90, P50 and P10 by forecast date (test)
  fig_ensemble     (a) single fits against five-fit ensembles on test; (b) ensemble gain against
                   member disagreement on the development holdout; (c) width-diverse ensemble
                   paired WIS gains (development, validation and test)

The width-diverse ("arch") series come from the --arch-* inputs; when a file is absent the series
is left out and summary_figures.json records "arch": null. fig_ensemble is skipped when the
development diversity file is absent.

Split checks: each loader asserts the well count (854 test, 450 validation) and that every result
has 28 forecast-date/horizon cells.

    python scripts/paper_summary_figures.py \
        --arch-val-scores results/scores/arch/score_val_full/scores.json \
        --arch-test-scores results/scores/arch/score_test_full/scores.json \
        --arch-hindcast results/scores/arch/hindcast_test/hindcast.json \
        --arch-val-diversity results/scores/arch/val_diversity.json \
        --arch-test-diversity results/scores/arch/test_diversity.json \
        --arch-test-pairs results/scores/arch/score_test_pairs/diversity.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from matplotlib.transforms import blended_transform_factory

from paper_figure_style import (FULL_W, HALF_W, LABEL_PT, MODEL_LABEL, MODEL_STYLE, NOTE_PT, PHASE_COLOR,
                                PHASE_LABEL, PHASES, ROOT, apply_style, figure_legend, legend_handles, new_figure,
                                note, panel_label, plain_ticks, run_manifest, save, source_record,
                                terminal_decline, write_json)

B = ROOT / "results"
DAYS = (0, 30, 60, 90, 180, 360, 720, 1080)
ORIGINS = (0, 30, 60, 90, 180, 360, 720)
CELLS = [(s, t) for s in ORIGINS for t in DAYS if t > s]  # origin-major, as in scripts/depletion_dev.py
ORIGIN_CELLS = {s: [i for i, (o, _) in enumerate(CELLS) if o == s] for s in ORIGINS}
MONTHS = {0: 0, 30: 1, 60: 2, 90: 3, 180: 6, 360: 12, 720: 24}
EXPECTED_WELLS = {"test": 854, "validation": 450}
SEEDS = (42, 43, 44, 45, 46)

# Result keys in the score files (five-fit ensembles unless stated; per-well baselines are single fits).
SCORE_KEY = {"v1": "bayes_cal_mix5", "v1_raw": "bayes_final_mix5", "v2": "bayes_v2_mix5", "basis": "basis_mix5",
             "dca": "dca_mix5", "neural": "neural_mix5", "forest": "forest_mix5", "arps": "pw_arps_bayes_seed42",
             "duong": "pw_duong_bayes_seed42", "sedm": "pw_sedm_bayes_seed42", "slab": "pw_scaling_bayes_seed42"}
HINDCAST_KEY = {"v1": "bayes_cal_mix5", "basis": "basis_mix5", "arps": "pw_arps_bayes"}
FAMILY_PREFIX = {"v1_raw": "bayes_final", "v2": "bayes_v2", "basis": "basis", "dca": "dca", "neural": "neural",
                 "forest": "forest"}
SHORT = {"v1": "First version", "v1_raw": "First, before widening", "v2": "Intermediate version", "basis": "Basis network", "dca": "DCA network",
         "neural": "Contextual network", "forest": "Forest", "arch": "The model", "arps": "Arps",
         "duong": "Duong", "sedm": "SEDM", "slab": "Slab scaling"}

# Printed values the figures must reproduce (Supplementary Tables s:tab:wis-origin and s:tab:seeds,
# results_bundle.md). Checked to the printed precision; a mismatch stops the script (--skip-checks
# downgrades it to a recorded warning).
EXPECTED_WIS_ORIGIN_TEST = {
    "bayes_cal_mix5": (0.1891, 0.1433, 0.1242, 0.1192, 0.1215, 0.1243, 0.1426),
    "basis_mix5": (0.1751, 0.1367, 0.1205, 0.1140, 0.1183, 0.1219, 0.1393),
    "dca_mix5": (0.1797, 0.1379, 0.1210, 0.1145, 0.1175, 0.1221, 0.1404),
    "neural_mix5": (0.1948, 0.1444, 0.1240, 0.1176, 0.1201, 0.1200, 0.1476),
    "forest_mix5": (0.1677, 0.1476, 0.1333, 0.1295, 0.1264, 0.1265, 0.1357),
    "pw_arps_bayes_seed42": (0.2235, 0.1800, 0.1539, 0.1531, 0.1470, 0.1438, 0.1528),
}
EXPECTED_GAIN_ORIGIN_TEST = {  # relative WIS gain of calibrated v1 by forecast date, %, point values
    "basis": (-7.9, -4.9, -3.0, -4.5, -2.7, -2.0, -2.4),
    "dca": (-5.2, -3.9, -2.7, -4.1, -3.4, -1.9, -1.5),
    "arps": (15.4, 20.4, 19.3, 22.1, 17.4, 13.6, 6.7),
}
EXPECTED_COVERAGE_WIDTH_TEST = {"v1": (79.0, 0.935), "v1_raw": (76.6, 0.876), "basis": (78.9, 0.871),
                                "dca": (80.4, 0.895), "neural": (83.2, 1.019), "forest": (82.5, 1.029),
                                "arps": (81.6, 1.192), "duong": (84.0, 1.473), "sedm": (85.2, 1.768),
                                "slab": (71.8, 1.206)}
EXPECTED_SEED_REDUCTION_TEST = {"v1_raw": 1.8, "v2": 1.7, "basis": 5.8, "dca": 5.6, "neural": 3.3, "forest": 1.1}
EXPECTED_CONFIRMATORY_TEST = {"basis": (-3.16, -5.54, -0.93), "dca": (-2.99, -4.96, -1.04),
                              "neural": (-0.82, -3.99, 3.70), "forest": (5.09, 2.71, 7.67),
                              "arps": (18.11, 14.87, 21.21)}
EXPECTED_CONFIRMATORY_VAL = {"basis": (-3.65, -6.84, -0.66), "dca": (-2.09, -4.99, 0.73),
                             "neural": (-1.38, -4.07, 1.26), "forest": (4.22, 1.09, 7.21),
                             "arps": (20.54, 14.91, 26.27)}
EXPECTED_HINDCAST_V1 = {0: (83.3, 84.1, 87.8), 720: (96.7, 97.0, 96.4)}
EXPECTED_TAUMAX = {1825: dict(eur=(0.75, 0.56, 0.58), cum=(1.006, 0.998, 1.006), wis=0.1377),
                   3650: dict(eur=(1.0, 1.0, 1.0), cum=(1.0, 1.0, 1.0), wis=0.1382),
                   7300: dict(eur=(1.25, 1.57, 1.91), cum=(1.000, 0.998, 1.003), wis=0.1375)}
EXPECTED_DEV_ARCH = {"base": (2.06, 1.34, 2.76), "basis5": (2.24, 0.07, 4.89), "dca5": (2.20, 0.50, 4.06),
                     "neural5": (2.18, 0.66, 3.78)}

CHECKS: list[dict] = []
SKIP_CHECKS = False


def check(condition: bool, message: str) -> None:
    CHECKS.append(dict(ok=bool(condition), check=message))
    if not condition:
        if SKIP_CHECKS:
            print(f"WARNING (check failed, continuing): {message}", flush=True)
        else:
            raise SystemExit(f"check failed: {message}")


def close(value: float, expected: float, decimals: int) -> bool:
    return abs(value - expected) <= 0.5 * 10.0 ** (-decimals) + 1e-9


def present(path) -> bool:
    return path is not None and Path(path).is_file()


def load_json(path):
    return json.loads(Path(path).read_text())


def load_scores(path, split: str, keys) -> dict:
    """Load a scores.json and assert the well count and cell count of every result used."""
    data = load_json(path)
    for key in keys:
        check(key in data["results"], f"{key} present in {path}")
        if key not in data["results"]:
            continue
        res = data["results"][key]
        check(int(res["all28"]["wells"]) == EXPECTED_WELLS[split],
              f"{key} in {path}: {res['all28']['wells']} wells, expected {EXPECTED_WELLS[split]} ({split})")
        check(len(res["all28"]["per_cell_phase"]["scaled_wis"]) == len(CELLS), f"{key} in {path}: 28 cells")
    return data


def wis_by_origin(result) -> list[float]:
    """Mean of the per-cell scaled WIS over each forecast date's cells and the three phases."""
    cells = np.asarray(result["all28"]["per_cell_phase"]["scaled_wis"], dtype=float)  # [28, 3]
    return [float(np.nanmean(cells[ORIGIN_CELLS[o]])) for o in ORIGINS]


def paired_gain(comparisons: dict, candidate: str, reference: str, scope: str = "original15"):
    """Relative WIS gain 1 - WIS(candidate)/WIS(reference) in % with the paired bootstrap 95% interval.

    Uses the reverse comparison when only that one was computed: g = 1 - 1/(1 - h) is decreasing in h,
    so the interval endpoints swap (exact for percentile intervals).
    """
    key = f"{candidate}_vs_{reference}"
    if key in comparisons and scope in comparisons[key]:
        c = comparisons[key][scope]
        return dict(point=100 * c["point"], lower95=100 * c["lower95"], upper95=100 * c["upper95"], source_key=key)
    key = f"{reference}_vs_{candidate}"
    if key in comparisons and scope in comparisons[key]:
        c = comparisons[key][scope]
        flip = lambda h: 100 * (1.0 - 1.0 / (1.0 - h))
        return dict(point=flip(c["point"]), lower95=flip(c["upper95"]), upper95=flip(c["lower95"]),
                    source_key=key, reversed=True)
    return None


def origin_axis(ax, label: bool = True) -> None:
    ax.set_xticks(range(len(ORIGINS)))
    ax.set_xticklabels(["0 (new)" if o == 0 else f"{MONTHS[o]}" for o in ORIGINS])
    ax.set_xlim(-0.4, len(ORIGINS) - 0.6)
    if label:
        ax.set_xlabel("History at the forecast date, months")


# ---- Fig. 5: tau_max sensitivity ------------------------------------------------------------------
def fig_taumax(args, output, summary, sources):
    if not present(args.taumax):
        return dict(skipped=f"missing {args.taumax}")
    data = load_json(args.taumax)
    sources["taumax"] = source_record(args.taumax)
    check(int(data["wells"]) == 200, f"taumax.json: {data['wells']} wells, expected 200 validation wells")
    taus = (1825, 3650, 7300)
    rows = {o: {t: data["table"][f"{t}/{o}"] for t in taus} for o in (90, 360, 720)}
    eur = np.array([rows[360][t]["eur_ratio"] for t in taus], dtype=float)  # [tau, phase]
    cum = np.array([rows[360][t]["cum1080_ratio"] for t in taus], dtype=float)
    for i, t in enumerate(taus):
        for k in range(3):
            check(close(eur[i, k], EXPECTED_TAUMAX[t]["eur"][k], 2), f"taumax EUR ratio {t}/360 phase {k}")
            check(close(cum[i, k], EXPECTED_TAUMAX[t]["cum"][k], 3), f"taumax cumulative ratio {t}/360 phase {k}")
    wis = {}
    for t, (path, key) in {1825: (args.taumax_wis_1825, "bayes_taumax1825_seed42"),
                           3650: (args.taumax_wis_3650, "bayes_final_seed42"),
                           7300: (args.taumax_wis_7300, "bayes_taumax7300_seed42")}.items():
        if present(path):
            d = load_scores(path, "validation", [key])
            wis[t] = float(d["results"][key]["wis_original15"])
            sources[f"taumax_wis_{t}"] = source_record(path)
            check(close(wis[t], EXPECTED_TAUMAX[t]["wis"], 4), f"taumax validation WIS {t}")
    eff = {t: terminal_decline(t)["effective_pct_per_year"] for t in taus}

    fig, axes = new_figure(HALF_W, 2.6)
    ax = axes[0, 0]
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.axhline(1.0, color="0.6", lw=0.6, ls=":", zorder=1)
    for o in (90, 720):
        faint = np.array([rows[o][t]["eur_ratio"] for t in taus], dtype=float)
        for k, phase in enumerate(PHASES):
            ax.plot(taus, faint[:, k], color=PHASE_COLOR[phase], lw=0.6, alpha=0.35, zorder=2)
    for k, phase in enumerate(PHASES):
        ax.plot(taus, eur[:, k], color=PHASE_COLOR[phase], ls="-", lw=1.3, marker="o", ms=3.5, zorder=4)
        ax.plot(taus, cum[:, k], color=PHASE_COLOR[phase], ls="--", lw=0.9, zorder=3)
        ax.annotate(PHASE_LABEL[phase], xy=(taus[-1], eur[-1, k]), xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=LABEL_PT - 1, color=PHASE_COLOR[phase])
    ax.set_xlim(taus[0] / 1.25, taus[-1] * 1.9)
    ax.set_ylim(0.5, 2.0)
    ax.set_xticks(taus)
    ax.set_xticklabels([f"{t / 365:.0f}\n{eff[t]:.1f}%/yr" + ("\n(model)" if t == 3650 else "") for t in taus])
    ax.minorticks_off()
    ax.set_yticks([0.5, 0.75, 1.0, 1.5, 2.0])
    plain_ticks(ax, "y")
    ax.set_xlabel("Terminal decay time, years\n(effective terminal decline)")
    ax.set_ylabel("Median ratio to the ten-year setting")
    ax.legend(handles=legend_handles([
        ("line", "EUR", dict(color="0.2", ls="-", lw=1.3, marker="o", ms=3)),
        ("line", "Three-year cumulative", dict(color="0.2", ls="--", lw=0.9)),
        ("line", "EUR, other forecast dates", dict(color="0.2", lw=0.6, alpha=0.35))]),
        loc="upper left", fontsize=NOTE_PT)
    files = save(fig, output, "fig_taumax")
    record = dict(files=files, origin=360, wells=200, model="v1 configuration, seed 42, all TRAIN; validation",
                  tau_max_days=list(taus), effective_terminal_decline_pct_per_year=eff,
                  median_eur_ratio={str(t): eur[i].tolist() for i, t in enumerate(taus)},
                  median_cum1080_ratio={str(t): cum[i].tolist() for i, t in enumerate(taus)},
                  faint_eur_ratio={str(o): {str(t): rows[o][t]["eur_ratio"] for t in taus} for o in (90, 720)},
                  validation_wis_original15_single_fit={str(t): v for t, v in wis.items()},
                  phase_order=list(PHASES), note="width ratios (NaN reference rows) are not plotted")
    summary["fig_taumax"] = record
    return record


# ---- Fig. 8: accuracy by forecast date and calibration against sharpness --------------------------
def fig_accuracy(args, output, summary, sources, arch):
    if not present(args.test_scores):
        return dict(skipped=f"missing {args.test_scores}")
    models = ("v1", "basis", "dca", "neural", "forest", "arps")
    scatter = ("v1", "v1_raw", "basis", "dca", "neural", "forest", "arps", "duong", "sedm", "slab")
    test = load_scores(args.test_scores, "test", sorted({SCORE_KEY[m] for m in models + scatter}))
    sources["test_scores"] = source_record(args.test_scores)
    res, comp = test["results"], test["comparisons"]
    per = {m: wis_by_origin(res[SCORE_KEY[m]]) for m in models}
    for m in models:
        for i, value in enumerate(per[m]):
            check(close(value, EXPECTED_WIS_ORIGIN_TEST[SCORE_KEY[m]][i], 4),
                  f"test WIS by forecast date {SCORE_KEY[m]} origin {ORIGINS[i]} = {value:.5f}")
    gains = {ref: [100 * (1 - a / b) for a, b in zip(per["v1"], per[ref])] for ref in ("basis", "dca", "arps")}
    for ref, values in gains.items():
        for i, value in enumerate(values):
            check(abs(value - EXPECTED_GAIN_ORIGIN_TEST[ref][i]) <= 0.06,
                  f"v1 gain vs {ref} at origin {ORIGINS[i]} = {value:.2f}")
    cov = {m: (100 * res[SCORE_KEY[m]]["all28"]["macro_coverage_80"], res[SCORE_KEY[m]]["all28"]["macro_scaled_width_80"])
           for m in scatter}
    for m, (c, w) in cov.items():
        check(close(c, EXPECTED_COVERAGE_WIDTH_TEST[m][0], 1) and close(w, EXPECTED_COVERAGE_WIDTH_TEST[m][1], 3),
              f"test coverage/width {m} = {c:.2f}/{w:.4f}")
    confirm_test = {ref: paired_gain(comp, SCORE_KEY["v1"], SCORE_KEY[ref]) for ref in EXPECTED_CONFIRMATORY_TEST}
    for ref, g in confirm_test.items():
        check(g is not None and all(close(v, e, 2) for v, e in zip((g["point"], g["lower95"], g["upper95"]),
                                                                    EXPECTED_CONFIRMATORY_TEST[ref])),
              f"test paired gain v1 calibrated vs {ref}")
    confirm_val = {}
    if present(args.val_scores):
        val = load_scores(args.val_scores, "validation", [SCORE_KEY["v1"]])
        sources["val_scores_v1_calibrated"] = source_record(args.val_scores)
        confirm_val = {ref: paired_gain(val["comparisons"], SCORE_KEY["v1"], SCORE_KEY[ref])
                       for ref in EXPECTED_CONFIRMATORY_VAL}
        for ref, g in confirm_val.items():
            check(g is not None and all(close(v, e, 2) for v, e in zip((g["point"], g["lower95"], g["upper95"]),
                                                                        EXPECTED_CONFIRMATORY_VAL[ref])),
                  f"validation paired gain v1 calibrated vs {ref}")
    arch_gains = {}
    if arch.get("test_mix"):
        per["arch"] = wis_by_origin(arch["test_mix"])
        cov["arch"] = (100 * arch["test_mix"]["all28"]["macro_coverage_80"], arch["test_mix"]["all28"]["macro_scaled_width_80"])
        arch_gains = {ref: [100 * (1 - a / b) for a, b in zip(per["arch"], per[ref])] for ref in ("basis", "dca", "arps", "v1")}

    fig, axes = new_figure(FULL_W, 2.6, 1, 2, gridspec_kw=dict(width_ratios=[1.6, 1.0]))
    ax = axes[0, 0]
    x = np.arange(len(ORIGINS))
    # The main-text figure shows the model and the baselines; the first version's values stay in the record.
    order = [m for m in ("arps", "forest", "neural", "dca", "basis", "arch") if m in per]
    for m in order:
        st = MODEL_STYLE[m]
        ax.plot(x, per[m], color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"], ms=3.2,
                mfc=st.get("mfc", st["color"]), mec=st["color"], label=MODEL_LABEL[m], zorder=5 if m == "arch" else 3)
    origin_axis(ax)
    top = max(max(v) for v in per.values())
    ax.set_ylim(0.095, top * 1.04)
    ax.set_ylabel("Scaled WIS (lower is better)")
    plain_ticks(ax, "y")
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot([2, 2, 6, 6], [0.075, 0.045, 0.045, 0.075], transform=trans, color="0.4", lw=0.6, clip_on=False)
    ax.text(4, 0.055, "main comparison", transform=trans, ha="center", va="bottom", fontsize=NOTE_PT, color="0.3")
    handles, labels = ax.get_legend_handles_labels()
    rank = {MODEL_LABEL[m]: i for i, m in enumerate(("arch", "basis", "dca", "neural", "forest", "arps"))}
    pairs = sorted(zip(handles, labels), key=lambda p: rank.get(p[1], 99))
    ax.legend([p[0] for p in pairs], [p[1] for p in pairs], loc="upper right")
    panel_label(ax, "a")

    ax = axes[0, 1]
    ax.axhline(80.0, color="0.4", lw=0.8, ls=":", zorder=1)
    ax.text(1.83, 80.3, "nominal 80%", ha="right", va="bottom", fontsize=NOTE_PT, color="0.3")
    offsets = {"v1_raw": (6, -9, "left"), "arps": (5, 0, "left"), "duong": (5, 0, "left"),
               "sedm": (-5, 0, "right"), "slab": (5, 0, "left")}  # the ensemble and v1 markers are named by the legend of (a)
    for m, (c, w) in cov.items():
        if m in ("v1", "v1_raw"):
            continue
        st = MODEL_STYLE[m]
        ax.plot(w, c, ls="none", marker=st["marker"], ms=5 if m == "arch" else 4, color=st["color"],
                mfc=st.get("mfc", st["color"]), mec=st["color"], zorder=4)
        if m in offsets:
            dx, dy, ha = offsets[m]
            ax.annotate(SHORT[m], xy=(w, c), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                        fontsize=NOTE_PT, color=st["color"])
    ax.set_xlim(0.8, 1.85)
    ax.set_ylim(70.0, 87.0)
    ax.set_xlabel("Scaled 80% interval width")
    ax.set_ylabel("80% interval coverage, %")
    plain_ticks(ax)
    panel_label(ax, "b")
    files = save(fig, output, "fig_accuracy")
    record = dict(files=files, split="test", wells=854, origins_days=list(ORIGINS),
                  wis_by_origin={m: v for m, v in per.items()},
                  v1_gain_by_origin_pct={ref: v for ref, v in gains.items()},
                  arch_gain_by_origin_pct=arch_gains,
                  coverage80_pct_width80_all_dates={m: dict(coverage80_pct=c, width80_scaled=w) for m, (c, w) in cov.items()},
                  confirmatory_gain_v1_calibrated_original15_pct=dict(test=confirm_test, validation=confirm_val),
                  score_keys={m: SCORE_KEY[m] for m in set(models) | set(scatter)},
                  note="per-date values are point values (no per-date paired intervals); width and coverage over all 28 cells")
    summary["fig_accuracy"] = record
    return record


# ---- Fig. 9: P90/P50/P10 reliability by forecast date -----------------------------------------------
def fig_reliability(args, output, summary, sources, arch):
    if not present(args.hindcast):
        return dict(skipped=f"missing {args.hindcast}")
    h = load_json(args.hindcast)
    sources["hindcast"] = source_record(args.hindcast)
    check(list(h["origins"]) == list(ORIGINS), "hindcast origins 0..720")
    check(list(h["phases"]) == list(PHASES), "hindcast phase order oil, gas, water")
    series = {}
    for m, key in HINDCAST_KEY.items():
        r = h["results"][key]["to1080_by_origin"]
        check(np.all(np.asarray(r["wells"]) == EXPECTED_WELLS["test"]), f"hindcast {key}: 854 test wells")
        series[m] = {lvl: 100 * np.asarray(r[f"exceed_{lvl}"], dtype=float) for lvl in ("p90", "p50", "p10")}
    for origin, expected in EXPECTED_HINDCAST_V1.items():
        i = ORIGINS.index(origin)
        check(all(close(series["v1"]["p90"][i, k], expected[k], 1) for k in range(3)),
              f"hindcast v1 share above P90 at origin {origin}")
    if arch.get("hindcast"):
        series["arch"] = {lvl: 100 * np.asarray(arch["hindcast"][f"exceed_{lvl}"], dtype=float) for lvl in ("p90", "p50", "p10")}
    n = EXPECTED_WELLS["test"]
    se = {p: 200 * math.sqrt(p * (1 - p) / n) for p in (0.9, 0.5, 0.1)}  # two binomial standard errors, points

    fig, axes = new_figure(FULL_W, 3.2, 2, 3, sharex=True)
    x = np.arange(len(ORIGINS))
    order = [m for m in ("arps", "basis", "arch") if m in series]
    for k, phase in enumerate(PHASES):
        top, bottom = axes[0, k], axes[1, k]
        for ax, levels in ((top, (0.9,)), (bottom, (0.5, 0.1))):
            for p in levels:
                ax.fill_between([-0.5, len(ORIGINS) - 0.5], 100 * p - se[p], 100 * p + se[p], color="0.88", lw=0, zorder=0)
                ax.axhline(100 * p, color="0.4", lw=0.7, ls=":", zorder=1)
        for m in order:
            st = MODEL_STYLE[m]
            top.plot(x, series[m]["p90"][:, k], color=st["color"], ls=st["ls"], lw=st["lw"], marker="o", ms=2.8,
                     label=MODEL_LABEL[m], zorder=4 if m == "arch" else 3)
            bottom.plot(x, series[m]["p50"][:, k], color=st["color"], ls=st["ls"], lw=st["lw"], marker="o", ms=2.8, zorder=3)
            bottom.plot(x, series[m]["p10"][:, k], color=st["color"], ls=st["ls"], lw=st["lw"], marker="s", ms=2.8,
                        mfc="white", zorder=3)
        top.set_ylim(75, 100)
        bottom.set_ylim(0, 65)
        top.set_title(PHASE_LABEL[phase])
        origin_axis(bottom, label=False)
        plain_ticks(top, "y")
        plain_ticks(bottom, "y")
        panel_label(top, "abc"[k])
        panel_label(bottom, "def"[k])
    axes[0, 0].set_ylabel("Above P90, %")
    axes[1, 0].set_ylabel("Above P50, P10, %")
    fig.supxlabel("History at the forecast date, months", fontsize=LABEL_PT)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    rank = {MODEL_LABEL[m]: i for i, m in enumerate(("arch", "basis", "arps"))}
    pairs = sorted(zip(handles, labels), key=lambda p: rank.get(p[1], 99))
    extra = legend_handles([("patch", "nominal ± 2 binomial SE", dict(color="0.88")),
                            ("marker", "above P50", dict(marker="o", ms=2.8, color="0.3")),
                            ("marker", "above P10", dict(marker="s", ms=2.8, mfc="white", mec="0.3"))])
    figure_legend(fig, [p[0] for p in pairs] + extra, [p[1] for p in pairs] + [h.get_label() for h in extra],
                  ncol=3 if len(pairs) < 4 else 4, fallback_ax=axes[1, 0], fallback_loc="center right")
    files = save(fig, output, "fig_reliability")
    record = dict(files=files, split="test", wells=n, target="reported volume from the forecast date to day 1,080",
                  origins_days=list(ORIGINS), phase_order=list(PHASES),
                  share_above_pct={m: {lvl: v.tolist() for lvl, v in s.items()} for m, s in series.items()},
                  binomial_two_se_points={str(p): v for p, v in se.items()},
                  hindcast_keys=HINDCAST_KEY, note="bands ignore clustering in 478 groups and multiplicity")
    summary["fig_reliability"] = record
    return record


# ---- Fig. 10: ensembles ----------------------------------------------------------------------------------
def fig_ensemble(args, output, summary, sources, arch):
    if not present(args.diversity):
        return dict(skipped=f"missing {args.diversity} (development diversity file)")
    if not present(args.test_scores):
        return dict(skipped=f"missing {args.test_scores}")
    keys = [f"{FAMILY_PREFIX[f]}_seed{s}" for f in FAMILY_PREFIX for s in SEEDS] + [f"{FAMILY_PREFIX[f]}_mix5" for f in FAMILY_PREFIX]
    test = load_scores(args.test_scores, "test", keys)
    sources["test_scores"] = source_record(args.test_scores)
    dev = load_json(args.diversity)
    sources["diversity_dev"] = source_record(args.diversity)
    rows = {}
    for fam, prefix in FAMILY_PREFIX.items():
        singles = [float(test["results"][f"{prefix}_seed{s}"]["wis_original15"]) for s in SEEDS]
        mix = float(test["results"][f"{prefix}_mix5"]["wis_original15"])
        gain = 100 * (1 - mix / float(np.mean(singles)))
        check(abs(gain - EXPECTED_SEED_REDUCTION_TEST[fam]) <= 0.06, f"test seed-ensemble reduction {fam} = {gain:.2f}%")
        rows[fam] = dict(single=singles, ensemble=mix, reduction_pct=gain)
    arch_mix = (arch["test_mix"]["wis_original15"] if arch.get("test_mix")
                else (arch.get("test_diversity_points") or {}).get("arch", {}).get("wis_original15"))
    if arch.get("test_members") and arch_mix is not None:
        singles = [float(v) for v in arch["test_members"]]
        mix = float(arch_mix)
        rows["arch"] = dict(single=singles, ensemble=mix, reduction_pct=100 * (1 - mix / float(np.mean(singles))))
    comparators = {"basis5": "basis", "dca5": "dca", "neural5": "neural"}
    points = {k: dict(gain_pct=float(v["ensemble_gain_pct"]), disagreement=float(v["disagreement_index"]),
                      wis_original15=float(v["wis_original15"]), group=v.get("group")) for k, v in dev["results"].items()}
    dev_rows = {}
    for ref, expected in EXPECTED_DEV_ARCH.items():
        g = paired_gain(dev["comparisons"], "arch", ref)
        check(g is not None and all(close(v, e, 2) for v, e in zip((g["point"], g["lower95"], g["upper95"]), expected)),
              f"development arch vs {ref}")
        dev_rows[ref] = g

    fig, axes = new_figure(FULL_W, 2.6, 1, 3, gridspec_kw=dict(width_ratios=[1.3, 1.0, 1.0]))
    # (a) single fits and five-fit ensembles on test, forecast dates 60-720 d
    ax = axes[0, 0]
    order = [f for f in ("arch", "v1_raw", "v2", "basis", "dca", "neural", "forest") if f in rows]
    for y, fam in enumerate(reversed(order)):
        r = rows[fam]
        st = MODEL_STYLE[fam]
        ax.plot(r["single"], [y] * len(r["single"]), ls="none", marker="o", ms=3.2, mfc="white", mec=st["color"], mew=0.8)
        ax.plot([r["ensemble"]], [y], ls="none", marker="D", ms=4.5, color=st["color"], mec=st["color"])
        mean = float(np.mean(r["single"]))
        ax.annotate("", xy=(r["ensemble"], y), xytext=(mean, y),
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="0.35", shrinkA=0, shrinkB=3))
        ax.annotate(f"\N{MINUS SIGN}{r['reduction_pct']:.1f}%", xy=(0.5 * (r["ensemble"] + mean), y), xytext=(0, 4.5),
                    textcoords="offset points", ha="center", va="bottom", fontsize=NOTE_PT, color="0.25")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT[f] for f in reversed(order)])
    ax.set_ylim(-0.6, len(order) + 1.3)  # headroom above the top row for the legend
    ax.set_xlabel("Scaled WIS, test, 60–720 days")
    plain_ticks(ax, "x")
    ax.legend(handles=legend_handles([("marker", "single fit (seeds 42–46)", dict(marker="o", ms=3.2, mfc="white", mec="0.2")),
                                      ("marker", "five-fit ensemble", dict(marker="D", ms=4.5, color="0.2"))]),
              loc="upper right", fontsize=NOTE_PT)
    panel_label(ax, "a")

    # (b) development holdout: ensemble gain against member disagreement (selection data)
    ax = axes[0, 1]
    for key, p in points.items():
        if key in comparators:
            st = MODEL_STYLE[comparators[key]]
            ax.plot(p["disagreement"], p["gain_pct"], ls="none", marker="s", ms=4.5, mfc="white", mec=st["color"], mew=1.0, zorder=3)
        elif key == "arch":
            ax.plot(p["disagreement"], p["gain_pct"], ls="none", marker="o", ms=5.5, color="black", zorder=5)
        elif key == "base":
            ax.plot(p["disagreement"], p["gain_pct"], ls="none", marker="o", ms=4.5, color="black", zorder=4)
        else:
            ax.plot(p["disagreement"], p["gain_pct"], ls="none", marker="o", ms=3.5, color="0.65", zorder=2)
    labels = {"arch": ("widths 32–512", (0, 9), "center", "black"),
              "base": ("same width (v2)", (6, -4), "left", "black"),
              "basis5": (SHORT["basis"], (-6, 3), "right", MODEL_STYLE["basis"]["color"]),
              "dca5": (SHORT["dca"], (-4, -9), "right", MODEL_STYLE["dca"]["color"]),
              "neural5": (SHORT["neural"], (6, -6), "left", MODEL_STYLE["neural"]["color"])}
    for key, (text, (dx, dy), ha, color) in labels.items():
        if key in points:
            ax.annotate(text, xy=(points[key]["disagreement"], points[key]["gain_pct"]), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va="center", fontsize=NOTE_PT, color=color)
    ax.set_xlim(0.025, 0.135)
    ax.set_ylim(0.0, 7.0)
    ax.set_xlabel("Member disagreement index")
    ax.set_ylabel("Reduction from pooling, %")
    plain_ticks(ax)
    note(ax, 0.02, 0.98, "internal holdout\n(selection data)", va="top", ha="left")
    note(ax, 0.98, 0.03, "gray: other variants\nof the model", ha="right", va="bottom", color="0.45")
    panel_label(ax, "b")

    # (c) width-diverse ensemble: paired relative WIS gain with 95% intervals
    # Rows: references; markers: development (selection data), validation, test (third evaluation).
    ax = axes[0, 2]
    candidates = [(("base", "v2ref"), SHORT["v2"], "bayes_v2_mix5"), (("basis5",), SHORT["basis"], "basis_mix5"),
                  (("dca5",), SHORT["dca"], "dca_mix5"), (("neural5",), SHORT["neural"], "neural_mix5"),
                  (("v1raw",), SHORT["v1"], "bayes_final_mix5")]
    held = {"validation": {}, "test": {}}
    for split, tag in (("validation", "val"), ("test", "test")):
        comps = arch.get(f"{tag}_comparisons")
        dcomps = arch.get(f"{tag}_diversity_comparisons")
        for dev_keys, _, ref_key in candidates:
            g = paired_gain(comps, arch["mix_key"], ref_key) if comps is not None else None
            if g is None and dcomps is not None:
                g = next((x for x in (paired_gain(dcomps, "arch", k) for k in dev_keys) if x is not None), None)
            if g is not None:
                held[split][ref_key] = g
    refs = [c for c in candidates if any(k in dev_rows for k in c[0]) or any(c[2] in held[s] for s in held)]
    styles = {"development": dict(marker="D", ms=3.8, color="0.55", mfc="0.55", dy=0.22),
              "validation": dict(marker="o", ms=3.8, color="black", mfc="white", dy=0.0),
              "test": dict(marker="o", ms=3.8, color="black", mfc="black", dy=-0.22)}
    n = len(refs)
    for i, (dev_keys, label, ref_key) in enumerate(refs):
        y = n - 1 - i
        values = {"development": next((dev_rows[k] for k in dev_keys if k in dev_rows), None),
                  "validation": held["validation"].get(ref_key), "test": held["test"].get(ref_key)}
        for split, g in values.items():
            if g is None:
                continue
            st = styles[split]
            ax.errorbar([g["point"]], [y + st["dy"]], xerr=[[g["point"] - g["lower95"]], [g["upper95"] - g["point"]]],
                        fmt=st["marker"], ms=st["ms"], color=st["color"], mfc=st["mfc"], mec=st["color"],
                        elinewidth=0.8, capsize=0, zorder=3)
    ax.axvline(0.0, color="0.4", lw=0.7, zorder=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([label for _, label, _ in reversed(refs)])
    ax.set_ylim(-1.6, n + 0.2)  # headroom: legend below the last row, note above the first
    ax.set_xlabel("ΔWIS of the ensemble, %")
    note(ax, 0.98, 0.99, "better \N{RIGHTWARDS ARROW}", ha="right", va="top")
    plain_ticks(ax, "x")
    entries = [("marker", "internal holdout", dict(marker="D", ms=3.8, color="0.55"))]
    if held["validation"]:
        entries.append(("marker", "validation", dict(marker="o", ms=3.8, mfc="white", mec="black")))
    if held["test"]:
        entries.append(("marker", "test", dict(marker="o", ms=3.8, color="black")))
    ax.legend(handles=legend_handles(entries), loc="lower right", fontsize=NOTE_PT)
    if not held["validation"] and not held["test"]:
        note(ax, 0.02, 0.99, "held-out rows pending", ha="left", va="top", style="italic")
    panel_label(ax, "c")
    files = save(fig, output, "fig_ensemble")
    record = dict(files=files, test_original15_single_vs_ensemble=rows,
                  development_points=points, development_wells_note="TRAIN-internal holdout (1,392 wells); selection data",
                  development_arch_gain_pct=dev_rows, heldout_arch_gain_pct=held,
                  arch_heldout_diversity=dict(validation=arch.get("val_diversity_points"), test=arch.get("test_diversity_points")),
                  arch_test_pairs_original15_pct={k: paired_gain(arch["test_pairs"], *k.split("_vs_")) for k in arch.get("test_pairs", {})})
    summary["fig_ensemble"] = record
    return record


# ---- optional width-diverse ensemble inputs ----------------------------------------------------------------
def load_arch(args, sources) -> dict | None:
    """Collect whatever ARCH artifacts exist. Keys: mix_key, test_mix, test_members, hindcast,
    val/test comparisons (from scores.json), val/test diversity comparisons and points."""
    label = args.arch_label
    mix_key = f"{label}_mix5"
    arch = dict(mix_key=mix_key)
    found = False
    for split, path in (("val", args.arch_val_scores), ("test", args.arch_test_scores)):
        if not present(path):
            continue
        name = "validation" if split == "val" else "test"
        data = load_scores(path, name, [mix_key])
        sources[f"arch_{split}_scores"] = source_record(path)
        arch[f"{split}_comparisons"] = data.get("comparisons", {})
        arch[f"{split}_mix"] = data["results"][mix_key]
        members = [data["results"].get(f"{label}_seed{s}") for s in SEEDS]
        if all(m is not None for m in members):
            arch[f"{split}_members"] = [m["wis_original15"] for m in members]
        found = True
    for split, path in (("val", args.arch_val_diversity), ("test", args.arch_test_diversity)):
        if not present(path):
            continue
        d = load_json(path)
        sources[f"arch_{split}_diversity"] = source_record(path)
        arch[f"{split}_diversity_comparisons"] = d.get("comparisons", {})
        arch[f"{split}_diversity_points"] = {k: dict(gain_pct=v.get("ensemble_gain_pct"), disagreement=v.get("disagreement_index"),
                                                     wis_original15=v.get("wis_original15"))
                                             for k, v in d.get("results", {}).items()}
        if f"{split}_members" not in arch and "arch" in d.get("results", {}):
            arch[f"{split}_members"] = d["results"]["arch"].get("member_wis_original15")
        found = True
    if present(args.arch_test_pairs):
        d = load_json(args.arch_test_pairs)
        sources["arch_test_pairs"] = source_record(args.arch_test_pairs)
        arch.setdefault("test_diversity_comparisons", {}).update(d.get("comparisons", {}))
        arch["test_pairs"] = d.get("comparisons", {})
    if present(args.arch_hindcast):
        h = load_json(args.arch_hindcast)
        sources["arch_hindcast"] = source_record(args.arch_hindcast)
        key = args.arch_hindcast_key or next((k for k in h["results"] if k.startswith((label, "arch"))), None)
        check(key is not None and key in h["results"], f"ARCH hindcast key present in {args.arch_hindcast}")
        r = h["results"][key]["to1080_by_origin"]
        check(list(h["origins"]) == list(ORIGINS), "ARCH hindcast origins 0..720")
        check(np.all(np.asarray(r["wells"]) == EXPECTED_WELLS["test"]), "ARCH hindcast: 854 test wells")
        arch["hindcast"] = r
        arch["hindcast_key"] = key
        found = True
    return arch if found else None


def main(argv=None):
    global SKIP_CHECKS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(ROOT / "outputs/figures"))
    parser.add_argument("--test-scores", default=str(B / "scores/test/score_test_paper/scores.json"))
    parser.add_argument("--val-scores", default=str(B / "v1/test/score_val_final/scores.json"),
                        help="validation scores with the calibrated v1 ensemble (bayes_cal_mix5) and its comparisons")
    parser.add_argument("--hindcast", default=str(B / "v1/analysis/hindcast_test/hindcast.json"))
    parser.add_argument("--taumax", default=str(B / "scores/taumax/taumax.json"))
    parser.add_argument("--taumax-wis-1825", default=str(B / "scores/taumax/runs/score_val_taumax1825/scores.json"))
    parser.add_argument("--taumax-wis-3650", default=str(B / "v1/val/score_val_bayes_s42/scores.json"))
    parser.add_argument("--taumax-wis-7300", default=str(B / "scores/taumax/runs/score_val_taumax7300/scores.json"))
    parser.add_argument("--diversity", default=str(B / "scores/diversity_dev/diversity.json"))
    parser.add_argument("--arch-label", default="bayes_arch",
                        help="result-key prefix of the width-diverse ensemble in ARCH scores.json files (<label>_mix5, <label>_seed42..46)")
    parser.add_argument("--arch-val-scores")
    parser.add_argument("--arch-test-scores")
    parser.add_argument("--arch-hindcast")
    parser.add_argument("--arch-hindcast-key")
    parser.add_argument("--arch-val-diversity", help="scripts/ensemble_diversity.py output on validation (variant 'arch')")
    parser.add_argument("--arch-test-diversity", help="scripts/ensemble_diversity.py output on test (variant 'arch')")
    parser.add_argument("--arch-test-pairs", help="extra scripts/ensemble_diversity.py output on test (arch vs v2ref, v1cal, pwarps)")
    parser.add_argument("--only", help="comma list of figures to render (fig_taumax,fig_accuracy,fig_reliability,fig_ensemble)")
    parser.add_argument("--skip-checks", action="store_true", help="record failed value checks instead of stopping")
    args = parser.parse_args(argv)
    SKIP_CHECKS = bool(args.skip_checks)
    apply_style()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "summary_figures.json"
    summary = load_json(json_path) if (args.only and json_path.is_file()) else {}
    sources = summary.get("sources", {})
    arch = load_arch(args, sources)
    summary["arch"] = None if arch is None else dict(label=args.arch_label, mix_key=arch["mix_key"],
                                                     hindcast_key=arch.get("hindcast_key"),
                                                     inputs=sorted(k for k in sources if k.startswith("arch_")))
    arch = arch or {"mix_key": f"{args.arch_label}_mix5"}
    wanted = set(args.only.split(",")) if args.only else {"fig_taumax", "fig_accuracy", "fig_reliability", "fig_ensemble"}
    status = {}
    if "fig_taumax" in wanted:
        status["fig_taumax"] = fig_taumax(args, output, summary, sources)
    if "fig_accuracy" in wanted:
        status["fig_accuracy"] = fig_accuracy(args, output, summary, sources, arch)
    if "fig_reliability" in wanted:
        status["fig_reliability"] = fig_reliability(args, output, summary, sources, arch)
    if "fig_ensemble" in wanted:
        status["fig_ensemble"] = fig_ensemble(args, output, summary, sources, arch)
    for name, result in status.items():
        if "skipped" in result:
            summary[name] = result
            print(json.dumps(dict(figure=name, skipped=result["skipped"])), flush=True)
    summary["sources"] = sources
    summary["checks"] = CHECKS
    summary["run"] = run_manifest(Path(__file__))
    write_json(json_path, summary)
    print(json.dumps(dict(event="done", output=str(output), figures={k: ("skipped" if "skipped" in v else "ok")
                                                                     for k, v in status.items()},
                          failed_checks=sum(not c["ok"] for c in CHECKS))), flush=True)


if __name__ == "__main__":
    main()
