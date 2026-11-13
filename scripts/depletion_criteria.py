"""Apply the frozen final-protocol criteria (configs/depletion_final_protocol.toml) to a scores.json.

Accuracy: reference = basis_mix5 (strongest validation mixture). Non-inferior if the
paired whole-group 95% upper bound of relative WIS loss <= 0.02 and macro R2 >=
reference - 0.01; superior if the lower bound of relative WIS gain > 0. Calibration:
|coverage - nominal| <= 0.03 (macro) and <= 0.04 (each phase) at 80/90/95%, and
<= 0.05 for every origin at 90% (each phase). All scopes use the frozen cells.

    python scripts/depletion_criteria.py <scores.json> --candidate bayes_cal_mix5
    python scripts/depletion_criteria.py <scores.json> --compare   # calibration/sharpness, all models

The comparison mode reports, per model, mean absolute coverage error over the
80/90/95% levels and phases (all28 cells), the same at 90% over every origin and
phase, the worst such cell, and macro scaled 80%/95% width (training-derived scales).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LEVELS = ("0.8", "0.9", "0.95")


def calibration_comparison(results):
    """Markdown table of coverage error and scaled width for every model in a scores.json."""
    rows = ["| Model | Cov 80/90/95 | MACE levels x phases | MACE 90% origins x phases | Worst 90% cell | "
            "Width80 | Width95 | WIS15 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, r in sorted(results.items(), key=lambda item: item[1]["wis_original15"]):
        coverage = r["coverage"]
        if coverage["0.9"]["macro_all28"] == 0.0:
            continue  # point (MAP) forecasts carry no intervals
        level_errors = [abs(p - float(level)) for level in LEVELS for p in coverage[level]["phase_all28"]]
        origin_errors = [abs(p - 0.9) for v in coverage["0.9"]["by_origin_phase"].values() for p in v if p == p]
        macro = " / ".join(f"{100 * coverage[level]['macro_all28']:.1f}" for level in LEVELS)
        rows.append(f"| {label} | {macro} | {100 * sum(level_errors) / len(level_errors):.2f} | "
                    f"{100 * sum(origin_errors) / len(origin_errors):.2f} | {100 * max(origin_errors):.1f} | "
                    f"{r['all28']['macro_scaled_width_80']:.3f} | {r['all28']['macro_scaled_width_95']:.3f} | "
                    f"{r['wis_original15']:.4f} |")
    return "\n".join(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores")
    parser.add_argument("--candidate", default="bayes_cal_mix5")
    parser.add_argument("--reference", default="basis_mix5")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args(argv)
    data = json.loads(Path(args.scores).read_text())
    if args.compare:
        print(calibration_comparison(data["results"]))
        return
    results, comparisons = data["results"], data["comparisons"]
    cand, ref = results[args.candidate], results[args.reference]
    report = {}
    for label in sorted(k for k in comparisons if k.startswith(args.candidate + "_vs_")):
        primary = comparisons[label]["original15"]
        other = label.split("_vs_", 1)[1]
        r2_ok = cand["r2_original15"] >= results[other]["r2_original15"] - 0.01
        report[label] = dict(gain=primary["point"], gain_lower95=primary["lower95"], gain_upper95=primary["upper95"],
                             noninferior=bool(-primary["lower95"] <= 0.02 and r2_ok), superior=bool(primary["lower95"] > 0),
                             r2_candidate=cand["r2_original15"], r2_other=results[other]["r2_original15"])
    coverage = cand["coverage"]
    calib = {}
    for level in LEVELS:
        nominal = float(level)
        macro = coverage[level]["macro_all28"]
        phases = coverage[level]["phase_all28"]
        calib[level] = dict(macro=macro, macro_pass=abs(macro - nominal) <= 0.03,
                            phases=phases, phase_pass=all(abs(p - nominal) <= 0.04 for p in phases))
    by_origin = coverage["0.9"]["by_origin_phase"]
    origin_fail = {o: v for o, v in by_origin.items() if any(abs(p - 0.9) > 0.05 for p in v)}
    calib["origin_90"] = dict(by_origin=by_origin, failing=origin_fail, pass_=not origin_fail)
    out = dict(candidate=args.candidate, reference=args.reference, accuracy=report, calibration=calib)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
