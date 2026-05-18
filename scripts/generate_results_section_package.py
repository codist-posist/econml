"""Build an Overleaf-ready results section from synthesis outputs.

The script consumes the consolidated ``outputs.zip`` created by
``scripts/build_results_synthesis.py`` and writes a compact results package:

* a main LaTeX results section with interpretation;
* an appendix LaTeX file that includes every available figure;
* selected original and derived CSV/LaTeX tables;
* extracted figure files with the same relative paths used by LaTeX.

The goal is not to preserve every diagnostic in the paper body.  The package
keeps the detailed evidence available, while the main text tells the economic
story around the scarcity-rent/adaptation trade-off.
"""

from __future__ import annotations

import argparse
import math
import shutil
import textwrap
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd


POLICY_LABELS = {
    "fixed": "Fixed Taylor",
    "ba": "Natural-rate adjusted",
    "bottleneck": "Bottleneck Taylor",
    "repair_aware": "Repair-aware Taylor",
    "repair_support_aggressive": "Aggressive repair-support rule",
    "discretion": "Discretion",
    "commitment": "Commitment",
}

POLICY_ORDER = [
    "fixed",
    "bottleneck",
    "repair_aware",
    "repair_support_aggressive",
]

PREFERRED_ARCHIVE = {
    "fixed": "fixed_taylor_interior_repair_probe.zip",
    "bottleneck": "bottleneck_interior_repair_probe.zip",
    "repair_aware": "repair_aware_interior_repair_probe.zip",
    "repair_support_aggressive": "repair_support_aggressive_interior_probe.zip",
}

SCENARIO_LABELS = {
    "no_event": "No disruption",
    "D_1x": "Mild disruption",
    "D_3x": "Severe disruption",
    "D_1x_X_lag": "Mild disruption with delayed relief",
    "D_3x_X_lag": "Severe disruption with delayed relief",
    "X_1x": "Relief-only shock",
    "active_ref": "Active-repair reference state",
    "higher_D": "Higher-disruption active state",
    "lower_D": "Lower-disruption active state",
    "higher_A": "Higher-adaptation active state",
    "lower_A": "Lower-adaptation active state",
    "relief_X": "Active state with relief shock",
}


def scenario_label(scenario: object) -> str:
    key = str(scenario)
    return SCENARIO_LABELS.get(key, key.replace("_", " "))


def scenario_from_text(text: str) -> str | None:
    # Longest/most specific names first.
    for key in [
        "D_3x_X_lag",
        "D_1x_X_lag",
        "D_3x",
        "D_1x",
        "X_1x",
        "active_ref",
        "higher_D",
        "lower_D",
        "higher_A",
        "lower_A",
        "relief_X",
        "no_event",
    ]:
        if key in text:
            return key
    return None


def read_csv(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with zf.open(name) as fh:
        return pd.read_csv(fh)


def ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fmt(x: object, digits: int = 3, na: str = "--") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return na
    if not math.isfinite(v):
        return na
    if abs(v) < 1e-14:
        return "0.000"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 10:
        return f"{v:.2f}"
    if abs(v) >= 1:
        return f"{v:.3f}"
    if abs(v) >= 0.01:
        return f"{v:.3f}"
    if abs(v) >= 0.0001:
        return f"{v:.4f}"
    return f"{v:.2e}"


def pct(x: object, digits: int = 1) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(v):
        return "--"
    return f"{100.0 * v:.{digits}f}"


def tex_escape(text: object) -> str:
    s = str(text)
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def detok_path(path: str | Path) -> str:
    return "\\detokenize{" + str(path).replace("\\", "/") + "}"


def canonical_policy_rows(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    rows = []
    for policy in POLICY_ORDER:
        sub = df[(df["policy"] == policy) & (df["scenario"] == scenario)].copy()
        if "calibration" in sub.columns:
            repair_active = sub[sub["calibration"] == "repair_active"]
            if not repair_active.empty:
                sub = repair_active
        preferred = PREFERRED_ARCHIVE.get(policy)
        if preferred and "source_archive" in sub.columns:
            preferred_rows = sub[sub["source_archive"] == preferred]
            if not preferred_rows.empty:
                sub = preferred_rows
        if "checkpoint" in sub.columns:
            saved = sub[sub["checkpoint"].isin(["saved", "best", "final"])]
            if not saved.empty:
                sub = saved
        if not sub.empty:
            rows.append(sub.iloc[0])
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["policy_label"] = out["policy"].map(POLICY_LABELS).fillna(out["policy"])
    return out


def write_table(path: Path, caption: str, label: str, header: list[str], rows: list[list[str]]) -> None:
    align = "l" + "r" * (len(header) - 1)
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def figure_block(path: str, caption: str, label: str, width: str = "0.96\\textwidth") -> str:
    return textwrap.dedent(
        f"""
        \\begin{{figure}}[!htbp]
        \\centering
        \\includegraphics[width={width}]{{{detok_path(path)}}}
        \\caption{{{caption}}}
        \\label{{{label}}}
        \\end{{figure}}
        """
    ).strip()


def extract_figures(zf: zipfile.ZipFile, out_dir: Path) -> list[str]:
    fig_names = [n for n in zf.namelist() if n.lower().endswith(".png")]
    for name in fig_names:
        target = out_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(name) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return sorted(fig_names)


def copy_selected_tables(zf: zipfile.ZipFile, table_dir: Path) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    wanted = [
        "manifest.json",
        "tables/main_text_policy_mechanism_summary.csv",
        "tables/main_text_natural_benchmark_summary.csv",
        "tables/natural_benchmark_policy_stance_summary.csv",
        "tables/conditional_policy_tightening_mechanism_summary.csv",
        "tables/taylor_sss_moments.csv",
        "tables/taylor_sss_regime_frequencies.csv",
        "tables/local_monetary_wedge_summary.csv",
        "tables/local_monetary_wedge_candidates.csv",
        "tables/unified_conditional_active_repair_references.csv",
        "tables/combined_ir_path_summary_long.csv",
    ]
    available = set(zf.namelist())
    for name in wanted:
        if name not in available:
            continue
        target_name = Path(name).name if name.startswith("tables/") else name
        target = table_dir / target_name
        with zf.open(name) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def build_tables(zf: zipfile.ZipFile, table_dir: Path) -> dict[str, pd.DataFrame]:
    policy = ensure_numeric(
        read_csv(zf, "tables/main_text_policy_mechanism_summary.csv"),
        [
            "event_d_policy_rate_ann_pct",
            "event_d_inflation_ann_pct",
            "event_chi",
            "event_cap_pressure_ratio",
            "event_I_A",
            "event_Q_over_threshold",
            "mean_dI_A",
            "binding_freq",
            "repair_positive_freq",
            "repair_interior_freq",
        ],
    )
    natural = ensure_numeric(
        read_csv(zf, "tables/main_text_natural_benchmark_summary.csv"),
        [
            "Y.event",
            "Y_n.event",
            "Y_over_Y_n.event",
            "output_gap_pct_log.event",
            "policy_rate_ann_pct.event",
            "natural_rate_ann_pct.event",
            "policy_rate_gap_ann_pct.event",
            "inflation_ann_pct.event",
            "chi.event",
            "I_A.event",
            "A.post_max",
        ],
    )
    sss_freq = ensure_numeric(
        read_csv(zf, "tables/taylor_sss_regime_frequencies.csv"),
        ["frequency", "n"],
    )
    sss_mom = ensure_numeric(
        read_csv(zf, "tables/taylor_sss_moments.csv"),
        ["mean", "p50", "p95", "p99", "max"],
    )
    local = ensure_numeric(
        read_csv(zf, "tables/local_monetary_wedge_summary.csv"),
        [
            "shock_bp_annualized",
            "baseline_chi_event",
            "baseline_I_A_event",
            "baseline_cap_pressure_event",
            "delta_R_event",
            "delta_I_A_event",
            "mean_delta_I_A",
            "Pr_delta_I_A_lt_0",
        ],
    )
    candidates = ensure_numeric(
        read_csv(zf, "tables/local_monetary_wedge_candidates.csv"),
        ["max_I_A", "max_cap_pressure", "strict_candidates", "relaxed_candidates"],
    )
    roots = ensure_numeric(
        read_csv(zf, "tables/unified_conditional_active_repair_references.csv"),
        [
            "D_multiple",
            "D",
            "A",
            "I_A",
            "Q_A",
            "required_Q_for_steady_A",
            "Q_over_required",
            "Q_over_zero_threshold",
            "cap_pressure_ratio",
            "chi",
            "C",
            "Y",
        ],
    )

    main_d3 = canonical_policy_rows(policy, "D_3x")
    main_d1 = canonical_policy_rows(policy, "D_1x")
    nat_d3 = canonical_policy_rows(natural, "D_3x")

    main_d3.to_csv(table_dir / "main_policy_d3_summary.csv", index=False)
    main_d1.to_csv(table_dir / "main_policy_d1_summary.csv", index=False)
    nat_d3.to_csv(table_dir / "natural_benchmark_d3_summary.csv", index=False)

    freq_pivot = (
        sss_freq[sss_freq["indicator"].isin(["cap_binding", "repair_positive", "repair_interior", "Q_above_threshold"])]
        .pivot_table(index="policy", columns="indicator", values="frequency", aggfunc="first")
        .reindex(POLICY_ORDER)
        .reset_index()
    )
    freq_pivot["policy_label"] = freq_pivot["policy"].map(POLICY_LABELS).fillna(freq_pivot["policy"])
    freq_pivot.to_csv(table_dir / "taylor_sss_frequency_summary.csv", index=False)

    mom = sss_mom[sss_mom["variable"].isin(["I_A", "A", "chi", "cap_pressure_ratio"])].copy()
    mom = mom[mom["policy"].isin(POLICY_ORDER)]
    mom["policy_label"] = mom["policy"].map(POLICY_LABELS).fillna(mom["policy"])
    mom.to_csv(table_dir / "taylor_sss_selected_moments.csv", index=False)

    local_100 = local[local["shock_bp_annualized"].round(6) == 100.0].copy()
    local_100["policy_label"] = local_100["policy"].map(POLICY_LABELS).fillna(local_100["policy"])
    local_100["delta_I_A_pct_of_baseline"] = 100.0 * local_100["delta_I_A_event"] / local_100["baseline_I_A_event"].replace(0.0, pd.NA)
    local_100.to_csv(table_dir / "local_monetary_wedge_summary_with_pct.csv", index=False)

    root_rows = roots[roots["source_entry"].str.contains("roots", na=False)].copy()
    root_rows.to_csv(table_dir / "conditional_active_repair_roots.csv", index=False)

    # LaTeX tables.
    write_table(
        table_dir / "main_policy_d3_summary.tex",
        "Severe bottleneck shock under repair-active rule calibrations.",
        "tab:main-policy-d3-summary",
        [
            "Policy",
            "$\\chi_0$",
            "Cap pressure",
            "$I^A_0$",
            "$Q^A/\\bar Q$",
            "IR repair freq. (\\%)",
            "IR interior freq. (\\%)",
        ],
        [
            [
                tex_escape(row["policy_label"]),
                fmt(row.get("event_chi")),
                fmt(row.get("event_cap_pressure_ratio")),
                fmt(row.get("event_I_A")),
                fmt(row.get("event_Q_over_threshold")),
                pct(row.get("repair_positive_freq")),
                pct(row.get("repair_interior_freq")),
            ]
            for _, row in main_d3.iterrows()
        ],
    )

    write_table(
        table_dir / "natural_benchmark_d3_summary.tex",
        "Natural-benchmark decomposition for the severe bottleneck shock.",
        "tab:natural-benchmark-d3-summary",
        [
            "Policy",
            "$Y_0/Y^n_0$",
            "Output gap (\\%)",
            "Policy rate (\\%)",
            "Rate gap (\\%)",
            "Inflation (\\%)",
            "$A_{\\max}$",
        ],
        [
            [
                tex_escape(row["policy_label"]),
                fmt(row.get("Y_over_Y_n.event")),
                fmt(row.get("output_gap_pct_log.event")),
                fmt(row.get("policy_rate_ann_pct.event")),
                fmt(row.get("policy_rate_gap_ann_pct.event")),
                fmt(row.get("inflation_ann_pct.event")),
                fmt(row.get("A.post_max")),
            ]
            for _, row in nat_d3.iterrows()
        ],
    )

    write_table(
        table_dir / "sss_frequency_summary.tex",
        "Stochastic steady-state regime frequencies in the Taylor-rule simulations.",
        "tab:sss-frequency-summary",
        [
            "Policy",
            "Cap binding (\\%)",
            "Repair positive (\\%)",
            "Interior repair (\\%)",
            "$Q^A$ above threshold (\\%)",
        ],
        [
            [
                tex_escape(row["policy_label"]),
                pct(row.get("cap_binding")),
                pct(row.get("repair_positive")),
                pct(row.get("repair_interior")),
                pct(row.get("Q_above_threshold")),
            ]
            for _, row in freq_pivot.iterrows()
            if isinstance(row.get("policy_label"), str)
        ],
    )

    local_rows = []
    for _, row in local_100.iterrows():
        if row.get("policy") not in ["fixed", "bottleneck"]:
            continue
        local_rows.append(
            [
                tex_escape(row["policy_label"]),
                tex_escape(scenario_label(row.get("scenario", ""))),
                fmt(row.get("baseline_I_A_event"), 4),
                fmt(row.get("baseline_chi_event")),
                fmt(row.get("baseline_cap_pressure_event")),
                fmt(row.get("delta_I_A_event"), 4),
                fmt(row.get("delta_I_A_pct_of_baseline")),
            ]
        )
    write_table(
        table_dir / "local_monetary_wedge_summary.tex",
        "Local effect of a 100 bp annualized monetary tightening in repair-active states.",
        "tab:local-monetary-wedge-summary",
        [
            "Policy",
            "Scenario",
            "$I^A_0$",
            "$\\chi_0$",
            "Cap pressure",
            "$\\Delta I^A_0$",
            "\\% of $I^A_0$",
        ],
        local_rows,
    )

    root_for_table = root_rows.head(5)
    write_table(
        table_dir / "conditional_active_repair_root.tex",
        "Static reference states where active repair is feasible.",
        "tab:conditional-active-repair-root",
        [
            "$D$ multiple",
            "$A$",
            "$I^A$",
            "$Q^A$",
            "$Q^A/\\bar Q$",
            "Cap pressure",
            "$\\chi$",
            "$Y$",
        ],
        [
            [
                fmt(row.get("D_multiple")),
                fmt(row.get("A")),
                fmt(row.get("I_A")),
                fmt(row.get("Q_A")),
                fmt(row.get("Q_over_zero_threshold")),
                fmt(row.get("cap_pressure_ratio")),
                fmt(row.get("chi")),
                fmt(row.get("Y")),
            ]
            for _, row in root_for_table.iterrows()
        ],
    )

    return {
        "policy": policy,
        "natural": natural,
        "main_d3": main_d3,
        "main_d1": main_d1,
        "nat_d3": nat_d3,
        "sss_freq": freq_pivot,
        "sss_mom": mom,
        "local_100": local_100,
        "candidates": candidates,
        "roots": root_rows,
    }


def one_row(df: pd.DataFrame, policy: str) -> pd.Series:
    sub = df[df["policy"] == policy]
    if sub.empty:
        return pd.Series(dtype=object)
    return sub.iloc[0]


def value(df: pd.DataFrame, policy: str, col: str) -> float:
    row = one_row(df, policy)
    if row.empty:
        return float("nan")
    try:
        return float(row[col])
    except Exception:
        return float("nan")


def build_main_tex(out_dir: Path, data: dict[str, pd.DataFrame]) -> str:
    main_d3 = data["main_d3"]
    nat_d3 = data["nat_d3"]
    freq = data["sss_freq"]
    local_100 = data["local_100"]
    roots = data["roots"]

    fixed_i = value(main_d3, "fixed", "event_I_A")
    fixed_chi = value(main_d3, "fixed", "event_chi")
    fixed_a_max = value(nat_d3, "fixed", "A.post_max")
    bottleneck_chi = value(main_d3, "bottleneck", "event_chi")
    bottleneck_i = value(main_d3, "bottleneck", "event_I_A")
    repair_aware_i = value(main_d3, "repair_aware", "event_I_A")
    fixed_repair_freq = value(freq, "fixed", "repair_positive")
    bottleneck_repair_freq = value(freq, "bottleneck", "repair_positive")
    aware_repair_freq = value(freq, "repair_aware", "repair_positive")
    root = roots.iloc[0] if not roots.empty else pd.Series(dtype=object)

    fixed_local = local_100[(local_100["policy"] == "fixed") & (local_100["scenario"] == "D_3x")]
    bottleneck_local = local_100[(local_100["policy"] == "bottleneck") & (local_100["scenario"] == "D_3x")]
    fixed_local_drop = abs(float(fixed_local["delta_I_A_pct_of_baseline"].iloc[0])) if not fixed_local.empty else float("nan")
    bottleneck_local_drop = abs(float(bottleneck_local["delta_I_A_pct_of_baseline"].iloc[0])) if not bottleneck_local.empty else float("nan")

    main_figs = "\n\n".join(
        [
            figure_block(
                "figures/policy_comparison/taylor_policy_comparison_D_3x.png",
                "Main policy comparison after the severe disruption.  The figure is the core visual evidence for the mechanism: the fixed Taylor rule leaves a large scarcity rent and activates repair, the bottleneck rule compresses current scarcity and reduces the size and persistence of repair, while the repair-aware rules eliminate repair because the private value of adaptation does not cross the KKT threshold.",
                "fig:policy-comparison-d3",
            ),
            figure_block(
                "figures/policy_comparison/taylor_policy_comparison_D_1x.png",
                "Mild disruption.  This figure shows that the mechanism is state dependent.  The cap pressure is close to the binding margin, the scarcity rent is smaller, and repair is much less robust than under the severe disruption.",
                "fig:policy-comparison-d1",
            ),
            figure_block(
                "figures/natural_benchmark/natural_benchmark_taylor_D_3x.png",
                "Natural-benchmark decomposition for the severe disruption.  The graph separates the physical fall in flexible-price output from the monetary stance.  The output gap and policy-rate gap show whether a rule is stabilizing current prices and activity or leaving enough current scarcity for private repair incentives to remain active.",
                "fig:natural-benchmark-d3",
            ),
            figure_block(
                "figures/local_monetary_wedge/local_monetary_wedge_fixed_D_3x_100bp.png",
                "Local 100 bp monetary wedge around a repair-active fixed-rule state.  Because the test holds the physical bottleneck state fixed, it isolates the KKT channel from the financing wedge: a higher nominal rate raises the private cost of repair and lowers $I^A$ at the margin.",
                "fig:local-wedge-fixed-d3",
            ),
            figure_block(
                "figures/local_monetary_wedge/local_monetary_wedge_bottleneck_D_3x_100bp.png",
                "Local 100 bp monetary wedge around a repair-active bottleneck-rule state.  The sign is the same as under the fixed rule, but the effect is evaluated at a state with lower current scarcity and lower repair intensity.",
                "fig:local-wedge-bottleneck-d3",
            ),
            figure_block(
                "figures/relief_sensitivity/relief_sensitivity_fixed_D_3x.png",
                "Relief-channel sensitivity under the fixed rule.  Relief lowers future disruption pressure, but in these repair-active calibrations it does not mechanically eliminate repair; the value of repair depends on the whole path of scarcity and expected future benefits.",
                "fig:relief-fixed-d3",
            ),
            figure_block(
                "figures/repair_threshold_sensitivity/repair_cost_sensitivity_fixed_D_3x.png",
                "Repair-cost sensitivity under the fixed rule.  The graph makes clear that repair is a threshold object: a relatively small change in the repair threshold can move the economy between no repair, interior repair, and the upper repair bound.",
                "fig:repair-cost-fixed-d3",
            ),
            figure_block(
                "figures/optimal_policy/optimal_baseline_no_repair_D_3x.png",
                "Baseline optimal-policy experiment.  The selected discretion and commitment solutions generally remain on the no-repair branch.  Economically, the policymaker stabilizes the current bottleneck enough that private adaptation does not cross its threshold.",
                "fig:optimal-no-repair-d3",
            ),
            figure_block(
                "figures/optimal_policy/optimal_active_region_diagnostic_D_3x.png",
                "Targeted active-region optimal-policy exercise.  Even when the state distribution is shifted toward the active-repair reference region, discretion and commitment do not robustly keep repair active.  The figure explains why the paper separates the implementable-rule mechanism from the full optimal-policy benchmark.",
                "fig:optimal-active-region-d3",
            ),
        ]
    )

    s = textwrap.dedent(
        r"""
        % Auto-generated by scripts/generate_results_section_package.py
        % Suggested preamble additions:
        % \usepackage{{graphicx}}
        % \usepackage{{booktabs}}
        % \usepackage{{float}}
        % \usepackage{{placeins}}

        \section{{Quantitative results: scarcity, repair, and policy design}}
        \label{{sec:quant-results}}

        The quantitative exercise has one organizing question.  When a disruption to a critical imported input is severe enough to make the physical import cap bind, monetary policy affects not only current demand and inflation but also the private incentive to adapt the production structure.  The model therefore contains two margins that are easy to confuse.  The first is the current scarcity margin, summarized by the scarcity rent $\chi_t$, the zero-rent import pressure, and the cap-pressure ratio.  The second is the dynamic repair margin, summarized by the value of repair $Q^A_t$ and the investment flow $I^A_t$ in the law of motion
        \[
            A_{t+1}=(1-\delta_A)A_t + I^A_t .
        \]
        The variable $I^A_t$ is not a percentage point of the policy rate and it is not a percent of GDP.  It is a quarterly flow into the adaptation stock.  In the repair-active calibration used for the central mechanism figures the upper bound is $I^A_{\max}=0.03$, so an observation $I^A_t=0.03$ means that the economy is adding three hundredths of a full adaptation unit in that quarter.  If it remained at the bound for a year, the gross addition to the stock before depreciation would be about $0.12$.  The accumulated stock $A_t$ then lowers import intensity through
        \[
            \mu(A_t)=\mu_{\min}+(\mu_0-\mu_{\min})\exp(-\kappa_A A_t).
        \]
        Thus the economically relevant object is not just the impact value of $I^A_t$, but the resulting path of $A_t$ and the decline in future exposure to the bottleneck.

        \paragraph{Scenario notation.}  The figures and data files use a compact scenario shorthand.  In the text, ``mild disruption'' corresponds to the file label \texttt{D\_1x}; ``severe disruption'' corresponds to \texttt{D\_3x}; and ``severe disruption with delayed relief'' corresponds to \texttt{D\_3x\_X\_lag}.  The suffix $X$ denotes a relief shock, so ``relief-only shock'' corresponds to \texttt{X\_1x}.  These names are kept in file paths for reproducibility, but the economic interpretation below uses the descriptive labels.

        The baseline calibration is deliberately conservative about repair.  In the selected discretion and commitment solutions, repair is typically inactive.  This is an economic feature of the benchmark rather than a mechanical plotting issue: the no-repair branch survives several variants, including versions with the repair flow written explicitly as a network output.  To study the mechanism itself, the rule-based exercises therefore use a repair-active calibration with a stronger physical bottleneck and lower repair cost.  This is not a change in the model's logic; it moves the simulated economy to the part of the state space where the KKT condition for repair is close enough to bind.  A useful reference state is reported in Table~\ref{{tab:conditional-active-repair-root}}: at a three-times disruption, $D=0.75$, and $A\simeq {fmt(root.get("A"))}$, the static calculation supports an interior repair flow $I^A\simeq {fmt(root.get("I_A"))}$ with $Q^A$ exactly at the required continuation value.  This reference is important because it shows that the active-repair branch exists economically even when the full optimal-policy benchmark often prefers not to use it.

        \input{{tables/conditional_active_repair_root.tex}}

        It is useful to state in advance which results confirm the mechanism and which do not.  The mechanism is confirmed in three places.  First, in the repair-active Taylor-rule experiments, a binding cap raises $\chi_t$, pushes $Q^A_t$ above the repair threshold, and produces positive $I^A_t$.  Second, in the local monetary-wedge exercise, a higher policy rate lowers $I^A_t$ conditional on being in a binding-cap, active-repair state.  Third, in the stochastic simulations, repair appears repeatedly under rules that preserve the scarcity signal.  The mechanism is not confirmed as an unconditional prediction of the baseline Ramsey/discretion problem.  In the main optimal-policy solutions the repair value remains below the KKT threshold and the economy stays on the no-repair branch.  This split is central to the interpretation: repair is a real state-contingent margin, not an automatic response to every disruption.

        \subsection{{Policy rules and what they are meant to test}}
        \label{{sec:policy-rule-map}}

        The rule-based exercises compare several policy formulas because each one removes a different ambiguity.  The fixed Taylor rule is the benchmark: it responds to inflation and activity but has no direct term for the import cap or repair.  The natural-rate adjusted rule replaces the constant intercept with the natural-rate benchmark, so it asks whether part of the result is just a mismeasured natural rate during the supply disruption.  The bottleneck Taylor rule adds a direct cap-pressure adjustment.  In the implementation used here, high cap pressure lowers the rule's gross rate relative to the standard Taylor component; it is therefore a bottleneck-accommodative rule, not a rule that leans mechanically against the current scarcity rent.  The repair-aware rule goes one step further and makes the accommodative term depend on both cap pressure and the repair margin $Q^A_t/(\Omega^A_t p^A_t\psi_A)$.  The aggressive repair-support rule is a stronger version of the same idea.

        The reason these rules can behave differently is that the model has two policy-relevant margins.  Lowering the rate can support current demand and reduce the financing cost of repair, but it can also change output, cap pressure, and the private value of future adaptation.  Raising or lowering rates is therefore not enough to predict repair by itself.  What matters is where the rule leaves the economy relative to the repair threshold.

        \subsection{{The main policy comparison}}
        \label{{sec:main-policy-comparison}}

        Figure~\ref{{fig:policy-comparison-d3}} and Table~\ref{{tab:main-policy-d3-summary}} are the central results.  A severe disruption raises the desired use of imported inputs above the physical cap.  Under the fixed Taylor rule, the scarcity rent reaches $\chi_0\simeq {fmt(fixed_chi)}$ and the repair flow jumps to the upper bound $I^A_0={fmt(fixed_i)}$.  This is the direct bottleneck-repair mechanism: the cap is costly today, but precisely because it is costly, the private value of reducing future import exposure is high.  In the same experiment the accumulated adaptation stock reaches about $A_{\max}\simeq {fmt(fixed_a_max)}$, so the impact repair flow is not a cosmetic response.  It cumulates into a meaningful change in the production structure.

        The bottleneck Taylor rule changes the composition of the response.  It lowers current scarcity relative to the fixed rule: the rent falls to $\chi_0\simeq {fmt(bottleneck_chi)}$, but repair still starts at the upper bound $I^A_0={fmt(bottleneck_i)}$ in the severe-disruption state.  The repair path is shorter and less persistent.  In stochastic simulations, repair is positive in about {pct(bottleneck_repair_freq)} percent of periods under the bottleneck rule, compared with {pct(fixed_repair_freq)} percent under the fixed rule.  The bottleneck rule therefore illustrates the trade-off rather than eliminating it: stabilizing current scarcity weakens, but does not fully destroy, the incentive to adapt.

        The repair-aware and aggressive repair-support rules deliver the sharpest lesson.  They sound more targeted, but in this calibration they do not activate repair in the severe-disruption comparison; $I^A_0$ is approximately {fmt(repair_aware_i)} under the repair-aware rule.  The reason is that these rules compress current scarcity and output enough to move $Q^A_t$ below the KKT threshold.  A rule can be ``aware'' of repair and still remove the price signal that makes private repair privately valuable.  The mechanism is therefore not simply that lower rates always produce more adaptation or that higher rates always prevent it.  It is a threshold interaction between the financing wedge and the private value of future bottleneck relief.

        \input{{tables/main_policy_d3_summary.tex}}

        The mild-disruption figure, Figure~\ref{{fig:policy-comparison-d1}}, is useful for interpretation.  When the shock is smaller, cap pressure is near the binding margin and the rent is much lower.  Repair is not a robust response across rules.  This state dependence is exactly what the model was designed to capture: adaptation should not happen mechanically after every disturbance.  It appears when the expected benefit of reducing import intensity is large enough to cover the convex repair cost and the financing wedge.

        The panels in the policy-comparison figures should be read as a sequence.  The policy-rate and inflation panels describe the nominal stabilization response.  The output-gap panel shows the real allocation relative to the flexible-price benchmark.  The scarcity-rent and cap-pressure panels show whether the import constraint is merely present or economically binding.  The $Q^A_t$ and repair-activation panels show whether the private value of adaptation crosses the KKT threshold.  Finally, the $I^A_t$ and $A_t$ panels distinguish a one-period repair spike from a persistent change in exposure.  The central trade-off is visible when the first group of panels improves current stabilization while the second group moves repair below threshold.

        \subsection{{Natural benchmark and the meaning of the output gap}}
        \label{{sec:natural-benchmark-results}}

        The natural benchmark is a measurement device.  It answers: what would output look like under the same physical disruption if prices were flexible and the natural-rate convention were used?  Table~\ref{{tab:natural-benchmark-d3-summary}} and Figure~\ref{{fig:natural-benchmark-d3}} show that the same physical bottleneck can coexist with very different policy gaps.

        Under the fixed Taylor rule, the severe disruption has $Y_0/Y^n_0\simeq {fmt(value(nat_d3, "fixed", "Y_over_Y_n.event"))}$ and a positive log output gap of about {fmt(value(nat_d3, "fixed", "output_gap_pct_log.event"))} percent.  This does not mean the shock is expansionary.  It means that the realized allocation is above the flexible-price benchmark at impact because nominal policy accommodates the current bottleneck enough to leave high inflation and high scarcity rents.  Under the bottleneck rule, the same physical shock produces a negative output gap of about {fmt(value(nat_d3, "bottleneck", "output_gap_pct_log.event"))} percent.  Under the repair-aware rules the gap is even more negative.  This is the current stabilization side of the trade-off: stronger anti-scarcity policy stabilizes the current allocation relative to the fixed rule, but it also lowers the private reward from structural adaptation.

        \input{{tables/natural_benchmark_d3_summary.tex}}

        Annualized inflation and policy rates in these tables are reported as $(\Pi_t^4-1)\times 100$ and $(R_t^4-1)\times 100$.  They are therefore annualized percent rates, not quarterly log points.  The output gap is a log percentage gap relative to the natural benchmark.  The repair flow $I^A_t$ remains in adaptation-stock units, so it should be read separately from the nominal variables.

        \subsection{{The local monetary wedge}}
        \label{{sec:local-monetary-wedge}}

        The policy-comparison figures combine several channels: current demand, inflation, cap pressure, repair values, and future states.  The local monetary-wedge test isolates one narrow piece of the mechanism.  Starting from states in which the cap is binding and repair is active, it applies a 100 bp annualized tightening while holding the physical bottleneck state fixed.  Table~\ref{{tab:local-monetary-wedge-summary}} reports the impact response.

        The sign is the one predicted by the repair KKT condition.  In the fixed-rule severe-disruption state, the tightening lowers the repair flow by about {fmt(fixed_local_drop)} percent of its baseline impact value.  In the bottleneck-rule severe-disruption state, the local decline is about {fmt(bottleneck_local_drop)} percent.  The magnitude is deliberately small because this is a one-step local wedge, not a retrained global policy experiment.  Its role is identification: conditional on being in a repair-active, binding-cap state, the financing channel has the correct sign.  Higher $R_t$ raises $\Omega^A_t=(1-\vartheta_A)+\vartheta_A R_t$, which raises the private cost of repair and pushes $I^A_t$ down at the margin.

        \input{{tables/local_monetary_wedge_summary.tex}}

        Figures~\ref{{fig:local-wedge-fixed-d3}} and~\ref{{fig:local-wedge-bottleneck-d3}} also show why actual imports need not fall much when the cap is binding.  Desired zero-rent import pressure reacts, but actual imports are pinned by the physical cap.  The adjustment shows up in rents and repair incentives rather than in a large contemporaneous change in imported quantities.

        \subsection{{Stochastic steady-state evidence}}
        \label{{sec:sss-results}}

        The impulse responses are not the only evidence for the mechanism.  Table~\ref{{tab:sss-frequency-summary}} summarizes long stochastic simulations of the Taylor-rule economies.  The fixed rule spends about {pct(value(freq, "fixed", "cap_binding"))} percent of periods in a binding-cap state and about {pct(fixed_repair_freq)} percent of periods with positive repair.  The bottleneck rule has a similar binding-cap frequency, about {pct(value(freq, "bottleneck", "cap_binding"))} percent, but positive repair appears in only about {pct(bottleneck_repair_freq)} percent of periods.  The repair-aware and aggressive rules generate no repair in these simulations.

        \input{{tables/sss_frequency_summary.tex}}

        The stochastic moments make the same point in levels.  Under the fixed rule the mean repair flow is positive and the upper bound is frequently reached in the right tail.  Under the bottleneck rule, the repair distribution is much thinner.  Under the repair-aware rules it collapses to zero.  This rules out the interpretation that the main figures are merely hand-picked event dates.  The active-repair margin appears in the ergodic behavior of some rules and disappears under others.

        The stochastic steady-state histograms are kept in Appendix~\ref{{app:additional-results-figures}} because they are distributional diagnostics rather than the main narrative figures.  They are nevertheless part of the evidence.  They show whether the model visits binding-cap and repair-active states repeatedly after many simulated shocks, rather than only in the deterministic event windows used for the main figures.

        \subsection{{Relief, repair costs, and why the branch is fragile}}
        \label{{sec:sensitivity-results}}

        The relief experiments ask whether future relief shocks crowd out adaptation.  The answer is nuanced.  Relief lowers future external pressure, so it can reduce the expected benefit of repair.  But in the repair-active rule calibrations, stronger relief does not mechanically eliminate repair.  In several fixed-rule and bottleneck-rule comparisons, repair remains active and can even be slightly larger because the path of scarcity, output, and repair values changes jointly.  Figure~\ref{{fig:relief-fixed-d3}} illustrates this point.  Relief is therefore not the sole explanation for the no-repair outcomes in the optimal-policy runs.

        Repair-cost sensitivity is more direct.  Figure~\ref{{fig:repair-cost-fixed-d3}} shows that repair is a threshold phenomenon.  The object that matters is not $Q^A_t$ alone, but $Q^A_t$ relative to the KKT threshold $\Omega^A_t p^A_t\psi_A$.  When the threshold is low enough and the bottleneck is severe enough, repair activates.  When policy or parameters move the economy below the threshold, the projection sends $I^A_t$ back to zero.  This also explains why the main text relies on stable policy comparisons rather than isolated positive-repair training episodes: a positive repair flow is economically meaningful only when it is supported by the surrounding scarcity, repair-value, and financing-margin paths.

        The additional sensitivity figures in Appendix~\ref{{app:additional-results-figures}} separate these two robustness questions.  The relief figures vary the expected external relief channel.  The repair-threshold figures vary the private cost side of the KKT condition.  The first set tells us that anticipated relief is not by itself enough to explain the disappearance of repair.  The second set tells us that the repair branch is genuinely threshold-sensitive.

        \subsection{{Discretion and commitment}}
        \label{{sec:optimal-policy-results}}

        The optimal-policy exercises are included because they answer a different question from the implementable Taylor-rule experiments.  The Taylor rules ask what happens under simple policy formulas.  Discretion and commitment ask whether a policymaker that internalizes the repair margin chooses to preserve it.  In the baseline discretion and commitment solutions, the answer is mostly no: repair remains inactive.  We then tried several ways to make sure this was not merely an artefact of how repair was represented.  We made $I^A_t$ an explicit network output, added the repair KKT condition directly to the private residual system, targeted the training distribution around a static active-repair reference state, and checked no-relief variants.  These exercises still did not deliver a stable optimal-policy allocation with robust positive repair.

        The economic reading is that the central bank does not choose the external shock or the physical cap.  It affects scarcity indirectly through demand, output, inflation dynamics, and the repair financing wedge.  By changing the equilibrium path of production and demand for the intermediate input, policy changes desired import pressure and hence the scarcity rent when the cap binds.  By changing $R_t$, it also changes $\Omega^A_t$ and the repair threshold.  In the main optimal-policy solutions these effects jointly keep $Q^A_t$ below the threshold needed for positive repair.  In other words, the central bank can reduce current scarcity and current distortions, but doing so removes the private return that would have made repair attractive.

        This is an important result for interpretation.  The paper should not claim that the Ramsey or discretion solution robustly invests in adaptation under the baseline calibration.  The stronger statement supported by the evidence is more precise: the model contains a real repair-scarcity mechanism; simple rules can either preserve or destroy that mechanism; and full optimal-policy training tends to select a no-repair branch unless the economy is deliberately moved into a repair-active region.  In economic terms, the planner's problem finds it attractive to stabilize current distortions enough that the private repair threshold is not crossed.  That does not invalidate the mechanism; it says the mechanism is conditional on being in the part of the state space where scarcity rents make private adaptation valuable.

        Figures~\ref{{fig:optimal-no-repair-d3}} and~\ref{{fig:optimal-active-region-d3}} document this distinction.  They should be read as a negative benchmark: the no-repair branch is not a failure to plot the right variable, but a stable outcome of the optimal-policy exercises considered here.  The appendix reports the intermediate optimal-policy variants: no-relief probes, explicit-$I^A$ runs, active-region targeting, and multi-solution comparisons.  Their role is to show why the main text does not overstate the optimal-policy result.

        \subsection{{Summary of the economic story}}
        \label{{sec:results-summary}}

        The results support a layered interpretation.  First, a critical-input disruption creates a scarcity rent only when desired input use exceeds the physical cap.  Second, that rent is not just a current distortion; it is also a signal that makes future import-intensity reduction valuable.  Third, policy rules differ in whether they preserve or compress this signal.  The fixed Taylor rule leaves large current scarcity and therefore generates strong repair.  The bottleneck rule reduces current scarcity and still permits some repair, but on a smaller scale.  The repair-aware rules stabilize current scarcity so aggressively that the private repair value falls below the threshold and repair disappears.  Fourth, the local monetary-wedge test confirms the sign of the financing channel: conditional on active repair, a monetary tightening reduces repair investment at the margin.  Finally, the optimal-policy experiments show that the active-repair branch is fragile in the full Ramsey/discretion system.  The central trade-off is therefore not a universal prediction that all bottlenecks create repair.  It is a conditional mechanism: when the economy is in a binding-cap, high-scarcity state, monetary policy can lower current inflation and scarcity while weakening the incentive to build future resilience.

        {main_figs}
        """
    ).strip() + "\n"
    replacements = {
        "{fmt(root.get(\"A\"))}": fmt(root.get("A")),
        "{fmt(root.get(\"I_A\"))}": fmt(root.get("I_A")),
        "{fmt(fixed_chi)}": fmt(fixed_chi),
        "{fmt(fixed_i)}": fmt(fixed_i),
        "{fmt(fixed_a_max)}": fmt(fixed_a_max),
        "{fmt(bottleneck_chi)}": fmt(bottleneck_chi),
        "{fmt(bottleneck_i)}": fmt(bottleneck_i),
        "{fmt(repair_aware_i)}": fmt(repair_aware_i),
        "{pct(bottleneck_repair_freq)}": pct(bottleneck_repair_freq),
        "{pct(fixed_repair_freq)}": pct(fixed_repair_freq),
        "{fmt(value(nat_d3, \"fixed\", \"Y_over_Y_n.event\"))}": fmt(value(nat_d3, "fixed", "Y_over_Y_n.event")),
        "{fmt(value(nat_d3, \"fixed\", \"output_gap_pct_log.event\"))}": fmt(value(nat_d3, "fixed", "output_gap_pct_log.event")),
        "{fmt(value(nat_d3, \"bottleneck\", \"output_gap_pct_log.event\"))}": fmt(value(nat_d3, "bottleneck", "output_gap_pct_log.event")),
        "{fmt(fixed_local_drop)}": fmt(fixed_local_drop),
        "{fmt(bottleneck_local_drop)}": fmt(bottleneck_local_drop),
        "{pct(value(freq, \"fixed\", \"cap_binding\"))}": pct(value(freq, "fixed", "cap_binding")),
        "{pct(value(freq, \"bottleneck\", \"cap_binding\"))}": pct(value(freq, "bottleneck", "cap_binding")),
        "{main_figs}": main_figs,
    }
    # The long block above is intentionally not an f-string: LaTeX math uses
    # many braces, and keeping it as plain text makes it much safer to edit.
    s = s.replace("{{", "{").replace("}}", "}")
    for old, new in replacements.items():
        s = s.replace(old, str(new))
    return s


def figure_caption(rel: str) -> str:
    name = Path(rel).stem
    pretty = name.replace("_", " ")
    scen_key = scenario_from_text(name)
    scen = scenario_label(scen_key) if scen_key is not None else None
    if "policy_comparison" in rel and "taylor_policy_comparison" in rel:
        return f"Taylor-rule policy comparison for {tex_escape(scen or pretty)}.  The panels compare current nominal stabilization, scarcity rents, cap pressure, repair value, repair investment, and the accumulated adaptation stock."
    if "all_saved_policy_diagnostic" in rel:
        return f"Additional policy comparison for {tex_escape(scen or pretty)}.  This broader figure shows that the main pattern is not driven by a single plotted series."
    if "natural_benchmark" in rel:
        return f"Natural-benchmark decomposition for {tex_escape(scen or pretty)}.  These panels compare realized output and rates with the flexible-price benchmark and clarify whether policy is tightening or accommodating relative to the physical shock."
    if "local_monetary_wedge" in rel:
        tail = f" for {scen}" if scen else ""
        return f"Local monetary-wedge exercise{tex_escape(tail)}.  The graph applies a small policy-rate wedge around a repair-active state to isolate the financing channel in the repair KKT condition."
    if "relief_sensitivity" in rel:
        tail = f" for {scen}" if scen else ""
        return f"Relief-channel sensitivity{tex_escape(tail)}.  The graph compares paths when the expected relief process is changed, testing whether repair disappears simply because future relief is anticipated."
    if "repair_threshold_sensitivity" in rel:
        tail = f" for {scen}" if scen else ""
        return f"Repair-cost and repair-threshold sensitivity{tex_escape(tail)}.  The graph shows how close the economy is to the projection threshold for positive adaptation."
    if "optimal_policy" in rel:
        tail = f" for {scen}" if scen else ""
        return f"Optimal-policy exercise{tex_escape(tail)}.  The figure documents whether discretion or commitment stays on a no-repair branch or can sustain an active-repair path when the state space is shifted toward that region."
    if "taylor_sss_histograms" in rel:
        return "Stochastic steady-state histograms for the corresponding Taylor rule.  These distributions show how often the economy visits binding-cap and repair-active regions beyond deterministic impulse responses."
    if "bottleneck_interior_repair" in rel:
        tail = f" for {scen}" if scen else ""
        return f"Bottleneck-rule interior-repair figure{tex_escape(tail)} under the repair-active calibration."
    if "repair_aware" in rel:
        tail = f" for {scen}" if scen else ""
        return f"Repair-aware rule figure{tex_escape(tail)} under the repair-active calibration."
    if "aggressive_repair_support" in rel:
        tail = f" for {scen}" if scen else ""
        return f"Aggressive repair-support rule figure{tex_escape(tail)} under the repair-active calibration."
    return f"Diagnostic figure: {tex_escape(pretty)}."


def build_appendix_tex(fig_names: list[str]) -> str:
    blocks = [
        "% Auto-generated figure appendix.",
        "\\section{Additional results figures}",
        "\\label{app:additional-results-figures}",
        "This appendix collects all figures extracted from the consolidated results archive.  The main text uses a subset of these figures; the remaining figures provide additional policy comparisons, sensitivity checks, stochastic steady-state distributions, natural-benchmark decompositions, and local monetary-wedge exercises.",
        "File paths retain the compact scenario names used in the computational output: \\texttt{D\\_1x} is the mild disruption, \\texttt{D\\_3x} is the severe disruption, \\texttt{D\\_3x\\_X\\_lag} is the severe disruption followed by delayed relief, and \\texttt{X\\_1x} is a relief-only shock.  Figure captions use the descriptive names.",
    ]
    for i, rel in enumerate(fig_names, start=1):
        label = "fig:appendix-results-" + str(i)
        blocks.append(figure_block(rel, figure_caption(rel), label))
        if i % 6 == 0:
            blocks.append("\\FloatBarrier")
    return "\n\n".join(blocks) + "\n"


def build_readme(out_dir: Path, zip_path: Path) -> str:
    return textwrap.dedent(
        f"""
        # Results Section Overleaf Package

        Generated from `result_learning/outputs.zip`.

        Main files:

        - `critical_input_results_section.tex`: main results section with interpretation.
        - `critical_input_results_appendix_figures.tex`: appendix file including every extracted PNG figure.
        - `figures/`: figures with relative paths already used in the TeX files.
        - `tables/`: selected original CSV summaries plus compact derived LaTeX tables.

        Suggested Overleaf usage:

        1. Upload the contents of this folder, preserving the `figures/` and `tables/` directories.
        2. Add these packages if they are not already in the preamble:

           ```latex
           \\usepackage{{graphicx}}
           \\usepackage{{booktabs}}
           \\usepackage{{float}}
           \\usepackage{{placeins}}
           ```

        3. Insert the main section with:

           ```latex
           \\input{{critical_input_results_section.tex}}
           ```

        4. Insert the figure appendix, if needed, with:

           ```latex
           \\input{{critical_input_results_appendix_figures.tex}}
           ```

        A zip copy was written to:

        `{zip_path}`
        """
    ).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="result_learning/outputs.zip", help="Consolidated outputs.zip path")
    parser.add_argument("--output", default="result_learning/results_section_overleaf", help="Output package directory")
    parser.add_argument("--no-zip", action="store_true", help="Do not create a zip archive of the package")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    out_dir = Path(args.output)
    table_dir = out_dir / "tables"

    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Zip archive failed integrity check at {bad}")
        fig_names = extract_figures(zf, out_dir)
        copy_selected_tables(zf, table_dir)
        data = build_tables(zf, table_dir)

    main_tex = build_main_tex(out_dir, data)
    (out_dir / "critical_input_results_section.tex").write_text(main_tex, encoding="utf-8")
    appendix_tex = build_appendix_tex(fig_names)
    (out_dir / "critical_input_results_appendix_figures.tex").write_text(appendix_tex, encoding="utf-8")

    zip_out = out_dir.with_name(out_dir.name + "_package.zip")
    (out_dir / "README.md").write_text(build_readme(out_dir, zip_out), encoding="utf-8")

    if not args.no_zip:
        if zip_out.exists():
            zip_out.unlink()
        shutil.make_archive(str(zip_out.with_suffix("")), "zip", out_dir)

    print(f"Wrote package folder: {out_dir}")
    print(f"Figures extracted: {len(fig_names)}")
    print(f"Main TeX: {out_dir / 'critical_input_results_section.tex'}")
    print(f"Appendix TeX: {out_dir / 'critical_input_results_appendix_figures.tex'}")
    if not args.no_zip:
        print(f"Zip package: {zip_out}")
        print(f"Zip size MB: {zip_out.stat().st_size / 1e6:.2f}")


if __name__ == "__main__":
    main()
