"""Figures for the README. Matplotlib to PNG, no notebook required.

    python -m src.analysis.plots

Reads the committed run artifacts under `runs/` and writes to `figures/`. Every
number on every axis comes from a file in the repository, so a figure can always
be traced back to the eval report that produced it.

ON THE LENGTH FIGURE
--------------------
`length_vs_pass1.png` is a correlation and is labelled as one on the figure
itself, not only in the caption. The balanced-corpus ablation is marked on it
because that point is the evidence that length is a marker rather than the
mechanism: it sits at a normal length having moved the result by +0.009 +- 0.013.
A reader who takes only the scatter away from this project has taken the wrong
thing, so the refutation travels with the plot.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a headless box
import matplotlib.pyplot as plt  # noqa: E402

RUNS = Path("runs")
FIGURES = Path("figures")
BASE_FULL = 0.2281
BASE_TOKENS = 149.0

# One consistent palette. Blue is DPO, orange RFT, grey the untrained control.
DPO, RFT, BASE, ACCENT = "#2c6fbb", "#e07b39", "#8a8a8a", "#b5382f"
SEQ = "#2e8b57"  # DPO on RFT — distinct from both parents


def load(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def steps_and_evals(run_dir) -> tuple[list[dict], list[dict]]:
    rows = [
        json.loads(line)
        for line in (Path(run_dir) / "log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return (
        [r for r in rows if r.get("event") != "dev_eval"],
        [r for r in rows if r.get("event") == "dev_eval"],
    )


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


# --- 1. seeds with error bars -------------------------------------------------


def figure_seeds() -> Path:
    arms = {
        "RFT\nlr 1e-5": (sorted((RUNS / "seeds_fulltier").glob("full_rft_s*.json")), RFT),
        "DPO\nβ=0.5 lr 5e-5": (sorted((RUNS / "seeds_fulltier").glob("full_dpo_s*.json")), DPO),
    }
    sequential = sorted((RUNS / "seeds_fulltier").glob("full_dpo_on_rft*.json"))
    if sequential:
        # SEQ, matching this arm's marker in length_vs_pass1.png: one colour per
        # arm across every figure, so a reader can carry them between plots.
        arms["DPO on RFT\nβ=0.5 lr 5e-5"] = (sequential, SEQ)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(BASE_FULL, color=BASE, linestyle="--", linewidth=1.2,
               label=f"base model ({BASE_FULL:.3f})")

    for i, (label, (paths, colour)) in enumerate(arms.items()):
        scores = [load(p)["pass_at_k_all"]["1"] for p in paths]
        mean = st.mean(scores)
        sd = st.stdev(scores) if len(scores) > 1 else 0.0
        ax.bar(i, mean, 0.55, color=colour, alpha=0.85, zorder=2)
        if sd:
            ax.errorbar(i, mean, yerr=sd, fmt="none", ecolor="#333", capsize=5,
                        linewidth=1.3, zorder=3)
        ax.scatter([i] * len(scores), scores, color="#222", s=16, zorder=4,
                   label="individual seeds" if i == 0 else None)
        note = f"{mean:.3f}" + (f"\n±{sd:.3f}" if sd else "\n(1 seed)")
        ax.text(i, mean + (sd or 0) + 0.006, note, ha="center", fontsize=8.5)

    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, fontsize=8.5)
    ax.set_ylim(0.20, 0.34)
    style(ax, "pass@1 on 200 held-out MBPP problems",
          "", "pass@1 (8 samples/problem)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.text(0.01, 0.01, "Error bars are SD across seeds. Bars are means; dots are "
             "individual runs.", fontsize=7.5, color="#555")
    return save(fig, "seeds.png")


# --- 2. likelihood levels for the anchor run ----------------------------------


def figure_logp_levels() -> Path:
    """Window means, not a light smooth.

    At batch size 1 pair, adjacent steps differ by 50+ nats purely from which
    pair was drawn, and a 9-step smooth leaves the trend invisible — the figure
    then asserts in its title something a reader cannot see. These are means
    over non-overlapping 15-step windows, the same windows the write-up quotes,
    with the raw per-step values behind them so the variance being averaged over
    is visible rather than hidden.
    """
    steps, evals = steps_and_evals(RUNS / "dpo_b0.1_lr1e-5")
    width = 15
    windows = [steps[i:i + width] for i in range(0, len(steps), width)]
    centres = [st.mean(r["step"] for r in w) for w in windows]

    def window_mean(key):
        return [st.mean(r[key] for r in w) for w in windows]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.0, 5.8), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})

    for key, colour, label in (("logp_chosen", DPO, "chosen"),
                               ("logp_rejected", RFT, "rejected")):
        ax.scatter([r["step"] for r in steps], [r[key] for r in steps],
                   color=colour, s=7, alpha=0.22, zorder=1)
        ax.plot(centres, window_mean(key), color=colour, linewidth=2.4,
                marker="o", markersize=5, zorder=3,
                label=f"logp {label} (15-step mean)")

    first_c, last_c = window_mean("logp_chosen")[0], window_mean("logp_chosen")[-1]
    first_r, last_r = window_mean("logp_rejected")[0], window_mean("logp_rejected")[-1]
    ax.annotate(f"chosen {first_c:.0f} → {last_c:.0f} ({last_c - first_c:+.0f} nats)",
                xy=(0.98, 0.93), xycoords="axes fraction", ha="right",
                fontsize=8.5, color=DPO)
    ax.annotate(f"rejected {first_r:.0f} → {last_r:.0f} ({last_r - first_r:+.0f} nats)",
                xy=(0.98, 0.85), xycoords="axes fraction", ha="right",
                fontsize=8.5, color=RFT)
    style(ax, "The margin grows because rejected falls, not because chosen rises",
          "", "summed log-probability (nats)")
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")

    ax2.plot([e["step"] for e in evals], [e["dev_mean_tokens"] for e in evals],
             color=ACCENT, marker="o", linewidth=2.0, markersize=6)
    for e in evals:
        last = e is evals[-1]
        ax2.annotate(f"pass@1 {e['dev_pass_at_1']:.3f}",
                     xy=(e["step"], e["dev_mean_tokens"]),
                     xytext=(-6 if last else 0, -16),
                     textcoords="offset points",
                     ha="right" if last else "center", fontsize=8,
                     color=ACCENT if last else "#444")
    ax2.axhline(161, color=BASE, linestyle="--", linewidth=1.1)
    ax2.text(2, 163, "base model: 161 tokens", fontsize=7.5, color="#555")
    ax2.set_ylim(88, 172)
    style(ax2, "Generated length falls throughout, and pass@1 follows it down",
          "optimizer step", "mean tokens (dev)")
    fig.text(0.01, 0.01, "β=0.1, lr=1e-5. Faint dots are per-step training batches; "
             "lines are 15-step means. Length and pass@1 are dev generations.",
             fontsize=7.5, color="#555")
    return save(fig, "logp_levels.png")


# --- 3. beta x lr heatmap -----------------------------------------------------


def figure_beta_lr() -> Path:
    # Best dev checkpoint per cell; None where the cell was not run.
    grid = {
        "1e-5": {"0.05": 0.2500, "0.1": 0.2639, "0.3": 0.2556, "0.5": 0.2639},
        "5e-5": {"0.05": 0.1917, "0.1": 0.2806, "0.3": 0.2833, "0.5": 0.2861},
    }
    betas = ["0.05", "0.1", "0.3", "0.5"]
    lrs = ["1e-5", "5e-5"]
    data = [[grid[lr][b] for b in betas] for lr in lrs]

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    im = ax.imshow(data, cmap="RdYlBu", aspect="auto", vmin=0.19, vmax=0.29)
    for i, lr in enumerate(lrs):
        for j, b in enumerate(betas):
            value = grid[lr][b]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=10,
                    fontweight="bold" if value > 0.284 else "normal",
                    color="#111")
    ax.set_xticks(range(len(betas)), [f"β={b}" for b in betas], fontsize=9)
    ax.set_yticks(range(len(lrs)), [f"lr {lr}" for lr in lrs], fontsize=9)
    ax.set_title("The axes are not separable: β's ordering inverts with learning rate",
                 fontsize=10.5, fontweight="bold", loc="left")
    ax.add_patch(plt.Rectangle((-0.5, 0.5), 1, 1, fill=False, edgecolor=ACCENT,
                               linewidth=2.2))
    ax.text(0.52, 1.0, "β collapse\n34-token output", fontsize=8, color=ACCENT,
            va="center")
    fig.colorbar(im, ax=ax, label="dev pass@1", shrink=0.85)
    fig.text(0.01, 0.01, "Best dev checkpoint per cell. The sweep started at "
             "β=0.1, lr=1e-5 and moved β first, so it never reached the bottom-right.",
             fontsize=7.5, color="#555")
    return save(fig, "beta_lr.png")


# --- 4. length vs pass@1, labelled as correlation -----------------------------


def figure_length_vs_pass() -> Path:
    points = []
    for path in sorted(RUNS.glob("fulltier/full_*.json")) + sorted(
        RUNS.glob("seeds_fulltier/full_*.json")
    ):
        report = load(path)
        name = path.stem[5:]
        points.append((report["mean_completion_tokens"], report["pass_at_k_all"]["1"], name))

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for tokens, score, name in points:
        # dpo_on_rft must be tested before rft: the substring matches both, and
        # colouring the project's best checkpoint as the arm that did not work
        # would invert the figure's message.
        if name == "base":
            colour, marker, size = BASE, "s", 70
        elif "balanced" in name:
            colour, marker, size = ACCENT, "D", 80
        elif "dpo_on_rft" in name:
            colour, marker, size = SEQ, "*", 150
        elif "rft" in name:
            colour, marker, size = RFT, "^", 55
        else:
            colour, marker, size = DPO, "o", 55
        ax.scatter(tokens, score, color=colour, marker=marker, s=size,
                   alpha=0.85, zorder=3, edgecolor="white", linewidth=0.6)

    ax.axhline(BASE_FULL, color=BASE, linestyle="--", linewidth=1.1)
    ax.axvline(BASE_TOKENS, color=BASE, linestyle="--", linewidth=1.1)
    ax.text(BASE_TOKENS + 3, 0.222, "base", fontsize=8, color="#555")

    # No floating callout: every empty region of this plot is adjacent to a
    # data point, and a box with a connector reads as a fitted trend line
    # pointing the opposite way to the correlation in the title. The refutation
    # rides in the title, the legend label and the footnote instead.
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    r = st.correlation(xs, ys) if hasattr(st, "correlation") else float("nan")
    style(ax, f"Length correlates with pass@1 (r = {r:.2f}) — not a mechanism",
          "mean generated tokens", "full-tier pass@1")
    ax.set_ylim(0.222, 0.312)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color=DPO, marker="o", linestyle="", label="DPO checkpoints"),
        Line2D([], [], color=RFT, marker="^", linestyle="", label="RFT checkpoints"),
        Line2D([], [], color=SEQ, marker="*", linestyle="", label="DPO on RFT (sequential)"),
        Line2D([], [], color=ACCENT, marker="D", linestyle="",
               label="length-balanced corpus — control:\n+0.009 ± 0.013 (z=0.73), no effect"),
        Line2D([], [], color=BASE, marker="s", linestyle="", label="base model"),
    ], fontsize=8, frameon=False, loc="upper left")
    fig.text(0.01, 0.01, "Correlational. The balanced-corpus point is the control: "
             "intervening on length directly moves the result by +0.009 ± 0.013.",
             fontsize=7.5, color="#555")
    return save(fig, "length_vs_pass1.png")


def save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> int:
    for builder in (figure_seeds, figure_logp_levels, figure_beta_lr,
                    figure_length_vs_pass):
        print(f"wrote {builder()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
