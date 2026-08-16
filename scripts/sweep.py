"""Phase 4 sweep driver. Runs unattended and stops itself when it should.

    python scripts/sweep.py

Order (PROJECT.md §4 — move one axis at a time, never the full grid):

  1. beta 0.1, lr 1e-5, unbalanced pairs   <- the anchor everything is read against
  2. RFT baseline
  3. beta 0.05 / 0.3 / 0.5 at lr 1e-5
  4. lr 5e-6 / 5e-5 at whichever beta won

Dev tier only. A full-tier eval is for final candidates, and running one
during a sweep is how the GPU budget goes.

STOPPING EARLY
--------------
Three conditions end the night rather than burning it on a broken config:

  - **divergence** — mean loss over the last third above the first third.
  - **saturated ranking, flat capability** — reward_accuracy above 0.95 while
    dev pass@1 is at or below baseline. This is the failure worth catching:
    DPO gets very good at ordering the pairs it was given while the policy
    gets no better at writing code, and the loss curve looks excellent
    throughout.
  - **OOM** — a checkpoint that cannot allocate. Retrying it just fails again.

Each run's results are committed as it lands, so a crash at 4am keeps
everything finished before it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

RUNS_DIR = REPO / "runs"
PYTHON = sys.executable

DEV_BASELINE = 0.2472  # runs/baseline/dev_temp0.8.json
FULL_BASELINE = 0.2281
EVAL_EVERY = 45
ACCURACY_CEILING = 0.95


def run_training(tag: str, module: str, extra: list[str]) -> tuple[int, str]:
    """One training run as a subprocess. Returns (returncode, combined output)."""
    run_dir = RUNS_DIR / tag
    cmd = [
        PYTHON, "-m", module,
        "--tag", tag,
        "--run-dir", str(run_dir),
        "--eval-every", str(EVAL_EVERY),
        *extra,
    ]
    print(f"\n{'=' * 72}\n{tag}\n{' '.join(cmd)}\n{'=' * 72}", flush=True)
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    minutes = (time.perf_counter() - started) / 60
    output = (proc.stdout or "") + (proc.stderr or "")
    for line in output.splitlines():
        if line.strip() and "it/s" not in line and "%|" not in line:
            print("   " + line[:160], flush=True)
    print(f"   -> exit {proc.returncode} in {minutes:.1f} min", flush=True)
    return proc.returncode, output


def read_run(tag: str) -> dict:
    """What a finished run produced: step rows, evals, and the numbers §4 wants."""
    run_dir = RUNS_DIR / tag
    rows, evals = [], []
    log = run_dir / "log.jsonl"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            (evals if record.get("event") == "dev_eval" else rows).append(record)

    losses = [r["loss"] for r in rows if "loss" in r]
    third = max(1, len(losses) // 3)
    final = rows[-1] if rows else {}
    best = max((e["dev_pass_at_1"] for e in evals), default=float("nan"))
    last_eval = evals[-1] if evals else {}

    return {
        "tag": tag,
        "steps": len(rows),
        "loss_first_third": sum(losses[:third]) / third if losses else float("nan"),
        "loss_last_third": sum(losses[-third:]) / third if losses else float("nan"),
        "reward_accuracy": final.get("reward_accuracy"),
        "reward_margin": final.get("reward_margin"),
        "logp_chosen": final.get("logp_chosen"),
        "logp_rejected": final.get("logp_rejected"),
        "logp_chosen_start": next((r["logp_chosen"] for r in rows if "logp_chosen" in r), None),
        "len_chosen": final.get("len_chosen") or final.get("len_completion"),
        "len_chosen_terminated": (
            final.get("len_chosen_terminated") or final.get("len_completion_terminated")
        ),
        "len_rejected": final.get("len_rejected"),
        "len_rejected_terminated": final.get("len_rejected_terminated"),
        "peak_vram_mb": max((r.get("peak_vram_mb", 0) for r in rows), default=0),
        "dev_pass_at_1": last_eval.get("dev_pass_at_1", float("nan")),
        "dev_pass_at_1_best": best,
        "dev_stub_args_rate": last_eval.get("dev_stub_args_rate"),
        "dev_mean_tokens": last_eval.get("dev_mean_tokens"),
        "evals": [(e["step"], e["dev_pass_at_1"], e.get("dev_stub_args_rate")) for e in evals],
    }


def check_stop(result: dict, returncode: int, output: str) -> str | None:
    """The three conditions from the brief. Returns a reason, or None."""
    if "OutOfMemoryError" in output or "CUDA out of memory" in output:
        return "a checkpoint ran out of VRAM"
    if returncode != 0:
        tail = "\n".join(output.strip().splitlines()[-12:])
        return f"the run exited {returncode}:\n\n```\n{tail}\n```"
    if result["loss_last_third"] > result["loss_first_third"]:
        return (
            f"loss increased across thirds "
            f"({result['loss_first_third']:.4f} -> {result['loss_last_third']:.4f}) — diverging"
        )
    accuracy = result["reward_accuracy"]
    if (
        accuracy is not None
        and accuracy > ACCURACY_CEILING
        and result["dev_pass_at_1"] <= DEV_BASELINE
    ):
        return (
            f"reward_accuracy saturated at {accuracy:.3f} while dev pass@1 "
            f"({result['dev_pass_at_1']:.4f}) stayed at or below the "
            f"{DEV_BASELINE:.4f} baseline — the policy is learning to rank the "
            "pairs, not to write code"
        )
    return None


def commit(message: str) -> None:
    subprocess.run(["git", "add", "runs/", "notes/"], cwd=REPO, check=False)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=REPO, check=False)


def write_sweep_summary(results: list[dict], stopped: str | None) -> Path:
    lines = [
        "# Phase 4 sweep",
        "",
        f"Dev-tier evals throughout (90 problems x 4 samples, temperature 0.8). "
        f"The dev baseline is **{DEV_BASELINE:.4f}**; the full-tier baseline is "
        f"{FULL_BASELINE:.4f} and is *not* what these are compared against.",
        "",
        "## Dev pass@1",
        "",
        "| run | dev pass@1 | vs baseline | best across checkpoints | steps |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        delta = r["dev_pass_at_1"] - DEV_BASELINE
        lines.append(
            f"| `{r['tag']}` | {r['dev_pass_at_1']:.4f} | {delta:+.4f} | "
            f"{r['dev_pass_at_1_best']:.4f} | {r['steps']} |"
        )

    lines += [
        "",
        "## Likelihood displacement",
        "",
        "DPO constrains the gap, never the levels. A run whose `logp_chosen` "
        "fell while its margin grew is displacing likelihood — the loss curve "
        "will look fine.",
        "",
        "| run | logp_chosen | logp_rejected | chosen drift | margin | reward_acc |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        drift = (
            r["logp_chosen"] - r["logp_chosen_start"]
            if r["logp_chosen"] is not None and r["logp_chosen_start"] is not None
            else float("nan")
        )
        lines.append(
            f"| `{r['tag']}` | {_fmt(r['logp_chosen'])} | {_fmt(r['logp_rejected'])} | "
            f"{drift:+.1f} | {_fmt(r['reward_margin'])} | {_fmt(r['reward_accuracy'])} |"
        )

    lines += [
        "",
        "## Length and style",
        "",
        "`*_terminated` excludes completions cut off at the token budget. The two "
        "columns differ because 11.6% of the rejected side of the corpus was "
        "truncated, and a truncated completion is long for a reason unrelated to "
        "verbosity.",
        "",
        "| run | train len (all) | train len (terminated) | dev mean tokens | stub_args |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r['tag']}` | {_fmt(r['len_chosen'])} | {_fmt(r['len_chosen_terminated'])} | "
            f"{_fmt(r['dev_mean_tokens'])} | {_pct(r['dev_stub_args_rate'])} |"
        )
    lines += [
        "",
        f"Baseline `stub_args_rate` is 80.8% on dev. The prediction registered in "
        "`notes/readme_draft.md` before any of these runs was that DPO would push "
        "it up, because the pair corpus prefers placeholder retention by 16.8 points.",
        "",
        "## Per-run detail",
        "",
    ]
    for r in results:
        lines.append(f"- [`{r['tag']}`](./{r['tag']}/summary.md) — "
                     f"loss {r['loss_first_third']:.4f} -> {r['loss_last_third']:.4f}, "
                     f"peak {r['peak_vram_mb']:.0f} MB")

    if stopped:
        lines += ["", "## Stopped early", "", stopped, "",
                  "Remaining runs were not started. Nothing was retried — a broken "
                  "config fails the same way the second time."]

    path = RUNS_DIR / "sweep_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.2f}"


def _pct(value) -> str:
    return "—" if value is None else f"{value:.1%}"


def main() -> int:
    results: list[dict] = []
    stopped: str | None = None

    plan = [
        ("dpo_b0.1_lr1e-5", "src.train_dpo", ["--beta", "0.1", "--lr", "1e-5"]),
        ("rft_lr1e-5", "src.train_rft", ["--lr", "1e-5"]),
        ("dpo_b0.05_lr1e-5", "src.train_dpo", ["--beta", "0.05", "--lr", "1e-5"]),
        ("dpo_b0.3_lr1e-5", "src.train_dpo", ["--beta", "0.3", "--lr", "1e-5"]),
        ("dpo_b0.5_lr1e-5", "src.train_dpo", ["--beta", "0.5", "--lr", "1e-5"]),
    ]

    for tag, module, extra in plan:
        returncode, output = run_training(tag, module, extra)
        result = read_run(tag)
        results.append(result)
        write_sweep_summary(results, None)
        commit(f"sweep: {tag} dev pass@1 {result['dev_pass_at_1']:.4f}")

        stopped = check_stop(result, returncode, output)
        if stopped:
            stopped = f"**Stopped after `{tag}`.** {stopped}"
            print(f"\n!! STOPPING: {stopped}", flush=True)
            break

    # The LR axis moves only after beta has been chosen, and only on the beta
    # that won — sweeping LR at every beta is the full grid §4 rules out.
    if not stopped:
        dpo = [r for r in results if r["tag"].startswith("dpo_")]
        best = max(dpo, key=lambda r: r["dev_pass_at_1_best"])
        beta = best["tag"].split("_")[1].removeprefix("b")
        print(f"\nbest beta so far: {beta} "
              f"(dev pass@1 {best['dev_pass_at_1_best']:.4f}) — sweeping LR on it",
              flush=True)

        for lr in ("5e-6", "5e-5"):
            tag = f"dpo_b{beta}_lr{lr}"
            returncode, output = run_training(
                tag, "src.train_dpo", ["--beta", beta, "--lr", lr]
            )
            result = read_run(tag)
            results.append(result)
            write_sweep_summary(results, None)
            commit(f"sweep: {tag} dev pass@1 {result['dev_pass_at_1']:.4f}")

            stopped = check_stop(result, returncode, output)
            if stopped:
                stopped = f"**Stopped after `{tag}`.** {stopped}"
                print(f"\n!! STOPPING: {stopped}", flush=True)
                break

    path = write_sweep_summary(results, stopped)
    commit("sweep: combined summary")
    print(f"\nsweep summary {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
