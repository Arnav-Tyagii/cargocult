"""
Reward ladder for execution-verified preference optimization.

Turns a sandbox ExecResult into a scalar in [-0.1, 1.0].

WHY A LADDER AND NOT PASS/FAIL
------------------------------
The obvious reward is `1.0 if all tests pass else 0.0`. At 0.5B scale that
is too sparse: on MBPP most sampled completions fail, so a binary reward
gives an almost-constant signal and almost no information about *how* a
completion failed.

The ladder makes the reward monotonic in "how close to correct" — code that
parses beats code that doesn't, code that runs beats code that crashes, and
2-of-3 asserts beats 1-of-3. This matters in two different ways:

  - For DPO: it lets pairs.py enforce a minimum reward margin, so we can
    discard pairs where chosen and rejected are effectively equivalent.
  - For GRPO later (see PROJECT.md §8): group advantages are computed from
    the spread of rewards within a group. A binary reward produces mostly
    all-zero groups, which yield zero advantage and a wasted training step.

TIER BOUNDARIES
---------------
Each band is closed at the top by the floor of the next band, so a completion
can never be rewarded for a lower tier than it actually reached.
"""

# Imported, not re-declared: a local mirror of this dataclass would let the
# reward ladder and the sandbox drift apart silently.
from src.sandbox import ExecResult


# --- Tier floors -------------------------------------------------------------
# Tuned by hand on the Phase 1 baseline generations. See docstring per constant
# for the reasoning; these are judgment calls, not derived values.

R_UNPARSEABLE = 0.0
# Nothing to reward. Not negative: a negative floor plus the token-limit
# penalty would make "emit nothing" score higher than "attempt and fail",
# which is the opposite of what we want.

R_PARSES = 0.10
# Syntactically valid Python. Low, but nonzero — this is the one rung the
# policy can reliably climb early, and it separates "produced code" from
# "produced prose about code", which is a common 0.5B failure mode.

R_RUNS = 0.30
# Executes without raising. Note this includes the degenerate case of code
# that runs but passes zero asserts (e.g. a stub returning None). The gap
# from 0.10 to 0.30 is deliberately large: not crashing is a real milestone.

R_TESTS_FLOOR = 0.30
R_TESTS_CEIL = 1.00
# Fraction of asserts passed maps linearly into this band. Starts exactly at
# R_RUNS so the function is continuous — 0-of-3 passing scores the same as
# "ran but had no tests", with no discontinuity to exploit.

P_TOKEN_LIMIT = -0.10
# Applied when the completion hit max_new_tokens without emitting EOS.
#
# Without this, rambling is a viable strategy: a long completion that keeps
# emitting plausible-looking code has more chances to stumble into a passing
# definition than a short one. That inflates reward without improving the
# policy, and it compounds with DPO's known length bias (PROJECT.md §4).
#
# Small on purpose. Large enough to break ties against verbose completions,
# small enough that a *correct* answer that ran long still outranks an
# incorrect short one: 1.00 - 0.10 = 0.90 > R_RUNS.

R_MIN = P_TOKEN_LIMIT
R_MAX = R_TESTS_CEIL


def compute_reward(result: ExecResult, hit_token_limit: bool) -> float:
    """Score one completion. FROZEN: see PROJECT.md §8.

    Stateless and per-completion by design — GRPO computes group-normalized
    advantages over the outputs of this function without modifying it.

    Args:
        result: sandbox execution result for this completion.
        hit_token_limit: True if generation stopped at max_new_tokens
            rather than at an EOS token.

    Returns:
        Scalar reward in [R_MIN, R_MAX].
    """
    if not result.parsed:
        reward = R_UNPARSEABLE

    elif result.timed_out:
        # Timeout is treated as parse-level, not run-level. An infinite loop
        # is a worse failure than a clean exception: it costs 6s of wall time
        # per occurrence and, unpenalized, "loop forever" can look safer to
        # the policy than "return something that might be wrong".
        reward = R_PARSES

    elif not result.ran:
        reward = R_PARSES

    elif result.n_tests == 0:
        # Ran, but nothing to check against. Shouldn't happen on MBPP (every
        # problem ships 3 asserts) — treat as a data bug and score at the
        # run tier rather than crashing the pipeline mid-generation.
        reward = R_RUNS

    else:
        frac = result.n_passed / result.n_tests
        reward = R_TESTS_FLOOR + frac * (R_TESTS_CEIL - R_TESTS_FLOOR)

    if hit_token_limit:
        reward += P_TOKEN_LIMIT

    return max(R_MIN, min(R_MAX, reward))


def reward_tier(result: ExecResult) -> str:
    """Coarse label for the failure taxonomy in the writeup. Not used in training."""
    if not result.parsed:
        return "unparseable"
    if result.timed_out:
        return "timeout"
    if not result.ran:
        return "runtime_error"
    if result.n_tests and result.n_passed == result.n_tests:
        return "pass"
    if result.n_passed > 0:
        return "partial"
    return "wrong"
