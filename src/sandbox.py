"""Subprocess sandbox for executing untrusted, model-generated Python.

Every completion sampled from the policy goes through here before it is
scored. Assume the model will eventually emit `os.system("rm -rf ~")` or
`while True: pass` — at temperature 1.0, over thousands of samples, it will.

The candidate code runs in a fresh interpreter, in a throwaway working
directory, with a stripped environment, under a wall-clock timeout and a
memory cap. Nothing is ever `exec`d in the parent process.

PARTIAL CREDIT
--------------
Each assert is executed separately inside the child so `n_passed` is a real
fraction, not all-or-nothing (the reward ladder in reward.py maps it into
[0.3, 1.0]). The child rewrites its result file after every test, so a
completion that passes two asserts and then hangs on the third still keeps
the two.

ISOLATION GUARANTEES
--------------------
This is a blast shield, not a jail. What it does give you:

  - separate process, killed hard (SIGKILL / TerminateProcess) on expiry
  - the whole process tree dies, not just the direct child
  - cwd is a fresh mkdtemp() that is removed afterwards
  - env is a whitelist: no proxy vars, no API tokens, HOME/TEMP point into
    the sandbox directory, PATH is trimmed to the system directory
  - `-I` (isolated mode) so PYTHONPATH and the user site-dir are ignored

What it does not give you, on any platform: a filesystem jail (an absolute
path outside the temp dir is still writable), network isolation, or a
syscall filter. Run it on a machine you are willing to have a bad day.

Weaker on Windows than on Linux, specifically:

  - Memory cap. This module polls RSS with psutil instead of using
    resource.setrlimit(RLIMIT_AS) in a preexec_fn, because neither exists on
    Windows and the same code has to run unmodified on Kaggle. A poll-based
    cap is soft: a single huge allocation can complete between two polls, so
    the child can briefly exceed the limit before it is killed. setrlimit
    would have refused the allocation outright. RSS is also not address
    space — memory that is reserved but never touched is not counted.
  - Process-group kill. On POSIX the child gets its own session
    (start_new_session=True), so os.killpg is a backstop that catches
    double-forked grandchildren which have been reparented away from our
    process tree. Windows has no equivalent; the psutil tree walk is all
    there is, and a process deliberately detached from its parent would
    survive it. Containing that properly would need a Job Object.
  - Kill semantics. TerminateProcess is not quite SIGKILL: a process blocked
    in a kernel call can linger briefly after being terminated.

USE FROM A PROCESS POOL
-----------------------
`run_tests` is a module-level function taking plain strings and returning a
module-level dataclass, so it and its results pickle cleanly and it can be
submitted straight to a ProcessPoolExecutor. Phase 2 overlaps sandbox
execution (CPU) with generation (GPU) that way.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import psutil

DEFAULT_TIMEOUT = 6.0
DEFAULT_MEMORY_MB = 512

# How often the parent checks the child's clock, memory and output volume.
# The cost of a coarser interval is overshoot on the memory cap: a tight
# allocation loop moves ~4 GB/s, so 25 ms of blindness is ~100 MB over the
# limit before the kill lands.
POLL_INTERVAL = 0.025

# Reading one process's RSS costs ~6 us; discovering its descendants costs
# ~6 ms, because psutil has to enumerate every pid on the box to find them.
# So descendants are *discovered* a few times a second and then polled from a
# cached list at the full rate. A process spawned by the candidate is
# therefore invisible for up to this long, which the wall-clock timeout
# bounds; it is not worth 6 ms per 25 ms in every pool worker to close.
TREE_POLL_INTERVAL = 0.25

STDERR_TAIL_CHARS = 500
MAX_STDERR_BYTES = 1 << 20  # kill a completion that is spewing to stderr


@dataclass
class ExecResult:
    """Structured outcome of one sandboxed execution. Consumed by reward.py."""

    parsed: bool  # ast.parse() succeeded
    ran: bool  # executed without raising
    n_tests: int
    n_passed: int
    timed_out: bool
    stderr_tail: str  # last 500 chars, for the failure taxonomy
    wall_time: float


# The child-side driver. Written into the sandbox directory and run by a fresh
# interpreter; never imported by the parent. Kept as a string so sandbox.py
# stays a single self-contained file that can be copied to Kaggle as-is.
_RUNNER_SOURCE = '''\
"""Runs one candidate against its asserts. Executed inside the sandbox."""
import json
import os
import sys
import traceback

_PAYLOAD, _RESULT = sys.argv[1], sys.argv[2]
_MAX_TRACEBACK = 2000


def _write(state):
    # Replace atomically: the parent may read this file at any moment, and
    # will read it after killing us mid-write.
    tmp = _RESULT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh)
    os.replace(tmp, _RESULT)


def main():
    with open(_PAYLOAD, encoding="utf-8") as fh:
        payload = json.load(fh)
    tests = payload["tests"]

    # Rewritten after every step, so that being killed for time or memory
    # still leaves whatever partial credit was earned before the kill.
    state = {"ran": False, "n_passed": 0, "stderr": ""}
    _write(state)

    namespace = {"__name__": "__candidate__"}
    try:
        exec(compile(payload["code"], "<candidate>", "exec"), namespace)
        if payload["setup_code"]:
            exec(compile(payload["setup_code"], "<setup>", "exec"), namespace)
    except BaseException:
        # BaseException, not Exception: sys.exit() and MemoryError inside the
        # candidate are failures of the candidate, not of this harness.
        state["stderr"] = traceback.format_exc()[-_MAX_TRACEBACK:]
        _write(state)
        return
    state["ran"] = True
    _write(state)

    failures = []
    for i, test in enumerate(tests):
        try:
            exec(compile(test, "<test%d>" % i, "exec"), namespace)
            state["n_passed"] += 1
        except BaseException:
            failures.append(
                "test %d: %s" % (i, traceback.format_exc()[-_MAX_TRACEBACK:])
            )
            state["stderr"] = "".join(failures)[-_MAX_TRACEBACK:]
        _write(state)


main()
sys.stderr.flush()
# os._exit, not a normal return: a candidate that left a non-daemon thread
# running would otherwise keep this process alive until the wall-clock
# timeout, and we would pay 6 s for a completion that already finished.
os._exit(0)
'''


def run_tests(
    code: str,
    tests: Sequence[str],
    *,
    setup_code: str = "",
    timeout: float = DEFAULT_TIMEOUT,
    memory_limit_mb: int = DEFAULT_MEMORY_MB,
) -> ExecResult:
    """Execute `code`, then each of `tests` against it, in a sandboxed child.

    Args:
        code: candidate program. Defines whatever the asserts call.
        tests: assert statements, run one at a time in the order given and in
            a namespace shared with `code`.
        setup_code: optional extra statements run after `code` and before the
            asserts (MBPP ships some problems with a `test_setup_code` field).
        timeout: wall-clock budget for the whole child, including interpreter
            startup. On expiry the process tree is killed.
        memory_limit_mb: RSS ceiling for the process tree. Soft — see the
            module docstring.

    Returns:
        ExecResult. Never raises for anything the candidate does.
    """
    tests = list(tests)
    started = time.perf_counter()

    # Parse in the parent: ast.parse does not execute anything, and skipping
    # the process spawn for unparseable output is worth a lot when a sizeable
    # slice of 0.5B samples is prose rather than code.
    ok, parse_error = _parses(code)
    if not ok:
        return ExecResult(
            parsed=False,
            ran=False,
            n_tests=len(tests),
            n_passed=0,
            timed_out=False,
            stderr_tail=parse_error[-STDERR_TAIL_CHARS:],
            wall_time=time.perf_counter() - started,
        )

    root = tempfile.mkdtemp(prefix="cargocult-")
    try:
        return _execute(root, code, tests, setup_code, timeout, memory_limit_mb, started)
    finally:
        _rmtree(root)


def _execute(
    root: str,
    code: str,
    tests: list[str],
    setup_code: str,
    timeout: float,
    memory_limit_mb: int,
    started: float,
) -> ExecResult:
    """Spawn the child, supervise it, and turn what is left behind into an ExecResult."""
    runner_path = os.path.join(root, "runner.py")
    payload_path = os.path.join(root, "payload.json")
    result_path = os.path.join(root, "result.json")
    stderr_path = os.path.join(root, "stderr.txt")
    # The child's cwd is a separate empty subdirectory, so a candidate that
    # lists or deletes its way through "." cannot reach the runner or the
    # result file it is being scored by.
    workdir = os.path.join(root, "work")
    os.mkdir(workdir)

    _write_text(runner_path, _RUNNER_SOURCE)
    _write_text(
        payload_path,
        json.dumps({"code": code, "setup_code": setup_code, "tests": tests}),
    )

    memory_limit = memory_limit_mb * 1024 * 1024
    timed_out = False
    kill_reason = ""

    stderr_fh = open(stderr_path, "wb")
    try:
        popen = subprocess.Popen(
            [sys.executable, "-I", "-X", "utf8", runner_path, payload_path, result_path],
            cwd=workdir,
            env=_child_env(workdir),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,  # candidate prints are noise; results go to the file
            stderr=stderr_fh,
            start_new_session=True,  # POSIX: own process group, so killpg works
            creationflags=_CREATION_FLAGS,
        )
        try:
            proc = psutil.Process(popen.pid)
        except psutil.Error:
            proc = None

        deadline = started + timeout
        tracked = [proc] if proc is not None else []
        next_tree_poll = started  # discover descendants on the first poll
        try:
            while True:
                try:
                    popen.wait(timeout=POLL_INTERVAL)
                    break
                except subprocess.TimeoutExpired:
                    pass
                now = time.perf_counter()
                if now >= deadline:
                    timed_out = True
                    kill_reason = f"wall-clock timeout after {timeout:.1f}s"
                    break
                if now >= next_tree_poll:
                    next_tree_poll = now + TREE_POLL_INTERVAL
                    tracked = _descendants(proc)
                rss = _tracked_rss(tracked)
                if rss > memory_limit:
                    kill_reason = (
                        f"exceeded {memory_limit_mb} MB memory cap "
                        f"(rss {rss / 1024 / 1024:.0f} MB)"
                    )
                    break
                if _file_size(stderr_path) > MAX_STDERR_BYTES:
                    kill_reason = f"wrote over {MAX_STDERR_BYTES} bytes to stderr"
                    break
        finally:
            # Also covers KeyboardInterrupt in the supervising loop: never
            # leave the child running after run_tests returns.
            if popen.poll() is None:
                _kill_tree(popen, proc)
    finally:
        stderr_fh.close()

    wall_time = time.perf_counter() - started
    state = _read_json(result_path)

    if state is None:
        # Killed before it could write anything, or the interpreter itself
        # failed to start.
        ran, n_passed, runner_stderr = False, 0, ""
    else:
        ran = bool(state.get("ran", False))
        n_passed = max(0, min(len(tests), int(state.get("n_passed", 0))))
        runner_stderr = str(state.get("stderr", ""))

    pieces = [p for p in (runner_stderr, _tail_file(stderr_path)) if p]
    if kill_reason:
        # Last, so it survives the tail truncation: it is the one line that
        # explains an otherwise empty result.
        pieces.append(f"sandbox: killed, {kill_reason}")

    return ExecResult(
        parsed=True,
        ran=ran,
        n_tests=len(tests),
        n_passed=n_passed,
        timed_out=timed_out,
        stderr_tail="\n".join(pieces)[-STDERR_TAIL_CHARS:],
        wall_time=wall_time,
    )


# --- process control ---------------------------------------------------------

# CREATE_NO_WINDOW: without it, every completion flashes a console window on
# Windows, which over 3,000 completions is unusable.
_CREATION_FLAGS = 0x08000000 if os.name == "nt" else 0


def _descendants(proc: psutil.Process | None) -> list[psutil.Process]:
    """The child and every process below it, as psutil handles to poll later."""
    if proc is None:
        return []
    try:
        return [proc] + proc.children(recursive=True)
    except psutil.Error:
        return [proc]  # already gone; memory_info() on it will just return 0


def _tracked_rss(procs: list[psutil.Process]) -> int:
    """Total resident memory of an already-discovered set of processes.

    Summing the tree matters more than it looks: on Windows a venv's
    Scripts/python.exe can be a launcher stub that re-execs the base
    interpreter as a *child*, so the process we spawned sits at 4 MB forever
    while the process actually running the candidate is one level down. A
    limit applied to the direct child alone would measure the wrong process.
    """
    total = 0
    for proc in procs:
        try:
            total += proc.memory_info().rss
        except psutil.Error:
            pass  # exited since it was discovered
    return total


def _kill_tree(popen: subprocess.Popen, proc: psutil.Process | None) -> None:
    """SIGKILL the child and every descendant. Reaps, so no zombies are left."""
    victims = []
    if proc is not None:
        try:
            victims = proc.children(recursive=True)
        except psutil.Error:
            victims = []
        victims.append(proc)

    pgid = None
    if os.name == "posix":
        try:
            pgid = os.getpgid(popen.pid)  # read before killing, while it still exists
        except OSError:
            pgid = None

    for victim in victims:
        try:
            victim.kill()  # SIGKILL on POSIX, TerminateProcess on Windows
        except psutil.Error:
            pass

    if pgid is not None:
        # Backstop for a grandchild that double-forked and was reparented out
        # of the tree walk above. No Windows equivalent — see module docstring.
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass

    psutil.wait_procs(victims, timeout=1.0)
    try:
        popen.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass


def _child_env(home: str) -> dict[str, str]:
    """Minimal environment: a whitelist, so nothing has to be remembered to be removed.

    Proxy variables are dropped by construction, which is the best-effort part
    of "no network" — it is not a network jail. HOME and the temp variables
    point into the sandbox so that `~/x` and tempfile both land somewhere that
    gets deleted.
    """
    env = {
        "HOME": home,
        "USERPROFILE": home,
        "TMPDIR": home,
        "TEMP": home,
        "TMP": home,
        "LANG": "C.UTF-8",
    }
    # The interpreter's own directory comes first: a conda or venv python can
    # need it to find itself, and losing that turns every completion into a
    # spawn failure on a machine we are not sitting in front of.
    path = [os.path.dirname(sys.executable)]
    if os.name == "nt":
        # Python fails to start on Windows without SystemRoot (it cannot
        # initialise its random seed). System32 is the only other PATH entry.
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        env["SystemRoot"] = system_root
        path.append(os.path.join(system_root, "System32"))
    else:
        path.extend(["/usr/bin", "/bin"])
        # Kaggle's interpreter is conda-built and resolves shared objects
        # through this. Not a secret, and dropping it breaks startup.
        if "LD_LIBRARY_PATH" in os.environ:
            env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    env["PATH"] = os.pathsep.join(path)
    return env


# --- small file helpers ------------------------------------------------------


def _parses(code: str) -> tuple[bool, str]:
    """ast.parse in the parent. ValueError covers null bytes; the parser can
    also blow the stack on the deeply nested junk a small model emits."""
    try:
        # Warnings are suppressed because the parent is parsing untrusted
        # source: a completion containing "\s" inside a plain string makes
        # the tokenizer emit a SyntaxWarning *here*, in our process, and a
        # 1,600-sample eval then interleaves that noise with its own output.
        # The candidate's own warnings still reach its stderr in the child.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _tail_file(path: str, n_chars: int = STDERR_TAIL_CHARS) -> str:
    """Last n_chars of a file, without reading the whole thing into memory."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - n_chars * 4))  # 4 bytes/char worst case
            data = fh.read()
    except OSError:
        return ""
    return data.decode("utf-8", "replace")[-n_chars:]


def _rmtree(path: str) -> None:
    """Delete the sandbox directory. Retries because a just-killed child on
    Windows can still hold a handle open for a moment."""
    for _ in range(3):
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            return
        time.sleep(0.05)
