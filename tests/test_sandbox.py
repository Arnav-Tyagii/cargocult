"""Adversarial tests for the execution sandbox.

The suite is deliberately hostile: infinite loops, memory bombs, escaping
writes, spawned grandchildren. Everything a temperature-1.0 sample from a
0.5B model will eventually produce.

Two tests assert *limitations* rather than protections
(test_absolute_path_write_is_not_blocked, test_network_is_not_blocked). They
exist so the docstring in sandbox.py cannot quietly drift away from what the
code actually does — if someone adds real jailing, those tests fail and the
documentation gets updated with them.
"""

import os
import pickle
import sys
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor

import psutil
import pytest

from src.sandbox import DEFAULT_MEMORY_MB, DEFAULT_TIMEOUT, ExecResult, run_tests

ADD = "def add(a, b):\n    return a + b"
ADD_TESTS = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"]


def literal(path) -> str:
    """A Windows path is full of backslashes; repr() makes it a safe literal."""
    return repr(str(path))


# --- the happy path ----------------------------------------------------------


def test_clean_pass():
    result = run_tests(ADD, ADD_TESTS)
    assert result == ExecResult(
        parsed=True,
        ran=True,
        n_tests=3,
        n_passed=3,
        timed_out=False,
        stderr_tail="",
        wall_time=result.wall_time,
    )
    assert 0 < result.wall_time < DEFAULT_TIMEOUT


def test_setup_code_runs_before_the_asserts():
    result = run_tests(
        "def add(a, b):\n    return a + b",
        ["assert add(*PAIR) == 3"],
        setup_code="PAIR = (1, 2)",
    )
    assert result.ran and result.n_passed == 1


def test_no_tests_still_reports_that_it_ran():
    result = run_tests(ADD, [])
    assert result.parsed and result.ran
    assert result.n_tests == 0 and result.n_passed == 0


# --- ordinary failure modes --------------------------------------------------


def test_assertion_failure_gives_partial_credit():
    result = run_tests(ADD, ["assert add(1, 2) == 3", "assert add(1, 2) == 4"])
    assert result.parsed and result.ran
    assert (result.n_passed, result.n_tests) == (1, 2)
    assert "AssertionError" in result.stderr_tail


def test_syntax_error_never_spawns_a_process():
    result = run_tests("def add(a, b:\n    return", ADD_TESTS)
    assert not result.parsed
    assert not result.ran and result.n_passed == 0
    assert result.n_tests == 3  # denominator survives, so the reward is 0/3
    assert "SyntaxError" in result.stderr_tail
    # Parsing happens in the parent; ~3,000 samples/run makes this worth it.
    assert result.wall_time < 0.05


def test_prose_instead_of_code_is_unparseable():
    # The 0.5B failure mode the parse rung of the reward ladder exists for.
    result = run_tests("Here is a function that adds two numbers:", ADD_TESTS)
    assert not result.parsed


def test_exception_at_module_level_is_not_a_run():
    result = run_tests("raise ValueError('boom')", ADD_TESTS)
    assert result.parsed and not result.ran
    assert result.n_passed == 0
    assert "ValueError: boom" in result.stderr_tail


def test_sys_exit_is_a_candidate_failure_not_a_harness_failure():
    result = run_tests("import sys\nsys.exit(1)", ADD_TESTS)
    assert result.parsed and not result.ran
    assert result.n_passed == 0


def test_name_error_in_one_test_does_not_stop_the_others():
    result = run_tests(
        ADD, ["assert add(1, 2) == 3", "assert nope(1) == 1", "assert add(2, 2) == 4"]
    )
    assert result.n_passed == 2  # the middle one is skipped, not fatal


# --- hostile completions -----------------------------------------------------


def test_infinite_loop_is_killed():
    started = time.perf_counter()
    result = run_tests("while True:\n    pass", ADD_TESTS, timeout=1.0)
    elapsed = time.perf_counter() - started

    assert result.timed_out
    assert result.parsed and not result.ran
    assert result.n_passed == 0
    assert elapsed < 5.0, "the kill must not wait on the child"
    assert "timeout" in result.stderr_tail


def test_hang_inside_a_test_keeps_the_credit_earned_before_it():
    # The reason the child rewrites its result file after every assert: this
    # completion is 1/3 correct and a kill must not erase that.
    result = run_tests(
        "import time\ndef slow():\n    time.sleep(60)",
        ["assert True", "assert slow() is None", "assert True"],
        timeout=1.5,
    )
    assert result.timed_out
    assert result.ran, "the module body finished before the hang"
    assert result.n_passed == 1


def test_memory_bomb_is_killed():
    code = textwrap.dedent(
        """
        chunks = []
        while True:
            chunks.append(bytearray(8 * 1024 * 1024))
        """
    )
    started = time.perf_counter()
    result = run_tests(code, ["assert True"], timeout=15.0, memory_limit_mb=128)
    elapsed = time.perf_counter() - started

    assert result.parsed and not result.ran
    assert result.n_passed == 0
    # Killed on memory, so it must land well inside the wall-clock budget.
    assert elapsed < 10.0
    assert not result.timed_out
    assert "memory" in result.stderr_tail


def test_process_tree_is_killed(tmp_path):
    """A spawned grandchild must not outlive the completion that started it."""
    pid_file = tmp_path / "grandchild.pid"
    code = textwrap.dedent(
        f"""
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        open({literal(pid_file)}, "w").write(str(child.pid))
        while True:
            time.sleep(0.01)
        """
    )
    result = run_tests(code, ["assert True"], timeout=2.0)
    assert result.timed_out

    pid = int(pid_file.read_text())
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline:
        if not _alive(pid):
            break
        time.sleep(0.05)
    assert not _alive(pid), f"grandchild {pid} survived the kill"


def _alive(pid: int) -> bool:
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return False
    return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE


def test_flood_of_stderr_does_not_fill_the_disk():
    code = textwrap.dedent(
        """
        import sys
        while True:
            sys.stderr.write("x" * 4096)
        """
    )
    result = run_tests(code, ["assert True"], timeout=15.0)
    assert not result.ran
    assert result.wall_time < 10.0
    assert "stderr" in result.stderr_tail


def test_deeply_nested_source_does_not_crash_the_parent():
    # ast.parse runs in *our* process, so pathological input has to fail safe.
    result = run_tests("x = " + "(" * 200 + "1" + ")" * 200, ["assert True"])
    assert isinstance(result, ExecResult)


# --- containment -------------------------------------------------------------


def test_relative_writes_land_in_the_sandbox_and_are_deleted(tmp_path):
    """cwd is a fresh temp dir, and it is gone once the call returns."""
    cwd_file = tmp_path / "cwd.txt"
    code = textwrap.dedent(
        f"""
        import os
        open("scratch.txt", "w").write("junk")
        open({literal(cwd_file)}, "w").write(os.getcwd())
        """
    )
    # The asserts share the candidate's namespace, so `os` is already imported.
    result = run_tests(code, ["assert os.path.exists('scratch.txt')"])
    assert result.ran and result.n_passed == 1

    sandbox_cwd = cwd_file.read_text()
    assert not os.path.exists(sandbox_cwd), "sandbox directory was left behind"
    assert not os.path.exists(os.path.join(os.getcwd(), "scratch.txt"))


def test_the_result_file_is_not_visible_to_the_candidate(tmp_path):
    """Listing '.' must not reach the runner or the file it is scored by."""
    listing = tmp_path / "listing.txt"
    code = textwrap.dedent(
        f"""
        import os
        open({literal(listing)}, "w").write(repr(os.listdir(".")))
        """
    )
    run_tests(code, ["assert True"])
    assert listing.read_text() == "[]"


def test_environment_is_stripped(monkeypatch):
    monkeypatch.setenv("CARGOCULT_SECRET", "hunter2")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    result = run_tests(
        "import os\nseen = os.environ",
        [
            "assert 'CARGOCULT_SECRET' not in seen",
            "assert 'HTTPS_PROXY' not in seen",
            "assert 'PATH' in seen",  # a minimal PATH is deliberately kept
        ],
    )
    assert result.n_passed == 3


def test_home_points_into_the_sandbox(tmp_path):
    """`~` has to resolve somewhere disposable: candidates write there."""
    home_file = tmp_path / "home.txt"
    code = textwrap.dedent(
        f"""
        import os
        open({literal(home_file)}, "w").write(os.path.expanduser("~"))
        """
    )
    run_tests(code, ["assert True"])
    assert not os.path.exists(home_file.read_text())


def test_absolute_path_write_is_not_blocked(tmp_path):
    """DOCUMENTED LIMITATION: this is not a filesystem jail.

    An absolute path outside the sandbox is still writable. Nothing here
    stops `open('/etc/passwd', 'w')` — only the fact that the temp dir is
    the cwd keeps the ordinary case contained. If this test ever fails
    because writes started being blocked, update the sandbox docstring.
    """
    escaped = tmp_path / "escaped.txt"
    result = run_tests(
        f"open({literal(escaped)}, 'w').write('i am outside')", ["assert True"]
    )
    assert result.ran
    assert escaped.read_text() == "i am outside"


def test_network_is_not_blocked():
    """DOCUMENTED LIMITATION: proxy vars are unset, sockets still work."""
    result = run_tests(
        "import socket\ns = socket.socket()", ["assert s is not None"]
    )
    assert result.ran and result.n_passed == 1


# --- usable from a process pool ----------------------------------------------


def test_result_is_picklable():
    result = run_tests(ADD, ADD_TESTS)
    assert pickle.loads(pickle.dumps(result)) == result


def test_runs_under_a_process_pool_executor():
    # Phase 2 overlaps sandbox scoring with GPU generation this way.
    codes = [ADD, "def add(a, b):\n    return a - b", "def add(a, b:"]
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run_tests, codes, [ADD_TESTS] * 3))
    assert [r.n_passed for r in results] == [3, 1, 0]  # add(0,0)==0 holds for a-b
    assert [r.parsed for r in results] == [True, True, False]


# --- defaults ----------------------------------------------------------------


def test_defaults_match_the_spec():
    assert DEFAULT_TIMEOUT == 6.0
    assert DEFAULT_MEMORY_MB >= 256


@pytest.mark.skipif(sys.platform != "win32", reason="windows-only kill semantics")
def test_no_console_window_flag_is_set():
    from src.sandbox import _CREATION_FLAGS

    assert _CREATION_FLAGS & 0x08000000  # CREATE_NO_WINDOW
