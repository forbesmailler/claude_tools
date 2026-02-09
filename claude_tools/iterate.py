#!/usr/bin/env python3
"""
Iterative Claude Code task runner with git integration.

Runs a series of coding tasks via Claude Code in print mode (-p), committing
after each successful iteration, squashing per task, and retrying on rate limits.

Usage:
    python iterate.py                          # Run default tasks
    python iterate.py --model opus             # Specify model
    python iterate.py -p "Fix all TODOs" -p "Add docstrings"  # Custom prompts
    python iterate.py --max-iterations 10      # Override iteration cap
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import json
import msvcrt
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Ctrl+C handling for Windows/PowerShell
#
# Disable ENABLE_PROCESSED_INPUT on the console so Ctrl+C is placed in the
# input buffer as character 0x03 instead of generating a CTRL_C_EVENT.  A
# daemon thread polls msvcrt.kbhit()/getwch() for that character and sets
# _interrupted.  check_interrupt() is called at every loop boundary.
# ---------------------------------------------------------------------------

_interrupted = False
_current_proc: Optional[subprocess.Popen] = None

_kernel32 = ctypes.windll.kernel32
_stdin_handle = _kernel32.GetStdHandle(ctypes.c_ulong(-10 & 0xFFFFFFFF))
_original_mode = ctypes.c_ulong()
_kernel32.GetConsoleMode(_stdin_handle, ctypes.byref(_original_mode))
_kernel32.SetConsoleMode(_stdin_handle, _original_mode.value & ~0x0001)


def _restore_console():
    _kernel32.SetConsoleMode(_stdin_handle, _original_mode.value)


atexit.register(_restore_console)


def _keypress_monitor():
    global _interrupted
    while not _interrupted:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch == "\x03":
                _interrupted = True
                return
        time.sleep(0.1)


threading.Thread(target=_keypress_monitor, daemon=True).start()


def check_interrupt():
    global _current_proc
    if _interrupted:
        if _current_proc and _current_proc.poll() is None:
            _current_proc.kill()
            _current_proc.wait()
            _current_proc = None
        _restore_console()
        print("\n  Interrupted by user.", flush=True)
        sys.exit(130)


class TaskStatus(Enum):
    CONVERGED = "converged"
    FAILED = "failed"
    MAX_ITERATIONS = "max iterations"


@dataclass
class Task:
    name: str
    prompt: str


@dataclass
class TaskResult:
    name: str
    status: TaskStatus
    iterations: int = 0
    elapsed_minutes: float = 0.0


@dataclass
class RunConfig:
    model: Optional[str] = None
    max_iterations: int = 20
    default_wait_seconds: int = 900
    log_file: Path = field(
        default_factory=lambda: Path.cwd() / "logs" / "iterate_log.md"
    )
    suffix: str = (
        "After making changes, run all tests and the code formatter. "
        "Only make changes you are confident are correct. "
        "If no changes are needed, respond with exactly NO_CHANGES and nothing else."
    )


DEFAULT_TASKS = [
    Task(
        "Bug fixes",
        "Search every source file for bugs: off-by-one errors, logic errors, edge cases, "
        "race conditions, resource leaks, incorrect boundary checks, silent data "
        "truncation, and swallowed exceptions. Do not rely on bounds or clamping to mask "
        "bugs; extreme values indicate bugs to fix. Make the smallest fix that corrects "
        "each issue.",
    ),
    Task(
        "Test coverage",
        "Run the test suite with coverage to find uncovered functions and branches. "
        "Target 90%+ line coverage. Write focused unit tests following the layout "
        "tests/foo/test_bar.py matching foo/bar.py. Assert exact expected values, not "
        "just truthiness or types. When checking subsets, also assert the total count. "
        "Cover: error handling paths, boundary values (zero, empty, max, negative), "
        "off-by-one boundaries, and uncommon but valid inputs.",
    ),
    Task(
        "Conciseness",
        "Make the codebase more concise without changing behavior. Remove dead code, "
        "unused imports, unreachable branches, commented-out code, and deprecated APIs. "
        "Inline trivial functions that are called once and add no clarity. Replace deep "
        "nesting with early returns or guard clauses. Prefer f-strings, context managers, "
        "dataclasses, and walrus operator where they simplify. Only extract shared helpers "
        "when there are 3+ duplications.",
    ),
    Task(
        "Optimization",
        "Find performance bottlenecks: O(n^2)+ algorithms that could be O(n log n) or "
        "O(n), repeated lookups that should be cached, unnecessary copies of large "
        "objects, allocations inside tight loops, and redundant recomputation. For "
        "I/O-bound thread pools, set max_workers=os.cpu_count(). Apply targeted fixes. "
        "Do not sacrifice readability for marginal gains.",
    ),
    Task(
        "Config",
        "Find hardcoded numeric constants, magic numbers, string literals, URLs, "
        "timeouts, thresholds, and tuning parameters in source files. Move each to a "
        "YAML config file (not JSON/TOML/INI). If a config loading mechanism already "
        "exists, use it; otherwise create one. Replace the hardcoded values with reads "
        "from the config.",
    ),
    Task(
        "Markdown",
        "Review every markdown file in the repo (README.md, CLAUDE.md, etc.). Fix "
        "factual inaccuracies, stale sections, and incorrect command examples so they "
        "reflect the current code. Fix broken links and inconsistent formatting. Remove "
        "duplication between files. Keep wording concise. Do not add new sections or "
        "boilerplate.",
    ),
]


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=check,
    )


def git_head_sha() -> str:
    return git("rev-parse", "HEAD").stdout.strip()


def commit_changes(message: str) -> bool:
    git("add", "-A", "--", ".", ":!logs/iterate_log.md", check=False)
    result = git("diff", "--cached", "--quiet", check=False)
    if result.returncode != 0:
        git("commit", "-m", message)
        return True
    return False


def discard_changes() -> None:
    git("checkout", "--", ".", check=False)
    git("clean", "-fd", check=False)


def squash_task_commits(base_sha: str, message: str) -> None:
    if git_head_sha() != base_sha:
        git("reset", "--soft", base_sha)
        git("commit", "-m", message)


@dataclass
class ClaudeResult:
    output: str
    exit_code: int

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0

    @property
    def signalled_no_changes(self) -> bool:
        return self.succeeded and "NO_CHANGES" in self.output


def run_subprocess(args: list[str]) -> subprocess.CompletedProcess:
    """Run a subprocess while remaining responsive to Ctrl+C.

    Uses Popen + daemon reader threads so the main thread polls with
    short sleeps and can check the _interrupted flag between them.
    """
    global _current_proc
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _current_proc = proc
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def _read(stream, dest):
        dest.append(stream.read())

    t_out = threading.Thread(target=_read, args=(proc.stdout, stdout_chunks), daemon=True)
    t_err = threading.Thread(target=_read, args=(proc.stderr, stderr_chunks), daemon=True)
    t_out.start()
    t_err.start()

    while proc.poll() is None:
        check_interrupt()
        time.sleep(0.5)

    check_interrupt()
    t_out.join()
    t_err.join()
    _current_proc = None

    stdout = stdout_chunks[0] if stdout_chunks else ""
    stderr = stderr_chunks[0] if stderr_chunks else ""
    return subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)


class ClaudeRunner:
    def __init__(self, config: RunConfig):
        self.config = config
        self._last_session_id: Optional[str] = None

    def _base_args(self) -> list[str]:
        args = ["claude", "-p", "--dangerously-skip-permissions"]
        if self.config.model:
            args += ["--model", self.config.model]
        args += ["--output-format", "json"]
        return args

    def _parse_rate_limit_wait(self, text: str) -> int:
        match = re.search(r"(\d{1,2}:\d{2}\s*(?:AM|PM))", text, re.IGNORECASE)
        if match:
            try:
                reset_time = datetime.strptime(match.group(1), "%I:%M %p").replace(
                    year=datetime.now().year,
                    month=datetime.now().month,
                    day=datetime.now().day,
                )
                if reset_time < datetime.now():
                    reset_time += timedelta(days=1)
                return int((reset_time - datetime.now()).total_seconds()) + 30
            except ValueError:
                pass
        match = re.search(r"(\d+)\s*minutes?", text)
        if match:
            return int(match.group(1)) * 60 + 30
        return self.config.default_wait_seconds

    def _looks_like_rate_limit(self, text: str) -> bool:
        patterns = [
            r"you've hit your limit",
            r"usage limit",
            r"rate limit",
            r"exceeded.*limit",
            r"too many requests",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def invoke(self, prompt: str, continue_session: bool = False) -> ClaudeResult:
        args = self._base_args()
        if continue_session:
            args.append("--continue")
        args.append(prompt)

        while True:
            check_interrupt()
            result = run_subprocess(args)
            check_interrupt()
            combined = result.stdout + result.stderr

            if self._looks_like_rate_limit(combined):
                wait = self._parse_rate_limit_wait(combined)
                resume_at = (datetime.now() + timedelta(seconds=wait)).strftime(
                    "%I:%M %p"
                )
                print(
                    f"  Rate limit hit. Waiting {wait / 60:.1f} min (until ~{resume_at})...",
                    flush=True,
                )
                for _ in range(wait):
                    check_interrupt()
                    time.sleep(1)
                print("  Resuming...", flush=True)
                continue

            output_text = combined
            try:
                data = json.loads(result.stdout)
                output_text = data.get("result", result.stdout)
                if session_id := data.get("session_id"):
                    self._last_session_id = session_id
            except (json.JSONDecodeError, TypeError):
                output_text = result.stdout or result.stderr

            return ClaudeResult(output=output_text, exit_code=result.returncode)


class TaskOrchestrator:
    def __init__(self, tasks: list[Task], config: RunConfig):
        self.tasks = tasks
        self.config = config
        self.runner = ClaudeRunner(config)
        self.results: list[TaskResult] = []

    def _output(self, text: str) -> None:
        print(text)
        with self.config.log_file.open("a") as f:
            f.write(text + "\n")

    def run_all(self) -> list[TaskResult]:
        self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.log_file.write_text("# Iteration Log\n\n")
        overall_start = time.monotonic()

        for task in self.tasks:
            check_interrupt()
            result = self._run_task(task)
            self.results.append(result)

        elapsed = (time.monotonic() - overall_start) / 60
        self._print_summary(elapsed)
        return self.results

    def _run_task(self, task: Task) -> TaskResult:
        self._output(f"\n{'=' * 60}")
        self._output(f"  Task: {task.name}")
        self._output(f"{'=' * 60}")

        task_start = time.monotonic()
        base_sha = git_head_sha()
        status = TaskStatus.MAX_ITERATIONS
        iterations = 0

        for iteration in range(1, self.config.max_iterations + 1):
            check_interrupt()
            iterations = iteration
            self._output(f"\n  --- {task.name} - iteration {iteration} ---")

            if iteration == 1:
                prompt = f"{task.prompt} {self.config.suffix}"
                result = self.runner.invoke(prompt, continue_session=False)
            else:
                prompt = f"Keep going with the same task. {self.config.suffix}"
                result = self.runner.invoke(prompt, continue_session=True)

            self._output(f"\n{result.output}\n")

            if not result.succeeded:
                self._output(
                    f"  FAIL: Claude exited with code {result.exit_code}"
                )
                status = TaskStatus.FAILED
                discard_changes()
                break

            if result.signalled_no_changes:
                elapsed = (time.monotonic() - task_start) / 60
                self._output(
                    f"  Converged after {iteration} iteration(s) ({elapsed:.1f}m)"
                )
                status = TaskStatus.CONVERGED
                commit_changes(f"{task.name} - iteration {iteration}")
                break

            commit_changes(f"{task.name} - iteration {iteration}")

        if status == TaskStatus.MAX_ITERATIONS:
            self._output(
                f"  Hit max iterations ({self.config.max_iterations}) for: {task.name}"
            )

        squash_task_commits(base_sha, f"{task.name} - automated iteration")

        elapsed = (time.monotonic() - task_start) / 60
        return TaskResult(task.name, status, iterations, elapsed)

    def _print_summary(self, elapsed_minutes: float) -> None:
        colors = {
            TaskStatus.CONVERGED: "\033[32m",
            TaskStatus.FAILED: "\033[31m",
            TaskStatus.MAX_ITERATIONS: "\033[33m",
        }
        reset = "\033[0m"

        self._output(f"\n{'=' * 60}")
        self._output(f"  Summary ({elapsed_minutes:.1f}m)")
        self._output(f"{'=' * 60}")
        for r in self.results:
            c = colors.get(r.status, "")
            print(
                f"  {c}{r.status.value:<15}{reset} {r.name} "
                f"({r.iterations} iter, {r.elapsed_minutes:.1f}m)"
            )
            with self.config.log_file.open("a") as f:
                f.write(
                    f"  {r.status.value:<15} {r.name} "
                    f"({r.iterations} iter, {r.elapsed_minutes:.1f}m)\n"
                )
        self._output("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative Claude Code task runner with git integration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", "-m", help="Claude model to use (e.g. opus, sonnet)")
    parser.add_argument(
        "--prompt",
        "-p",
        action="append",
        dest="prompts",
        help="Custom task prompt (can be repeated). Overrides default tasks.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Max iterations per task (default: 20)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = RunConfig(
        model=args.model,
        max_iterations=args.max_iterations,
    )

    if args.prompts:
        tasks = [Task(f"Task {i + 1}", p) for i, p in enumerate(args.prompts)]
    else:
        tasks = DEFAULT_TASKS

    orchestrator = TaskOrchestrator(tasks, config)
    results = orchestrator.run_all()

    if any(r.status == TaskStatus.FAILED for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
