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

_interrupted = False
_current_proc: subprocess.Popen | None = None


def _keypress_monitor():
    global _interrupted
    while not _interrupted:
        if msvcrt.kbhit():
            if msvcrt.getwch() == "q":
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
        print("\n  Stopped (q pressed).", flush=True)
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
    model: str | None = None
    max_iterations: int = 10
    cooldown_seconds: int = 10
    default_wait_seconds: int = 300
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
        "Read CLAUDE.md, then search every source file for bugs. For each bug found, "
        "make the smallest correct fix. Look for: off-by-one errors, logic errors, "
        "edge cases (empty inputs, None, zero, negative), race conditions, resource "
        "leaks, swallowed exceptions, incorrect boundary checks, and silent data "
        "truncation. Do not mask bugs with clamping or bounds checks.",
    ),
    Task(
        "Test coverage",
        "Read CLAUDE.md, then run the test suite with coverage. For every function or "
        "branch below 90% coverage, write focused unit tests. Follow the layout "
        "tests/foo/test_bar.py for foo/bar.py. Each assertion must check an exact "
        "expected value, not just truthiness or type. When asserting on a subset also "
        "assert the total count. Prioritize: error paths, boundary values (zero, empty, "
        "max, negative, one-off), and uncommon but valid inputs.",
    ),
    Task(
        "Conciseness",
        "Read CLAUDE.md, then make the codebase more concise without changing behavior. "
        "Remove dead code, unused imports, unreachable branches, commented-out code, and "
        "deprecated APIs. Inline trivial one-call functions that add no clarity. Replace "
        "deep nesting with early returns. Do not extract helpers unless logic is "
        "duplicated 3+ times.",
    ),
    Task(
        "Optimization",
        "Read CLAUDE.md, then profile or inspect the codebase for performance issues. "
        "Fix: O(n^2)+ algorithms that can be O(n log n) or O(n), repeated lookups that "
        "should be cached, unnecessary copies of large objects, allocations inside tight "
        "loops, redundant recomputation. For I/O-bound thread pools set "
        "max_workers=os.cpu_count(). Do not sacrifice readability for marginal gains.",
    ),
    Task(
        "Config",
        "Read CLAUDE.md, then find hardcoded constants, magic numbers, URLs, timeouts, "
        "thresholds, and tuning parameters in source files. Move each to a YAML config "
        "file (not JSON/TOML/INI). Reuse the existing config loader if one exists; "
        "otherwise create one. Replace every hardcoded value with a config read.",
    ),
    Task(
        "Markdown",
        "Read CLAUDE.md, then review every markdown file in the repo. Fix factual "
        "inaccuracies, stale sections, and incorrect command examples so they reflect "
        "the current code. Fix broken links and inconsistent formatting. Remove "
        "duplication between files. Keep wording concise. Do not add new sections.",
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

    t_out = threading.Thread(
        target=lambda: stdout_chunks.append(proc.stdout.read()), daemon=True
    )
    t_err = threading.Thread(
        target=lambda: stderr_chunks.append(proc.stderr.read()), daemon=True
    )
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
        self._last_session_id: str | None = None

    def _base_args(self) -> list[str]:
        return [
            "claude",
            "-p",
            "--dangerously-skip-permissions",
            *(["--model", self.config.model] if self.config.model else []),
            "--output-format",
            "json",
        ]

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

    _RATE_LIMIT_RE = re.compile(
        r"you've hit your limit|usage limit|rate limit|exceeded.*limit|too many requests",
        re.IGNORECASE,
    )

    def _looks_like_rate_limit(self, text: str) -> bool:
        return bool(self._RATE_LIMIT_RE.search(text))

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
            if iteration > 1 and self.config.cooldown_seconds > 0:
                for _ in range(self.config.cooldown_seconds):
                    check_interrupt()
                    time.sleep(1)
            iterations = iteration
            self._output(f"\n  --- {task.name} - iteration {iteration} ---")

            base_prompt = (
                task.prompt if iteration == 1 else "Keep going with the same task."
            )
            result = self.runner.invoke(
                f"{base_prompt} {self.config.suffix}",
                continue_session=iteration > 1,
            )

            self._output(f"\n{result.output}\n")

            if not result.succeeded:
                self._output(f"  FAIL: Claude exited with code {result.exit_code}")
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
        default=10,
        help="Max iterations per task (default: 10)",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=10,
        help="Seconds to wait between iterations to avoid rate limits (default: 10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = RunConfig(
        model=args.model,
        max_iterations=args.max_iterations,
        cooldown_seconds=args.cooldown,
    )

    if args.prompts:
        tasks = [Task(f"Task {i + 1}", p) for i, p in enumerate(args.prompts)]
    else:
        tasks = DEFAULT_TASKS

    print("  Press q to stop.\n", flush=True)
    orchestrator = TaskOrchestrator(tasks, config)
    results = orchestrator.run_all()

    if any(r.status == TaskStatus.FAILED for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
