#!/usr/bin/env python3
"""Iterative Claude Code task runner with git integration."""

from __future__ import annotations

import argparse
import json
import msvcrt
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from constants import load_config

_cfg = load_config()["iterate"]
_interrupted = False
_current_proc: subprocess.Popen | None = None


def _keypress_monitor():
    global _interrupted
    while not _interrupted:
        if msvcrt.kbhit() and msvcrt.getwch() == "q":
            _interrupted = True
            return
        time.sleep(_cfg["poll_interval"])


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
    max_iterations: int = _cfg["max_iterations"]
    cooldown_seconds: int = _cfg["cooldown_seconds"]
    default_wait_seconds: int = _cfg["default_wait_seconds"]
    rate_limit_padding_seconds: int = _cfg["rate_limit_padding_seconds"]
    log_file: Path = field(default_factory=lambda: Path.cwd() / _cfg["log_file"])
    suffix: str = _cfg["suffix"]
    continuation_prompt: str = _cfg["continuation_prompt"]


DEFAULT_TASKS = [
    Task(
        "Bug fixes",
        "search every source file for bugs and make the smallest correct fix. "
        "Look for: off-by-one errors, logic errors, edge cases (empty, None, zero, "
        "negative), race conditions, resource leaks, swallowed exceptions. "
        "Do not mask bugs with clamping or bounds checks.",
    ),
    Task(
        "Test coverage",
        "run the test suite with coverage. For every function or branch below 90%, "
        "write focused unit tests in tests/foo/test_bar.py for foo/bar.py. "
        "Assert exact expected values, not just truthiness. "
        "Prioritize error paths, boundary values, and uncommon but valid inputs.",
    ),
    Task(
        "Conciseness",
        "aggressively reduce codebase size without changing behavior or sacrificing "
        "performance. Remove dead code, unused imports, commented-out code, unnecessary "
        "comments. Inline trivial functions, use comprehensions and ternaries, merge "
        "duplicate logic, replace nesting with early returns.",
    ),
    Task(
        "Optimization",
        "inspect the codebase for performance issues. Fix: O(n^2)+ algorithms, "
        "repeated lookups that should be cached, unnecessary copies, allocations "
        "in tight loops, redundant recomputation. "
        "Do not sacrifice readability for marginal gains.",
    ),
    Task(
        "Config",
        "find hardcoded constants, magic numbers, URLs, timeouts, and tuning "
        "parameters in source files. Move each to a YAML config file. "
        "Reuse the existing config loader if one exists; otherwise create one. "
        "Replace every hardcoded value with a config read.",
    ),
    Task(
        "Markdown",
        "review every markdown file. Fix factual inaccuracies, stale sections, "
        "and incorrect command examples so they reflect the current code. "
        "Fix broken links, remove duplication between files. "
        "Keep wording concise. Do not add new sections.",
    ),
]

TASK_KEYS = ["bugs", "tests", "concise", "optimize", "config", "markdown"]
TASK_MAP = dict(zip(TASK_KEYS, DEFAULT_TASKS, strict=True))


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
    git("add", "-A", "--", ".", f":!{_cfg['log_file']}", check=False)
    result = git("diff", "--cached", "--quiet", check=False)
    if result.returncode != 0:
        git("commit", "-m", message)
        return True
    return False


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


_SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".ruff_cache",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
}


def _has_recent_edit(since: float) -> bool:
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for f in files:
            try:
                if os.path.getmtime(os.path.join(root, f)) > since:
                    return True
            except OSError:
                pass
    return False


def run_subprocess(args: list[str]) -> subprocess.CompletedProcess:
    global _current_proc
    f_out = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    f_err = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    try:
        proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=f_out, stderr=f_err)
        proc.stdin.close()
        _current_proc = proc

        last_size = 0
        last_activity = time.monotonic()
        last_edit_wall = time.time()
        edit_check_interval = _cfg["edit_check_interval"]
        next_edit_check = last_activity + edit_check_interval
        stall_timeout = _cfg["stall_timeout_seconds"]

        while proc.poll() is None:
            check_interrupt()
            time.sleep(_cfg["poll_interval"])
            now = time.monotonic()

            current_size = f_out.tell() + f_err.tell()
            if current_size != last_size:
                last_size = current_size
                last_activity = now

            if now >= next_edit_check:
                next_edit_check = now + edit_check_interval
                if _has_recent_edit(last_edit_wall):
                    last_edit_wall = time.time()
                    last_activity = now

            if now - last_activity > stall_timeout:
                print(
                    f"  Process stalled for {stall_timeout}s, killing...",
                    flush=True,
                )
                proc.kill()
                proc.wait()
                break

        check_interrupt()
        _current_proc = None

        f_out.seek(0)
        f_err.seek(0)
        stdout = f_out.read()
        stderr = f_err.read()
        return subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)
    finally:
        f_out.close()
        f_err.close()


class ClaudeRunner:
    def __init__(self, config: RunConfig):
        self.config = config

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
        padding = self.config.rate_limit_padding_seconds
        if match := re.search(r"(\d{1,2})(?::(\d{2}))?\s*([APap][Mm])", text):
            try:
                time_str = f"{match.group(1)}:{match.group(2) or '00'} {match.group(3).upper()}"
                now = datetime.now()
                reset_time = datetime.strptime(time_str, "%I:%M %p").replace(
                    year=now.year, month=now.month, day=now.day
                )
                if (wait := int((reset_time - now).total_seconds()) + padding) > 0:
                    return wait
            except ValueError:
                pass
        if match := re.search(r"(\d+)\s*minutes?", text):
            return int(match.group(1)) * 60 + padding
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
        with self.config.log_file.open("a", encoding="utf-8") as f:
            f.write(text + "\n")

    def _format_pass(self) -> None:
        self._output("  Running formatter...")
        check_interrupt()
        result = self.runner.invoke("Run the code formatter.")
        self._output(f"\n{result.output}\n")
        if result.succeeded:
            commit_changes("format")

    def run_all(self) -> list[TaskResult]:
        self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.log_file.write_text("# Iteration Log\n\n", encoding="utf-8")
        overall_start = time.monotonic()

        self._format_pass()

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
        restart = True

        for iteration in range(1, self.config.max_iterations + 1):
            check_interrupt()
            if iteration > 1 and self.config.cooldown_seconds > 0:
                for _ in range(self.config.cooldown_seconds):
                    check_interrupt()
                    time.sleep(1)
            iterations = iteration
            self._output(f"\n  --- {task.name} - iteration {iteration} ---")

            for _attempt in range(2):
                base_prompt = (
                    f"Read CLAUDE.md, then {task.prompt}"
                    if restart
                    else self.config.continuation_prompt
                )
                result = self.runner.invoke(
                    f"{base_prompt} {self.config.suffix}",
                    continue_session=not restart,
                )
                if (
                    not result.succeeded
                    and not restart
                    and "prompt is too long" in result.output.lower()
                ):
                    self._output("  Prompt too long, restarting fresh session...")
                    restart = True
                    continue
                break

            self._output(f"\n{result.output}\n")

            if not result.succeeded:
                self._output(f"  FAIL: Claude exited with code {result.exit_code}")
                restart = True
            else:
                restart = False
                if result.signalled_no_changes:
                    elapsed = (time.monotonic() - task_start) / 60
                    self._output(
                        f"  Converged after {iteration} iteration(s) ({elapsed:.1f}m)"
                    )
                    status = TaskStatus.CONVERGED

            commit_changes(f"{task.name} - iteration {iteration}")
            if status == TaskStatus.CONVERGED:
                break

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
            TaskStatus.MAX_ITERATIONS: "\033[33m",
        }
        reset = "\033[0m"

        self._output(f"\n{'=' * 60}")
        self._output(f"  Summary ({elapsed_minutes:.1f}m)")
        self._output(f"{'=' * 60}")
        for r in self.results:
            tail = f"{r.name} ({r.iterations} iter, {r.elapsed_minutes:.1f}m)"
            c = colors.get(r.status, "")
            print(f"  {c}{r.status.value:<15}{reset} {tail}")
            with self.config.log_file.open("a", encoding="utf-8") as f:
                f.write(f"  {r.status.value:<15} {tail}\n")
        self._output("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative Claude Code task runner with git integration.",
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
        default=_cfg["max_iterations"],
        help=f"Max iterations per task (default: {_cfg['max_iterations']})",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=_cfg["cooldown_seconds"],
        help=f"Seconds to wait between iterations (default: {_cfg['cooldown_seconds']})",
    )
    parser.add_argument(
        "--task",
        "-t",
        action="append",
        dest="tasks",
        choices=TASK_KEYS,
        help=f"Run only these default tasks (choices: {', '.join(TASK_KEYS)}). Repeatable.",
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
    elif args.tasks:
        tasks = [TASK_MAP[k] for k in args.tasks]
    else:
        tasks = DEFAULT_TASKS

    print("  Press q to stop.\n", flush=True)
    orchestrator = TaskOrchestrator(tasks, config)
    orchestrator.run_all()


if __name__ == "__main__":
    main()
