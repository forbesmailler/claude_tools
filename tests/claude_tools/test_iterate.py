"""Tests for claude_tools.iterate."""

import subprocess
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claude_tools.iterate import (
    DEFAULT_TASKS,
    TASK_KEYS,
    TASK_MAP,
    ClaudeResult,
    ClaudeRunner,
    RunConfig,
    Task,
    TaskOrchestrator,
    TaskResult,
    TaskStatus,
    _has_recent_edit,
    _keypress_monitor,
    check_interrupt,
    commit_changes,
    git,
    git_head_sha,
    main,
    parse_args,
    run_subprocess,
    squash_task_commits,
)


class TestTaskStatus:
    def test_values(self):
        assert TaskStatus.CONVERGED.value == "converged"
        assert TaskStatus.MAX_ITERATIONS.value == "max iterations"

    def test_all_members(self):
        assert len(TaskStatus) == 2


class TestTask:
    def test_fields(self):
        t = Task(name="Bug fix", prompt="Fix bugs")
        assert t.name == "Bug fix"
        assert t.prompt == "Fix bugs"


class TestTaskResult:
    def test_defaults(self):
        r = TaskResult(name="test", status=TaskStatus.CONVERGED)
        assert r.iterations == 0
        assert r.elapsed_minutes == 0.0

    def test_custom_values(self):
        r = TaskResult(
            "t", TaskStatus.MAX_ITERATIONS, iterations=5, elapsed_minutes=12.3
        )
        assert r.name == "t"
        assert r.status == TaskStatus.MAX_ITERATIONS
        assert r.iterations == 5
        assert r.elapsed_minutes == 12.3


class TestRunConfig:
    def test_defaults(self):
        c = RunConfig()
        assert c.model is None
        assert c.max_iterations == 10
        assert c.cooldown_seconds == 60
        assert c.default_wait_seconds == 300
        assert c.rate_limit_padding_seconds == 60
        assert c.log_file == Path.cwd() / "logs" / "iterate_log.md"
        assert "NO_CHANGES" in c.suffix
        assert c.continuation_prompt == "Keep going with the same task."

    def test_custom(self):
        c = RunConfig(
            model="opus",
            max_iterations=5,
            default_wait_seconds=60,
            rate_limit_padding_seconds=10,
        )
        assert c.model == "opus"
        assert c.max_iterations == 5
        assert c.default_wait_seconds == 60
        assert c.rate_limit_padding_seconds == 10


class TestClaudeResult:
    def test_succeeded_true(self):
        r = ClaudeResult(output="done", exit_code=0)
        assert r.succeeded is True

    def test_succeeded_false(self):
        r = ClaudeResult(output="error", exit_code=1)
        assert r.succeeded is False

    def test_signalled_no_changes_true(self):
        r = ClaudeResult(output="NO_CHANGES", exit_code=0)
        assert r.signalled_no_changes is True

    def test_signalled_no_changes_with_surrounding_text(self):
        r = ClaudeResult(output="Result: NO_CHANGES found", exit_code=0)
        assert r.signalled_no_changes is True

    def test_signalled_no_changes_false_when_failed(self):
        r = ClaudeResult(output="NO_CHANGES", exit_code=1)
        assert r.signalled_no_changes is False

    def test_signalled_no_changes_false_when_not_present(self):
        r = ClaudeResult(output="Changes made", exit_code=0)
        assert r.signalled_no_changes is False

    def test_empty_output(self):
        r = ClaudeResult(output="", exit_code=0)
        assert r.succeeded is True
        assert r.signalled_no_changes is False


class TestGit:
    @patch("claude_tools.iterate.subprocess.run")
    def test_git_passes_args(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            ["git", "status"], 0, "clean", ""
        )
        result = git("status")
        mock_run.assert_called_once_with(
            ["git", "status"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout == "clean"

    @patch("claude_tools.iterate.subprocess.run")
    def test_git_check_false(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(["git", "diff"], 1, "", "")
        result = git("diff", "--cached", "--quiet", check=False)
        mock_run.assert_called_once_with(
            ["git", "diff", "--cached", "--quiet"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1

    @patch("claude_tools.iterate.subprocess.run")
    def test_git_multiple_args(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            ["git", "log", "--oneline", "-5"], 0, "abc123", ""
        )
        git("log", "--oneline", "-5")
        mock_run.assert_called_once_with(
            ["git", "log", "--oneline", "-5"],
            capture_output=True,
            text=True,
            check=True,
        )


class TestGitHeadSha:
    @patch("claude_tools.iterate.git")
    def test_returns_stripped_sha(self, mock_git):
        mock_git.return_value = subprocess.CompletedProcess([], 0, "abc123\n", "")
        assert git_head_sha() == "abc123"

    @patch("claude_tools.iterate.git")
    def test_strips_whitespace(self, mock_git):
        mock_git.return_value = subprocess.CompletedProcess([], 0, "  def456  \n", "")
        assert git_head_sha() == "def456"


class TestCommitChanges:
    @patch("claude_tools.iterate.git")
    def test_commits_when_staged_changes(self, mock_git):
        mock_git.side_effect = [
            None,  # git add
            subprocess.CompletedProcess([], 1, "", ""),  # diff --cached (changes)
            None,  # commit
        ]
        assert commit_changes("test message") is True
        assert mock_git.call_count == 3
        mock_git.assert_any_call(
            "add", "-A", "--", ".", ":!logs/iterate_log.md", check=False
        )
        mock_git.assert_any_call("commit", "-m", "test message")

    @patch("claude_tools.iterate.git")
    def test_no_commit_when_clean(self, mock_git):
        mock_git.side_effect = [
            None,  # git add
            subprocess.CompletedProcess([], 0, "", ""),  # diff --cached (clean)
        ]
        assert commit_changes("msg") is False
        assert mock_git.call_count == 2


class TestSquashTaskCommits:
    @patch("claude_tools.iterate.git")
    @patch("claude_tools.iterate.git_head_sha", return_value="newsha")
    def test_squashes_when_head_changed(self, mock_sha, mock_git):
        squash_task_commits("oldsha", "squash msg")
        mock_git.assert_any_call("reset", "--soft", "oldsha")
        mock_git.assert_any_call("commit", "-m", "squash msg")

    @patch("claude_tools.iterate.git")
    @patch("claude_tools.iterate.git_head_sha", return_value="same")
    def test_no_squash_when_head_unchanged(self, mock_sha, mock_git):
        squash_task_commits("same", "msg")
        mock_git.assert_not_called()


class TestClaudeRunnerBaseArgs:
    def test_base_args_no_model(self):
        config = RunConfig(model=None)
        runner = ClaudeRunner(config)
        args = runner._base_args()
        assert args == [
            "claude",
            "-p",
            "--dangerously-skip-permissions",
        ]

    def test_base_args_with_model(self):
        config = RunConfig(model="opus")
        runner = ClaudeRunner(config)
        args = runner._base_args()
        assert args == [
            "claude",
            "-p",
            "--dangerously-skip-permissions",
            "--model",
            "opus",
        ]


class TestLooksLikeRateLimit:
    def setup_method(self):
        self.runner = ClaudeRunner(RunConfig())

    def test_hits_limit_pattern(self):
        assert self.runner._looks_like_rate_limit("you've hit your limit") is True

    def test_usage_limit(self):
        assert self.runner._looks_like_rate_limit("Usage limit reached") is True

    def test_rate_limit(self):
        assert self.runner._looks_like_rate_limit("Rate limit exceeded") is True

    def test_exceeded_limit(self):
        assert (
            self.runner._looks_like_rate_limit("You have exceeded your limit") is True
        )

    def test_too_many_requests(self):
        assert self.runner._looks_like_rate_limit("Too many requests") is True

    def test_no_match(self):
        assert self.runner._looks_like_rate_limit("All good") is False

    def test_empty_string(self):
        assert self.runner._looks_like_rate_limit("") is False

    def test_case_insensitive(self):
        assert self.runner._looks_like_rate_limit("RATE LIMIT") is True


class TestParseRateLimitWait:
    def setup_method(self):
        self.runner = ClaudeRunner(RunConfig(default_wait_seconds=900))

    def test_minutes_pattern(self):
        wait = self.runner._parse_rate_limit_wait("Please wait 5 minutes")
        assert wait == 5 * 60 + 60

    def test_minute_singular(self):
        wait = self.runner._parse_rate_limit_wait("Wait 1 minute")
        assert wait == 1 * 60 + 60

    @patch("claude_tools.iterate.datetime")
    def test_time_pattern_future(self, mock_dt):
        fixed_now = datetime(2025, 6, 15, 14, 30, 0)
        mock_dt.now.return_value = fixed_now
        mock_dt.strptime = datetime.strptime
        future = fixed_now + timedelta(minutes=10)
        time_str = future.strftime("%I:%M %p")
        wait = self.runner._parse_rate_limit_wait(f"Resets at {time_str}")
        assert 8 * 60 <= wait <= 12 * 60

    @patch("claude_tools.iterate.datetime")
    def test_time_pattern_past_falls_through_to_default(self, mock_dt):
        fixed_now = datetime(2025, 6, 15, 14, 30, 0)
        mock_dt.now.return_value = fixed_now
        mock_dt.strptime = datetime.strptime
        past = fixed_now - timedelta(minutes=5)
        time_str = past.strftime("%I:%M %p")
        wait = self.runner._parse_rate_limit_wait(f"Resets at {time_str}")
        assert wait == 900

    def test_no_pattern_returns_default(self):
        wait = self.runner._parse_rate_limit_wait("Rate limited, try later")
        assert wait == 900

    def test_custom_default_wait(self):
        runner = ClaudeRunner(RunConfig(default_wait_seconds=300))
        wait = runner._parse_rate_limit_wait("Rate limited")
        assert wait == 300

    def test_zero_minutes(self):
        wait = self.runner._parse_rate_limit_wait("Wait 0 minutes")
        assert wait == 0 * 60 + 60


class TestClaudeRunnerInvoke:
    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.run_subprocess")
    def test_invoke_success_plain_text(self, mock_run, mock_interrupt):
        mock_run.return_value = subprocess.CompletedProcess([], 0, "done", "")
        runner = ClaudeRunner(RunConfig())
        result = runner.invoke("do stuff")
        assert result.output == "done"
        assert result.exit_code == 0

    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.run_subprocess")
    def test_invoke_non_json_output(self, mock_run, mock_interrupt):
        mock_run.return_value = subprocess.CompletedProcess(
            [], 0, "plain text output", ""
        )
        runner = ClaudeRunner(RunConfig())
        result = runner.invoke("test")
        assert result.output == "plain text output"
        assert result.exit_code == 0

    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.run_subprocess")
    def test_invoke_non_json_with_stderr(self, mock_run, mock_interrupt):
        mock_run.return_value = subprocess.CompletedProcess([], 1, "", "error happened")
        runner = ClaudeRunner(RunConfig())
        result = runner.invoke("test")
        assert result.output == "error happened"
        assert result.exit_code == 1

    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.run_subprocess")
    def test_invoke_continue_session(self, mock_run, mock_interrupt):
        mock_run.return_value = subprocess.CompletedProcess([], 0, "ok", "")
        runner = ClaudeRunner(RunConfig())
        runner.invoke("continue", continue_session=True)
        args = mock_run.call_args[0][0]
        assert "--continue" in args

    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.run_subprocess")
    @patch("claude_tools.iterate.time.sleep")
    def test_invoke_rate_limit_then_success(self, mock_sleep, mock_run, mock_interrupt):
        rate_limit_result = subprocess.CompletedProcess(
            [], 1, "", "you've hit your limit. Wait 1 minute"
        )
        success_result = subprocess.CompletedProcess([], 0, "ok", "")
        mock_run.side_effect = [rate_limit_result, success_result]

        runner = ClaudeRunner(RunConfig())
        result = runner.invoke("test")
        assert result.output == "ok"
        assert mock_run.call_count == 2


class TestParseArgs:
    def test_defaults(self):
        with patch("sys.argv", ["iterate.py"]):
            args = parse_args()
        assert args.model is None
        assert args.prompts is None
        assert args.max_iterations == 10

    def test_model_flag(self):
        with patch("sys.argv", ["iterate.py", "--model", "opus"]):
            args = parse_args()
        assert args.model == "opus"

    def test_model_short_flag(self):
        with patch("sys.argv", ["iterate.py", "-m", "sonnet"]):
            args = parse_args()
        assert args.model == "sonnet"

    def test_custom_prompts(self):
        with patch(
            "sys.argv",
            ["iterate.py", "-p", "Fix bugs", "-p", "Add tests"],
        ):
            args = parse_args()
        assert args.prompts == ["Fix bugs", "Add tests"]

    def test_max_iterations(self):
        with patch("sys.argv", ["iterate.py", "--max-iterations", "5"]):
            args = parse_args()
        assert args.max_iterations == 5


class TestTaskOrchestrator:
    def _make_orchestrator(self, tasks, tmp_path, max_iterations=2):
        config = RunConfig(
            max_iterations=max_iterations,
            cooldown_seconds=0,
            log_file=tmp_path / "log.md",
        )
        return TaskOrchestrator(tasks, config)

    @patch(
        "claude_tools.iterate.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, "", ""),
    )
    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.squash_task_commits")
    @patch("claude_tools.iterate.commit_changes", return_value=False)
    @patch("claude_tools.iterate.git_head_sha", return_value="sha1")
    @patch.object(ClaudeRunner, "invoke")
    def test_task_converges_on_no_changes(
        self,
        mock_invoke,
        mock_sha,
        mock_commit,
        mock_squash,
        mock_interrupt,
        mock_subproc_run,
        tmp_path,
    ):
        mock_invoke.return_value = ClaudeResult(output="NO_CHANGES", exit_code=0)
        orch = self._make_orchestrator([Task("T1", "prompt1")], tmp_path)
        results = orch.run_all()

        assert len(results) == 1
        assert results[0].status == TaskStatus.CONVERGED
        assert results[0].iterations == 1

    @patch(
        "claude_tools.iterate.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, "", ""),
    )
    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.squash_task_commits")
    @patch("claude_tools.iterate.commit_changes", return_value=True)
    @patch("claude_tools.iterate.git_head_sha", return_value="sha1")
    @patch.object(ClaudeRunner, "invoke")
    def test_task_continues_on_nonzero_exit(
        self,
        mock_invoke,
        mock_sha,
        mock_commit,
        mock_squash,
        mock_interrupt,
        mock_subproc_run,
        tmp_path,
    ):
        mock_invoke.return_value = ClaudeResult(output="error", exit_code=1)
        orch = self._make_orchestrator([Task("T1", "prompt1")], tmp_path)
        results = orch.run_all()

        assert len(results) == 1
        assert results[0].status == TaskStatus.MAX_ITERATIONS
        assert results[0].iterations == 2

    @patch(
        "claude_tools.iterate.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, "", ""),
    )
    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.squash_task_commits")
    @patch("claude_tools.iterate.commit_changes", return_value=True)
    @patch("claude_tools.iterate.git_head_sha", return_value="sha1")
    @patch.object(ClaudeRunner, "invoke")
    def test_task_hits_max_iterations(
        self,
        mock_invoke,
        mock_sha,
        mock_commit,
        mock_squash,
        mock_interrupt,
        mock_subproc_run,
        tmp_path,
    ):
        mock_invoke.return_value = ClaudeResult(output="changes made", exit_code=0)
        orch = self._make_orchestrator(
            [Task("T1", "prompt1")], tmp_path, max_iterations=3
        )
        results = orch.run_all()

        assert len(results) == 1
        assert results[0].status == TaskStatus.MAX_ITERATIONS
        assert results[0].iterations == 3

    @patch(
        "claude_tools.iterate.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, "", ""),
    )
    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.squash_task_commits")
    @patch("claude_tools.iterate.commit_changes", return_value=False)
    @patch("claude_tools.iterate.git_head_sha", return_value="sha1")
    @patch.object(ClaudeRunner, "invoke")
    def test_multiple_tasks(
        self,
        mock_invoke,
        mock_sha,
        mock_commit,
        mock_squash,
        mock_interrupt,
        mock_subproc_run,
        tmp_path,
    ):
        mock_invoke.return_value = ClaudeResult(output="NO_CHANGES", exit_code=0)
        tasks = [Task("T1", "p1"), Task("T2", "p2")]
        orch = self._make_orchestrator(tasks, tmp_path)
        results = orch.run_all()

        assert len(results) == 2
        assert all(r.status == TaskStatus.CONVERGED for r in results)

    @patch(
        "claude_tools.iterate.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, "", ""),
    )
    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.squash_task_commits")
    @patch("claude_tools.iterate.commit_changes", return_value=True)
    @patch("claude_tools.iterate.git_head_sha", return_value="sha1")
    @patch.object(ClaudeRunner, "invoke")
    def test_first_iteration_uses_task_prompt(
        self,
        mock_invoke,
        mock_sha,
        mock_commit,
        mock_squash,
        mock_interrupt,
        mock_subproc_run,
        tmp_path,
    ):
        mock_invoke.return_value = ClaudeResult(output="NO_CHANGES", exit_code=0)
        orch = self._make_orchestrator([Task("T1", "Fix bugs")], tmp_path)
        orch.run_all()

        first_task_call = mock_invoke.call_args_list[0]
        assert "Fix bugs" in first_task_call[0][0]
        assert first_task_call[1].get("continue_session", False) is False

    @patch(
        "claude_tools.iterate.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, "", ""),
    )
    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.squash_task_commits")
    @patch("claude_tools.iterate.commit_changes", return_value=True)
    @patch("claude_tools.iterate.git_head_sha", return_value="sha1")
    @patch.object(ClaudeRunner, "invoke")
    def test_subsequent_iterations_use_continue(
        self,
        mock_invoke,
        mock_sha,
        mock_commit,
        mock_squash,
        mock_interrupt,
        mock_subproc_run,
        tmp_path,
    ):
        call_count = [0]

        def side_effect(prompt, continue_session=False):
            call_count[0] += 1
            if call_count[0] >= 2:
                return ClaudeResult(output="NO_CHANGES", exit_code=0)
            return ClaudeResult(output="changes", exit_code=0)

        mock_invoke.side_effect = side_effect
        orch = self._make_orchestrator([Task("T1", "Fix bugs")], tmp_path)
        orch.run_all()

        second_call = mock_invoke.call_args_list[1]
        assert "Keep going" in second_call[0][0]
        assert second_call[1].get("continue_session") is True

    @patch(
        "claude_tools.iterate.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, "", ""),
    )
    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.squash_task_commits")
    @patch("claude_tools.iterate.commit_changes", return_value=True)
    @patch("claude_tools.iterate.git_head_sha", return_value="sha1")
    @patch.object(ClaudeRunner, "invoke")
    def test_prompt_too_long_retries_with_fresh_session(
        self,
        mock_invoke,
        mock_sha,
        mock_commit,
        mock_squash,
        mock_interrupt,
        mock_subproc_run,
        tmp_path,
    ):
        call_count = [0]

        def side_effect(prompt, continue_session=False):
            call_count[0] += 1
            # Call 1: iteration 1 (fresh) -> success
            # Call 2: iteration 2 (continue) -> prompt too long
            # Call 3: iteration 2 retry (fresh) -> NO_CHANGES
            if call_count[0] == 2:
                return ClaudeResult(output="Prompt is too long", exit_code=1)
            if call_count[0] == 3:
                return ClaudeResult(output="NO_CHANGES", exit_code=0)
            return ClaudeResult(output="changes", exit_code=0)

        mock_invoke.side_effect = side_effect
        orch = self._make_orchestrator(
            [Task("T1", "Fix bugs")], tmp_path, max_iterations=3
        )
        results = orch.run_all()

        assert results[0].status == TaskStatus.CONVERGED
        assert results[0].iterations == 2
        retry_call = mock_invoke.call_args_list[2]
        assert "Fix bugs" in retry_call[0][0]
        assert retry_call[1].get("continue_session") is False

    @patch(
        "claude_tools.iterate.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, "", ""),
    )
    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.squash_task_commits")
    @patch("claude_tools.iterate.commit_changes", return_value=False)
    @patch("claude_tools.iterate.git_head_sha", return_value="sha1")
    @patch.object(ClaudeRunner, "invoke")
    def test_squash_called_with_correct_args(
        self,
        mock_invoke,
        mock_sha,
        mock_commit,
        mock_squash,
        mock_interrupt,
        mock_subproc_run,
        tmp_path,
    ):
        mock_invoke.return_value = ClaudeResult(output="NO_CHANGES", exit_code=0)
        orch = self._make_orchestrator([Task("Bug fix", "fix bugs")], tmp_path)
        orch.run_all()

        mock_squash.assert_called_once_with("sha1", "Bug fix - automated iteration")


class TestTaskOrchestratorOutput:
    @patch(
        "claude_tools.iterate.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, "", ""),
    )
    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.squash_task_commits")
    @patch("claude_tools.iterate.commit_changes", return_value=False)
    @patch("claude_tools.iterate.git_head_sha", return_value="sha1")
    @patch.object(ClaudeRunner, "invoke")
    def test_log_file_created(
        self,
        mock_invoke,
        mock_sha,
        mock_commit,
        mock_squash,
        mock_interrupt,
        mock_subproc_run,
        tmp_path,
    ):
        mock_invoke.return_value = ClaudeResult(output="NO_CHANGES", exit_code=0)
        config = RunConfig(
            max_iterations=1,
            log_file=tmp_path / "logs" / "test.md",
        )
        orch = TaskOrchestrator([Task("T1", "p1")], config)
        orch.run_all()

        assert (tmp_path / "logs" / "test.md").exists()
        content = (tmp_path / "logs" / "test.md").read_text()
        assert "T1" in content


class TestPrintSummary:
    @patch(
        "claude_tools.iterate.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, "", ""),
    )
    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.squash_task_commits")
    @patch("claude_tools.iterate.commit_changes", return_value=False)
    @patch("claude_tools.iterate.git_head_sha", return_value="sha1")
    @patch.object(ClaudeRunner, "invoke")
    def test_summary_includes_all_statuses(
        self,
        mock_invoke,
        mock_sha,
        mock_commit,
        mock_squash,
        mock_interrupt,
        mock_subproc_run,
        tmp_path,
        capsys,
    ):
        mock_invoke.return_value = ClaudeResult(output="NO_CHANGES", exit_code=0)
        config = RunConfig(max_iterations=1, log_file=tmp_path / "log.md")
        orch = TaskOrchestrator([Task("T1", "p1")], config)
        orch.run_all()

        captured = capsys.readouterr()
        assert "T1" in captured.out
        assert "converged" in captured.out
        assert "Summary" in captured.out


class TestMainFunction:
    @patch("claude_tools.iterate.TaskOrchestrator")
    @patch("claude_tools.iterate.parse_args")
    def test_main_with_custom_prompts(self, mock_args, mock_orch_cls):
        mock_args.return_value = MagicMock(
            model="opus", prompts=["p1", "p2"], max_iterations=5, cooldown=5
        )
        mock_orch = MagicMock()
        mock_orch.run_all.return_value = [
            TaskResult("Task 1", TaskStatus.CONVERGED, 1, 1.0)
        ]
        mock_orch_cls.return_value = mock_orch

        main()

        tasks_arg = mock_orch_cls.call_args[0][0]
        assert len(tasks_arg) == 2
        assert tasks_arg[0].name == "Task 1"
        assert tasks_arg[0].prompt == "p1"
        assert tasks_arg[1].name == "Task 2"
        assert tasks_arg[1].prompt == "p2"

    @patch("claude_tools.iterate.TaskOrchestrator")
    @patch("claude_tools.iterate.parse_args")
    def test_main_with_default_tasks(self, mock_args, mock_orch_cls):
        mock_args.return_value = MagicMock(
            model=None, prompts=None, tasks=None, max_iterations=20, cooldown=5
        )
        mock_orch = MagicMock()
        mock_orch.run_all.return_value = [TaskResult("T", TaskStatus.CONVERGED, 1, 1.0)]
        mock_orch_cls.return_value = mock_orch

        main()

        tasks_arg = mock_orch_cls.call_args[0][0]
        assert tasks_arg is DEFAULT_TASKS

    @patch("claude_tools.iterate.TaskOrchestrator")
    @patch("claude_tools.iterate.parse_args")
    def test_main_no_exit_on_success(self, mock_args, mock_orch_cls):
        mock_args.return_value = MagicMock(
            model=None, prompts=None, tasks=None, max_iterations=20, cooldown=5
        )
        mock_orch = MagicMock()
        mock_orch.run_all.return_value = [
            TaskResult("T", TaskStatus.CONVERGED, 1, 1.0),
            TaskResult("T2", TaskStatus.MAX_ITERATIONS, 20, 10.0),
        ]
        mock_orch_cls.return_value = mock_orch

        main()  # should not raise


class TestDefaultTasks:
    def test_default_tasks_count(self):
        assert len(DEFAULT_TASKS) == 6

    def test_default_tasks_names(self):
        names = [t.name for t in DEFAULT_TASKS]
        assert names == [
            "Bug fixes",
            "Test coverage",
            "Conciseness",
            "Optimization",
            "Config",
            "Markdown",
        ]

    def test_all_tasks_have_nonempty_prompts(self):
        for t in DEFAULT_TASKS:
            assert len(t.prompt) > 0

    def test_task_keys_match_default_tasks_length(self):
        assert len(TASK_KEYS) == len(DEFAULT_TASKS)

    def test_task_map_keys_match_task_keys(self):
        assert list(TASK_MAP.keys()) == TASK_KEYS

    def test_task_map_values_match_default_tasks(self):
        assert list(TASK_MAP.values()) == DEFAULT_TASKS


class TestCheckInterrupt:
    @patch("claude_tools.iterate._interrupted", False)
    def test_no_op_when_not_interrupted(self):
        check_interrupt()  # should not raise

    @patch("claude_tools.iterate._interrupted", True)
    @patch("claude_tools.iterate._current_proc", None)
    def test_exits_130_when_interrupted_no_proc(self):
        with pytest.raises(SystemExit) as exc_info:
            check_interrupt()
        assert exc_info.value.code == 130

    @patch("claude_tools.iterate._interrupted", True)
    def test_kills_running_proc(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        with patch("claude_tools.iterate._current_proc", mock_proc):
            with pytest.raises(SystemExit) as exc_info:
                check_interrupt()
            assert exc_info.value.code == 130
        mock_proc.kill.assert_called_once()
        mock_proc.wait.assert_called_once()

    @patch("claude_tools.iterate._interrupted", True)
    def test_skips_kill_if_proc_already_finished(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # already finished
        with patch("claude_tools.iterate._current_proc", mock_proc):
            with pytest.raises(SystemExit):
                check_interrupt()
        mock_proc.kill.assert_not_called()


class TestRunSubprocess:
    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.time.sleep")
    @patch("claude_tools.iterate.subprocess.Popen")
    @patch("claude_tools.iterate.tempfile.TemporaryFile")
    def test_captures_stdout_and_stderr(
        self, mock_tmpfile, mock_popen, mock_sleep, mock_interrupt
    ):
        f_err = StringIO()
        mock_tmpfile.return_value = f_err
        f_err.write("hello stderr")

        mock_stdout = MagicMock()
        mock_stdout.__iter__ = MagicMock(return_value=iter(["hello stdout\n"]))
        mock_proc = MagicMock()
        mock_proc.stdout = mock_stdout
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        result = run_subprocess(["echo", "hi"])
        assert result.stdout == "hello stdout\n"
        assert result.stderr == "hello stderr"
        assert result.returncode == 0

    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.time.sleep")
    @patch("claude_tools.iterate.subprocess.Popen")
    @patch("claude_tools.iterate.tempfile.TemporaryFile")
    def test_polls_until_complete(
        self, mock_tmpfile, mock_popen, mock_sleep, mock_interrupt
    ):
        mock_tmpfile.return_value = StringIO()
        mock_stdout = MagicMock()
        mock_stdout.__iter__ = MagicMock(return_value=iter([]))
        mock_proc = MagicMock()
        mock_proc.stdout = mock_stdout
        mock_proc.poll.side_effect = [None, None, 0]
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        result = run_subprocess(["cmd"])
        assert result.returncode == 0
        assert mock_sleep.call_count == 2

    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.time.sleep")
    @patch("claude_tools.iterate.subprocess.Popen")
    @patch("claude_tools.iterate.tempfile.TemporaryFile")
    def test_nonzero_exit_code(
        self, mock_tmpfile, mock_popen, mock_sleep, mock_interrupt
    ):
        f_err = StringIO()
        mock_tmpfile.return_value = f_err
        f_err.write("fail")

        mock_stdout = MagicMock()
        mock_stdout.__iter__ = MagicMock(return_value=iter([]))
        mock_proc = MagicMock()
        mock_proc.stdout = mock_stdout
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc

        result = run_subprocess(["bad"])
        assert result.returncode == 1
        assert result.stderr == "fail"


class TestParseRateLimitWaitValueError:
    def test_invalid_time_format_falls_through(self):
        runner = ClaudeRunner(RunConfig(default_wait_seconds=900))
        # "99:99 AM" matches the regex r"(\d{1,2}:\d{2}\s*(?:AM|PM))"
        # but strptime will raise ValueError for invalid time
        wait = runner._parse_rate_limit_wait("Resets at 99:99 AM")
        # Falls through to minutes pattern or default
        assert wait == 900


class TestHasRecentEdit:
    def test_returns_true_for_recently_modified_file(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        past = time.time() - 10
        with patch("claude_tools.iterate.os.walk") as mock_walk:
            mock_walk.return_value = [(str(tmp_path), [], ["file.txt"])]
            with patch(
                "claude_tools.iterate.os.path.getmtime",
                return_value=time.time(),
            ):
                assert _has_recent_edit(past) is True

    def test_returns_false_when_no_files_modified(self):
        past = time.time() + 100
        with patch("claude_tools.iterate.os.walk") as mock_walk:
            mock_walk.return_value = [(".", [], ["old.txt"])]
            with patch(
                "claude_tools.iterate.os.path.getmtime",
                return_value=time.time() - 200,
            ):
                assert _has_recent_edit(past) is False

    def test_skips_excluded_dirs(self):
        with patch("claude_tools.iterate.os.walk") as mock_walk:
            dirs = [".git", "__pycache__", "src"]
            mock_walk.return_value = [(".", dirs, [])]
            _has_recent_edit(0)
            assert dirs == ["src"]

    def test_handles_oserror_gracefully(self):
        with patch("claude_tools.iterate.os.walk") as mock_walk:
            mock_walk.return_value = [(".", [], ["gone.txt"])]
            with patch(
                "claude_tools.iterate.os.path.getmtime",
                side_effect=OSError("file gone"),
            ):
                assert _has_recent_edit(0) is False

    def test_returns_false_for_empty_dir(self):
        with patch("claude_tools.iterate.os.walk") as mock_walk:
            mock_walk.return_value = [(".", [], [])]
            assert _has_recent_edit(0) is False


class TestKeypressMonitor:
    @patch("claude_tools.iterate.time.sleep")
    @patch("claude_tools.iterate.msvcrt")
    def test_sets_interrupted_on_q(self, mock_msvcrt, mock_sleep):
        import claude_tools.iterate as mod

        original = mod._interrupted
        try:
            mod._interrupted = False
            mock_msvcrt.kbhit.return_value = True
            mock_msvcrt.getwch.return_value = "q"
            _keypress_monitor()
            assert mod._interrupted is True
        finally:
            mod._interrupted = original

    @patch("claude_tools.iterate.time.sleep")
    @patch("claude_tools.iterate.msvcrt")
    def test_ignores_non_q_key(self, mock_msvcrt, mock_sleep):
        import claude_tools.iterate as mod

        original = mod._interrupted
        try:
            mod._interrupted = False
            call_count = [0]

            def kbhit_side():
                call_count[0] += 1
                return call_count[0] <= 2

            def getwch_side():
                return "x"

            def check_loop(*a):
                # After a couple iterations, set interrupted to break the loop
                if call_count[0] >= 2:
                    mod._interrupted = True

            mock_msvcrt.kbhit.side_effect = kbhit_side
            mock_msvcrt.getwch.side_effect = getwch_side
            mock_sleep.side_effect = check_loop
            _keypress_monitor()
        finally:
            mod._interrupted = original


class TestRunSubprocessStall:
    def _make_mock_proc(self, poll_returns, returncode, stdout_lines=None):
        mock_stdout = MagicMock()
        mock_stdout.__iter__ = MagicMock(return_value=iter(stdout_lines or []))
        mock_proc = MagicMock()
        mock_proc.stdout = mock_stdout
        if isinstance(poll_returns, list):
            mock_proc.poll.side_effect = poll_returns
        else:
            mock_proc.poll.return_value = poll_returns
        mock_proc.returncode = returncode
        return mock_proc

    @patch("claude_tools.iterate._has_recent_edit", return_value=False)
    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.subprocess.Popen")
    @patch("claude_tools.iterate.tempfile.TemporaryFile")
    def test_kills_on_stall_timeout(
        self, mock_tmpfile, mock_popen, mock_interrupt, mock_recent
    ):
        mock_tmpfile.return_value = StringIO()
        mock_popen.return_value = self._make_mock_proc(None, -9)

        mono_values = iter([0, 0, 0, 1000, 1000, 1000, 1000, 1000, 1000, 1000])

        with patch("claude_tools.iterate.time.monotonic", side_effect=mono_values):
            with patch("claude_tools.iterate.time.sleep"):
                with patch("claude_tools.iterate.time.time", return_value=0):
                    result = run_subprocess(["stalling"])

        mock_popen.return_value.kill.assert_called_once()
        mock_popen.return_value.wait.assert_called_once()
        assert result.returncode == -9

    @patch("claude_tools.iterate._has_recent_edit", return_value=False)
    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.subprocess.Popen")
    @patch("claude_tools.iterate.tempfile.TemporaryFile")
    def test_stdout_growth_resets_activity(
        self, mock_tmpfile, mock_popen, mock_interrupt, mock_recent
    ):
        f_err = MagicMock()
        mock_tmpfile.return_value = f_err
        f_err.tell.return_value = 0
        f_err.seek = MagicMock()
        f_err.read.return_value = ""
        f_err.close = MagicMock()

        mock_popen.return_value = self._make_mock_proc(
            [None, 0], 0, stdout_lines=["output\n"]
        )

        mono = iter([0, 0, 1, 1, 1])
        with patch("claude_tools.iterate.time.monotonic", side_effect=mono):
            with patch("claude_tools.iterate.time.sleep"):
                with patch("claude_tools.iterate.time.time", return_value=0):
                    result = run_subprocess(["cmd"])

        assert result.returncode == 0
        assert result.stdout == "output\n"

    @patch("claude_tools.iterate._has_recent_edit", return_value=True)
    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.subprocess.Popen")
    @patch("claude_tools.iterate.tempfile.TemporaryFile")
    def test_recent_edit_resets_activity(
        self, mock_tmpfile, mock_popen, mock_interrupt, mock_recent
    ):
        f_err = MagicMock()
        mock_tmpfile.return_value = f_err
        f_err.tell.return_value = 0
        f_err.seek = MagicMock()
        f_err.read.return_value = ""
        f_err.close = MagicMock()

        mock_popen.return_value = self._make_mock_proc([None, 0], 0)

        mono = iter([0, 10, 10])
        with patch("claude_tools.iterate.time.monotonic", side_effect=mono):
            with patch("claude_tools.iterate.time.sleep"):
                with patch("claude_tools.iterate.time.time", return_value=100):
                    result = run_subprocess(["cmd"])

        mock_recent.assert_called()
        assert result.returncode == 0


class TestTaskOrchestratorCooldown:
    @patch(
        "claude_tools.iterate.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, "", ""),
    )
    @patch("claude_tools.iterate.time.sleep")
    @patch("claude_tools.iterate.check_interrupt")
    @patch("claude_tools.iterate.squash_task_commits")
    @patch("claude_tools.iterate.commit_changes", return_value=True)
    @patch("claude_tools.iterate.git_head_sha", return_value="sha1")
    @patch.object(ClaudeRunner, "invoke")
    def test_cooldown_waits_between_iterations(
        self,
        mock_invoke,
        mock_sha,
        mock_commit,
        mock_squash,
        mock_interrupt,
        mock_sleep,
        mock_subproc_run,
        tmp_path,
    ):
        call_count = [0]

        def side_effect(prompt, continue_session=False):
            call_count[0] += 1
            if call_count[0] >= 2:
                return ClaudeResult(output="NO_CHANGES", exit_code=0)
            return ClaudeResult(output="changes", exit_code=0)

        mock_invoke.side_effect = side_effect
        config = RunConfig(
            max_iterations=3,
            cooldown_seconds=2,
            log_file=tmp_path / "log.md",
        )
        orch = TaskOrchestrator([Task("T1", "prompt")], config)
        orch.run_all()

        # Cooldown sleeps: iteration 2 each sleep 2 times (2 seconds)
        sleep_calls = [c for c in mock_sleep.call_args_list if c[0] == (1,)]
        assert len(sleep_calls) >= 2


class TestMainWithTaskKeys:
    @patch("claude_tools.iterate.TaskOrchestrator")
    @patch("claude_tools.iterate.parse_args")
    def test_main_with_task_keys(self, mock_args, mock_orch_cls):
        mock_args.return_value = MagicMock(
            model=None,
            prompts=None,
            tasks=["bugs", "tests"],
            max_iterations=10,
            cooldown=60,
        )
        mock_orch = MagicMock()
        mock_orch.run_all.return_value = [TaskResult("T", TaskStatus.CONVERGED, 1, 1.0)]
        mock_orch_cls.return_value = mock_orch

        main()

        tasks_arg = mock_orch_cls.call_args[0][0]
        assert len(tasks_arg) == 2
        assert tasks_arg[0] is TASK_MAP["bugs"]
        assert tasks_arg[1] is TASK_MAP["tests"]
