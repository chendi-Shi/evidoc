"""Training lifecycle controls are tested with fake processes, never CUDA."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

import psutil

from scripts import manage_training as control


class TrainingControlTests(TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / "runs/experiment"
        self.output.mkdir(parents=True)
        pilot = self.root / "runs/pilot"
        pilot.mkdir()
        self.pilot = pilot / "pilot_report.json"
        self.pilot.write_text("{}", encoding="utf-8")
        (pilot / "run_manifest.json").write_text(json.dumps({"arguments": {
            "model": "models/model", "data": "data/train", "max_length": 512,
            "accumulation": 16, "epochs": 3, "hours": 72,
        }}), encoding="utf-8")
        self.python = self.root / ".tools/python/python.exe"
        self.summary = {
            "state": "interrupted", "budget_hours": 84, "elapsed_seconds": 12345,
            "pilot": str(self.pilot), "model": "models/model", "data": "data/train",
            "configurations": [{"name": "r8-hybrid-s42", "seed": 42}, {"name": "r8-hybrid-s43", "seed": 43}],
            "runs": [{"name": "r8-hybrid-s42", "state": "interrupted", "command": [
                str(self.python), "-m", "training.train", "--model", "models/model", "--data", "data/train",
                "--max-length", "384", "--accumulation", "8", "--epochs", "2", "--hours", "80",
            ]}], "evaluate_after": True,
        }
        self.save_summary()

    def save_summary(self):
        (self.output / "experiment.json").write_text(json.dumps(self.summary), encoding="utf-8")

    def process(self, module="scripts.run_experiments", output=None, root=None, pid=123):
        root = root or self.root
        target = output or self.output
        command = [str(root / ".tools/python/python.exe"), "-u", "-m", module]
        command += ([str(target)] if module == "scripts.evaluate_experiments" else ["--output", str(target)])
        process = Mock(pid=pid)
        process.is_running.return_value = True
        process.status.return_value = psutil.STATUS_RUNNING
        process.cmdline.return_value = command
        process.exe.return_value = str(root / ".tools/python/python.exe")
        process.cwd.return_value = str(root)
        process.create_time.return_value = 123.0
        return process

    def test_legacy_reconstruction_keeps_budget_and_real_run_settings(self):
        command = control.reconstruct_command(self.summary, self.output, self.root)
        self.assertEqual(command[:4], [str(self.python), "-u", "-m", "scripts.run_experiments"])
        self.assertEqual(control.option(command, "--hours"), "84")
        self.assertEqual(control.option(command, "--max-length"), "384")
        self.assertEqual(control.option(command, "--accumulation"), "8")
        self.assertEqual(control.option(command, "--epochs"), "2")
        self.assertEqual(control.option(command, "--seeds"), "42,43")
        self.assertIn("--evaluate-after", command)
        self.assertIn("--resume", command)

    def test_legacy_unknown_evaluation_policy_fails_instead_of_guessing(self):
        self.summary.pop("evaluate_after")
        with self.assertRaisesRegex(ValueError, "evaluate_after"):
            control.reconstruct_command(self.summary, self.output, self.root)

    def test_saved_supervisor_command_restores_legacy_evaluation_policy(self):
        command = control.reconstruct_command(self.summary, self.output, self.root)
        command.remove("--evaluate-after")
        self.summary.pop("evaluate_after")
        active = {"output": str(self.output), "command": command}
        reconstructed = control.reconstruct_command(self.summary, self.output, self.root, active=active)
        self.assertNotIn("--evaluate-after", reconstructed)
        self.assertEqual(control.option(reconstructed, "--hours"), "84")

    def test_conflicting_recorded_budget_is_rejected(self):
        self.summary["arguments"] = {"hours": 96}
        with self.assertRaisesRegex(ValueError, "original time budget"):
            control.reconstruct_command(self.summary, self.output, self.root)

    def test_process_matching_covers_supervisor_orphan_worker_and_evaluators(self):
        processes = [self.process(), self.process("training.train", self.output / "run"),
                     self.process("scripts.evaluate_experiments"), self.process("training.evaluate", self.output / "run/evaluation")]
        with patch.object(control.psutil, "process_iter", return_value=processes):
            result = control.find_live_processes(self.output, self.root)
        self.assertEqual({item["module"] for item in result}, control.MODULES)

    def test_other_project_and_other_experiment_and_reused_pid_do_not_match(self):
        unrelated = [self.process(root=self.root / "other", pid=123),
                     self.process(output=self.root / "runs/other", pid=123),
                     self.process(module="unrelated.module", pid=123)]
        unrelated[2].cmdline.return_value += ["scripts.run_experiments"]
        with patch.object(control.psutil, "process_iter", return_value=unrelated):
            self.assertEqual(control.find_live_processes(self.output, self.root), [])

    def test_direct_script_and_relative_output_resolve_against_process_cwd(self):
        process = self.process()
        process.cmdline.return_value = [str(self.python), str(self.root / "scripts/run_experiments.py"), "--output=runs/experiment"]
        with patch.object(control.psutil, "process_iter", return_value=[process]):
            self.assertEqual(len(control.find_live_processes(self.output, self.root)), 1)

    def test_disappearing_process_does_not_break_scan(self):
        dead = self.process()
        dead.exe.side_effect = psutil.NoSuchProcess(123)
        with patch.object(control.psutil, "process_iter", return_value=[dead, self.process()]):
            self.assertEqual(len(control.find_live_processes(self.output, self.root)), 1)

    def test_resume_with_live_worker_is_friendly_and_never_launches(self):
        live = [{"pid": 4, "module": "training.train"}]
        with patch.object(control, "find_live_processes", return_value=live), patch.object(control, "launch") as launch:
            result = control.manage("resume", self.output, root=self.root)
        self.assertEqual(result["state"], "already_running")
        launch.assert_not_called()
        self.assertFalse((self.root / "runtime").exists())

    def test_dry_run_writes_no_logs_pid_or_stop_file(self):
        before = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with patch.object(control, "find_live_processes", return_value=[]), patch.object(control, "launch") as launch:
            self.assertEqual(control.manage("resume", self.output, True, self.root)["state"], "would_resume")
            self.assertEqual(control.manage("stop", self.output, True, self.root)["state"], "would_request_stop")
        after = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        launch.assert_not_called()

    def test_stop_requests_checkpoint_without_killing_process(self):
        with patch.object(control, "find_live_processes", return_value=[]), patch.object(control.subprocess, "Popen") as popen:
            result = control.manage("stop", self.output, root=self.root)
        self.assertEqual(result["state"], "stop_requested")
        self.assertEqual((self.output / "STOP_REQUESTED").read_text(), "user_stop")
        popen.assert_not_called()

    def test_resume_rechecks_process_before_launch(self):
        self.python.parent.mkdir(parents=True)
        self.python.write_bytes(b"mock")
        with patch.object(control, "find_live_processes", side_effect=[[], [{"pid": 7}]]), patch.object(control, "launch") as launch:
            result = control.manage("resume", self.output, root=self.root)
        self.assertEqual(result["state"], "already_running")
        launch.assert_not_called()

    def test_stop_reaches_orphan_worker_and_excludes_unrelated_outputs(self):
        worker = self.output / "r8-hybrid-s42"
        outside = self.root / "runs/unrelated"
        live = [{"module": "training.train", "cwd": str(self.root), "command": [
                    str(self.python), "-m", "training.train", "--output", str(path)]}
                for path in [worker, outside, self.output / "../unrelated", self.output]]
        with patch.object(control, "find_live_processes", return_value=live):
            result = control.manage("stop", self.output, root=self.root)
        self.assertEqual(set(result["stop_files"]), {str(self.output / "STOP_REQUESTED"), str(worker / "STOP_REQUESTED")})
        self.assertEqual((worker / "STOP_REQUESTED").read_text(), "user_stop")
        self.assertFalse(outside.exists())

    def test_orphan_stop_dry_run_lists_targets_without_writing(self):
        worker = self.output / "r8-hybrid-s42"
        live = [{"module": "training.train", "cwd": str(self.root), "command": [
            str(self.python), "-m", "training.train", "--output", str(worker)]}]
        with patch.object(control, "find_live_processes", return_value=live):
            result = control.manage("stop", self.output, True, self.root)
        self.assertIn(str(worker / "STOP_REQUESTED"), result["stop_files"])
        self.assertFalse(worker.exists())
        self.assertFalse((self.output / "STOP_REQUESTED").exists())

    def test_launch_uses_fresh_logs_and_records_reproducible_command(self):
        command = control.reconstruct_command(self.summary, self.output, self.root)
        with patch.object(control.subprocess, "Popen", return_value=Mock(pid=123)) as popen:
            first = control.launch(command, self.output, self.root)
            second = control.launch(command, self.output, self.root)
        self.assertNotEqual(first["stdout"], second["stdout"])
        self.assertTrue(Path(first["stdout"]).exists())
        self.assertEqual(control.read_json(self.root / "reports/active_experiment.json")["command"], command)
        self.assertEqual((self.root / "runtime/experiment.pid").read_text(), "123")
        self.assertEqual(popen.call_args.kwargs["stdin"], control.subprocess.DEVNULL)
        self.assertEqual(popen.call_args.kwargs["cwd"], self.root)

    def test_bom_metadata_is_accepted(self):
        metadata = self.root / "active.json"
        metadata.write_text('{"evaluate_after": true}', encoding="utf-8-sig")
        self.assertTrue(control.read_json(metadata)["evaluate_after"])

    def test_completed_runs_are_omitted_and_invalid_checkpoint_is_reported(self):
        self.summary["runs"][0]["state"] = "completed"
        (self.output / "r8-hybrid-s43").mkdir()
        with patch("training.checkpoints.latest_checkpoint", side_effect=ValueError("truncated checkpoint")):
            entries = control.checkpoint_inventory(self.summary, self.output)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["checkpoint_error"], "truncated checkpoint")
