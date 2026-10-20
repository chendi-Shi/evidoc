"""Checkpoint recovery regression tests require no torch, network or GPU."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
import warnings
import zipfile

from training.checkpoints import commit_checkpoint, latest_checkpoint, validate_checkpoint


def checkpoint(root, step):
    path = root / f"checkpoint-{step}"
    path.mkdir()
    for name, value in (("adapter_config.json", {"r": 8}),
                        ("trainer_state.json", {"global_step": step}),
                        ("monitor_state.json", {"step": step, "training_seconds": step * 20})):
        (path / name).write_text(json.dumps(value), encoding="utf-8")
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (path / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        with zipfile.ZipFile(path / name, "w") as archive:
            archive.writestr("archive/data.pkl", b"fixture-only-no-deserialization")
            archive.writestr("archive/version", b"3\n")
    return path


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_old_checkpoint_is_supported(self):
        path = checkpoint(self.root, 50)
        result = validate_checkpoint(path)
        self.assertEqual(result["format"], "legacy")
        self.assertEqual(result["global_step"], 50)
        self.assertEqual(latest_checkpoint(self.root), path)

    def test_incomplete_latest_falls_back_without_deleting_it(self):
        old = checkpoint(self.root, 50)
        new = checkpoint(self.root, 100)
        (new / "scheduler.pt").unlink()
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(latest_checkpoint(self.root), old)
        self.assertTrue(new.exists())

    def test_new_format_requires_commit_marker(self):
        path = checkpoint(self.root, 50)
        (self.root / "run_manifest.json").write_text('{"checkpoint_format": 1}')
        with self.assertRaisesRegex(ValueError, "commit manifest is missing"):
            validate_checkpoint(path)
        commit_checkpoint(path)
        self.assertEqual(validate_checkpoint(path)["format"], "committed")

    def test_commit_hash_detects_equal_length_weight_changes(self):
        path = checkpoint(self.root, 50)
        commit_checkpoint(path)
        weights = path / "adapter_model.safetensors"
        data = weights.read_bytes()
        weights.write_bytes(data[:-1] + b"\1")
        with self.assertRaisesRegex(ValueError, "hashes"):
            validate_checkpoint(path)

    def test_migrated_run_keeps_legacy_checkpoints_before_threshold(self):
        old = checkpoint(self.root, 50)
        new = checkpoint(self.root, 100)
        (self.root / "run_manifest.json").write_text(json.dumps({
            "checkpoint_format": 1, "checkpoint_commit_required_from_step": 51}))
        self.assertEqual(validate_checkpoint(old)["format"], "legacy")
        with self.assertRaisesRegex(ValueError, "commit manifest is missing"):
            validate_checkpoint(new)
        commit_checkpoint(new)
        self.assertEqual(latest_checkpoint(self.root), new)

    def test_step_mismatch_rejected(self):
        for filename, value in (("trainer_state.json", {"global_step": 49}),
                                ("monitor_state.json", {"step": 49, "training_seconds": 100})):
            with self.subTest(filename=filename):
                path = checkpoint(self.root, 50 if filename.startswith("trainer") else 100)
                (path / filename).write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "step differs"):
                    validate_checkpoint(path)

    def test_truncated_archive_and_safetensors_rejected(self):
        for index, name in enumerate(("optimizer.pt", "adapter_model.safetensors")):
            with self.subTest(name=name):
                path = checkpoint(self.root, index + 1)
                item = path / name
                item.write_bytes(item.read_bytes()[:-2])
                with self.assertRaises(ValueError):
                    validate_checkpoint(path)

    def test_archive_crc_corruption_rejected(self):
        path = checkpoint(self.root, 50)
        item = path / "optimizer.pt"
        data = item.read_bytes()
        item.write_bytes(data.replace(b"fixture-only", b"changed-only", 1))
        with self.assertRaises(ValueError):
            validate_checkpoint(path)

    def test_invalid_json_and_nonfinite_elapsed_rejected(self):
        path = checkpoint(self.root, 50)
        item = path / "monitor_state.json"
        for text in ('{"step":', '{"step":50,"training_seconds":NaN}'):
            item.write_text(text)
            with self.assertRaises(ValueError):
                validate_checkpoint(path)

    def test_no_valid_checkpoint_fails_without_restarting_from_zero(self):
        (self.root / "checkpoint-partial").mkdir()
        (self.root / "checkpoint-50").mkdir()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaisesRegex(ValueError, "No complete"):
                latest_checkpoint(self.root)


if __name__ == "__main__":
    unittest.main()
