"""Portable deployment export/load roundtrips and corruption checks use CPU only."""
import json
from pathlib import Path
import struct
import zipfile

import pytest

from inference.bundle import MEMBERS, MODEL_METADATA, export_bundle, verify_bundle
from training.data import file_digest


def fixture_experiment(root: Path):
    model = root / "model"
    model.mkdir()
    for name in MODEL_METADATA:
        (model / name).write_text("{}", encoding="utf-8")
    (model / "weights.safetensors").write_bytes(b"fixture base weights")
    (model / "snapshot_manifest.json").write_text(json.dumps({
        "repo": "Qwen/Qwen3.5-2B", "revision": "fixture-revision",
        "size": (model / "weights.safetensors").stat().st_size,
        "sha256": file_digest(model / "weights.safetensors"),
        "files": sorted(path.name for path in model.iterdir() if path.name != "snapshot_manifest.json")}))
    fingerprint = {path.name: file_digest(path) for path in model.iterdir()}
    experiment = root / "experiment"
    run = experiment / "r16-hybrid-s43"
    checkpoint = run / "checkpoint-400"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "C:/private/old-checkout/models/Qwen3.5-2B", "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 16}))
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (checkpoint / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 400, "best_metric": .14, "best_global_step": 400}))
    (checkpoint / "monitor_state.json").write_text(json.dumps({"step": 400, "training_seconds": 100, "best_loss": .14}))
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        with zipfile.ZipFile(checkpoint / name, "w") as archive:
            archive.writestr("archive/data.pkl", b"fixture-not-unpickled")
            archive.writestr("archive/version", b"3\n")
    manifest = {"training_complete": True, "pilot": False, "dataset_manifest_sha256": "fixture-dataset",
        "arguments": {"dtype": "bf16", "quantization": "none", "max_length": 512, "eval_samples": 128},
        "best_checkpoint": "old/checkout/r16-hybrid-s43/checkpoint-400",
        "model_fingerprint": {"local_files_sha256": fingerprint},
        "private_field_must_not_be_exported": "C:/private/training.jsonl"}
    (run / "run_manifest.json").write_text(json.dumps(manifest))
    (run / "monitor_state.json").write_text(json.dumps({"best_loss": .14}))
    (experiment / "final_comparison.json").write_text("not read for validation selection")
    return experiment, model


def test_export_extract_verify_roundtrip_has_only_portable_whitelist(tmp_path):
    experiment, model = fixture_experiment(tmp_path)
    output, archive = tmp_path / "bundle", tmp_path / "bundle.zip"
    result = export_bundle(experiment, model, output, archive)
    assert result["model_info"]["adapter_run"] == "r16-hybrid-s43"
    assert result["model_info"]["checkpoint_step"] == 400
    assert result["archive_sha256"] == file_digest(archive)
    assert "Copyright 2026 Alibaba Cloud" in (output / "LICENSE").read_text()
    assert "CMRC2018" in (output / "NOTICE").read_text()
    with zipfile.ZipFile(archive) as zipped:
        assert set(zipped.namelist()) == MEMBERS
        assert zipped.testzip() is None
        extracted = tmp_path / "relocated"
        zipped.extractall(extracted)
    identity = verify_bundle(extracted, model)
    assert identity["adapter_sha256"] == result["model_info"]["adapter_sha256"]
    assert identity["deployment_format"] == "evidoc-adapter-bundle-v1"
    for name in ("adapter_config.json", "deployment_manifest.json"):
        text = (extracted / name).read_text()
        assert "C:/" not in text and "private_field" not in text and "optimizer" not in text
    config = json.loads((extracted / "adapter_config.json").read_text())
    assert config["base_model_name_or_path"] == "Qwen/Qwen3.5-2B"


@pytest.mark.parametrize("target", ["adapter_model.safetensors", "base", "LICENSE", "NOTICE"])
def test_bundle_rejects_changed_weights_or_wrong_base(tmp_path, target):
    experiment, model = fixture_experiment(tmp_path)
    output = tmp_path / "bundle"
    export_bundle(experiment, model, output, tmp_path / "bundle.zip")
    item = model / "weights.safetensors" if target == "base" else output / target
    item.write_bytes(item.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_bundle(output, model)


def test_bundle_rejects_extra_files_and_path_traversal(tmp_path):
    experiment, model = fixture_experiment(tmp_path)
    output = tmp_path / "bundle"
    export_bundle(experiment, model, output, tmp_path / "bundle.zip")
    leaked = output / "optimizer.pt"
    leaked.write_bytes(b"not allowed")
    with pytest.raises(ValueError, match="exactly"):
        verify_bundle(output, model)
    leaked.unlink()
    manifest_path = output / "deployment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model"]["files_sha256"]["../secret"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="plain filenames"):
        verify_bundle(output, model)


def test_bundle_rejects_prompt_mismatch_and_export_overwrite(tmp_path):
    experiment, model = fixture_experiment(tmp_path)
    output, archive = tmp_path / "bundle", tmp_path / "bundle.zip"
    export_bundle(experiment, model, output, archive)
    with pytest.raises(ValueError, match="destination exists"):
        export_bundle(experiment, model, output, archive)
    manifest_path = output / "deployment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["training"]["prompt_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="prompt or length"):
        verify_bundle(output, model)


def test_snapshot_metadata_format_and_download_details_are_portable(tmp_path):
    experiment, model = fixture_experiment(tmp_path)
    output = tmp_path / "bundle"
    original = export_bundle(experiment, model, output, tmp_path / "bundle.zip")["model_info"]
    path = model / "snapshot_manifest.json"
    snapshot = json.loads(path.read_text())
    snapshot.update(downloaded_at="2030-01-01", mirror="a different download endpoint")
    snapshot["files"] = list(reversed(snapshot["files"]))
    path.write_bytes(json.dumps(snapshot, indent=4).replace("\n", "\r\n").encode())
    portable = verify_bundle(output, model)
    assert portable["adapter_sha256"] == original["adapter_sha256"]
    assert portable["snapshot_identity_sha256"] == original["snapshot_identity_sha256"]
    assert portable["training_snapshot_manifest_sha256"] == original["training_snapshot_manifest_sha256"]
    assert portable["model_files_sha256"]["snapshot_manifest.json"] != original["model_files_sha256"]["snapshot_manifest.json"]
    snapshot["size"] += 1
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="immutable identity"):
        verify_bundle(output, model)
