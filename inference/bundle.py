"""Portable LoRA deployment bundles with provenance, no training data or optimizer."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path, PureWindowsPath
import re
import shutil
import zipfile

from training.checkpoints import _validate_safetensors
from training.data import digest, file_digest, messages_for

from .adapter import Limits, read_json, select_adapter, verify_artifacts


FORMAT = "evidoc-adapter-bundle-v1"
MODEL_ID = "Qwen/Qwen3.5-2B"
MEMBERS = {"adapter_model.safetensors", "adapter_config.json", "deployment_manifest.json", "LICENSE", "NOTICE"}
MODEL_METADATA = {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "snapshot_manifest.json"}
UPSTREAM_LICENSE_SHA256 = "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
UPSTREAM_LICENSE = Path(__file__).resolve().parents[1] / "docs/licenses/Qwen3.5-LICENSE.txt"


def prompt_fingerprint() -> str:
    template = messages_for({"context": "{context}", "question": "{question}"})
    return digest(json.dumps(template, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def snapshot_identity(snapshot: dict) -> dict:
    """Canonical immutable identity, excluding local download timestamps/mirrors.

    Only snapshot_manifest.json is generated locally. All upstream model and
    tokenizer files continue to require their original exact byte fingerprints.
    """
    files = snapshot.get("files")
    if (not isinstance(files, list) or any(not isinstance(name, str) for name in files)
            or len(files) != len(set(files)) or type(snapshot.get("size")) is not int or snapshot["size"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(snapshot.get("sha256", "")))):
        raise ValueError("Snapshot manifest lacks immutable weight identity")
    return {"repo": snapshot.get("repo"), "revision": snapshot.get("revision"),
            "size": snapshot["size"], "sha256": snapshot["sha256"], "files": sorted(files)}


def snapshot_fingerprint(identity: dict) -> str:
    return digest(json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _portable_json(value):
    if isinstance(value, dict):
        for child in value.values():
            _portable_json(child)
    elif isinstance(value, list):
        for child in value:
            _portable_json(child)
    elif isinstance(value, str) and (PureWindowsPath(value).is_absolute() or value.startswith("/")):
        raise ValueError("Deployment metadata contains an absolute path")


def _file_hashes(directory: Path, hashes: dict, *, sizes: bool):
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Missing file fingerprints")
    for name, metadata in hashes.items():
        if not isinstance(name, str) or any(char in name for char in ("/", "\\", ":")) or name in {".", ".."}:
            raise ValueError("Fingerprint paths must be plain filenames")
        item = (directory / name).resolve()
        expected = metadata.get("sha256") if sizes and isinstance(metadata, dict) else metadata
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Invalid SHA256: {name}")
        if item.parent != directory or not item.is_file() or file_digest(item) != expected:
            raise ValueError(f"File fingerprint mismatch: {name}")
        if sizes and (type(metadata.get("size")) is not int or item.stat().st_size != metadata["size"]):
            raise ValueError(f"File size mismatch: {name}")


def verify_bundle(bundle: Path, model_dir: Path) -> dict:
    """Read only inert JSON/safetensors headers and hashes; do not import CUDA."""
    bundle, model_dir = bundle.resolve(), model_dir.resolve()
    if not bundle.is_dir() or {path.name for path in bundle.iterdir()} != MEMBERS:
        raise ValueError("Bundle must contain exactly the adapter files, manifest, LICENSE and NOTICE")
    manifest = read_json(bundle / "deployment_manifest.json")
    expected_keys = {"format", "created_at_utc", "model", "adapter", "selection", "training", "files"}
    if set(manifest) != expected_keys or manifest.get("format") != FORMAT:
        raise ValueError("Unsupported deployment manifest format")
    _portable_json(manifest)
    model, adapter = manifest["model"], manifest["adapter"]
    training, selection = manifest["training"], manifest["selection"]
    if (model.get("id") != MODEL_ID or model.get("dtype") != "bf16" or model.get("quantization") != "none"
            or not isinstance(model.get("revision"), str) or not model["revision"]):
        raise ValueError("Bundle model profile is not the supported Qwen3.5-2B BF16 profile")
    if (training.get("max_length") != 512 or training.get("prompt_sha256") != prompt_fingerprint()
            or training.get("serving_limits") != Limits().as_dict()):
        raise ValueError("Bundle prompt or length profile does not match this inference implementation")
    value = selection.get("value")
    if (selection.get("metric") != "validation_loss" or type(value) not in (float, int)
            or not math.isfinite(value) or value < 0):
        raise ValueError("Invalid bundle validation-selection metadata")
    if (not re.fullmatch(r"[a-zA-Z0-9_-]+", str(adapter.get("run", "")))
            or not re.fullmatch(r"checkpoint-\d+", str(adapter.get("checkpoint", "")))
            or type(adapter.get("checkpoint_step")) is not int
            or adapter["checkpoint"] != f"checkpoint-{adapter['checkpoint_step']}"):
        raise ValueError("Invalid portable adapter identity")
    if set(manifest["files"]) != MEMBERS - {"deployment_manifest.json"}:
        raise ValueError("Unexpected adapter file list")
    _file_hashes(bundle, manifest["files"], sizes=True)
    if manifest["files"]["LICENSE"]["sha256"] != UPSTREAM_LICENSE_SHA256:
        raise ValueError("Bundle LICENSE differs from the pinned upstream license")
    _validate_safetensors(bundle / "adapter_model.safetensors")
    config = read_json(bundle / "adapter_config.json")
    _portable_json(config)
    if config.get("base_model_name_or_path") != MODEL_ID or config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise ValueError("Bundle adapter config has a different base model or adapter type")
    fingerprints = model.get("files_sha256", {})
    if not MODEL_METADATA <= set(fingerprints) or not any(name.endswith(".safetensors") for name in fingerprints):
        raise ValueError("Bundle lacks required base model and tokenizer fingerprints")
    _file_hashes(model_dir, {name: checksum for name, checksum in fingerprints.items()
                            if name != "snapshot_manifest.json"}, sizes=False)
    snapshot = read_json(model_dir / "snapshot_manifest.json")
    if snapshot.get("repo") != MODEL_ID or snapshot.get("revision") != model["revision"]:
        raise ValueError("Bundle model revision does not match downloaded snapshot")
    observed_snapshot = snapshot_identity(snapshot)
    if (observed_snapshot != model.get("snapshot_identity")
            or snapshot_fingerprint(observed_snapshot) != model.get("snapshot_identity_sha256")):
        raise ValueError("Downloaded snapshot immutable identity does not match the bundle")
    actual_hashes = {**fingerprints, "snapshot_manifest.json": file_digest(model_dir / "snapshot_manifest.json")}
    return {"model": "Qwen3.5-2B + Evidoc LoRA", "model_id": MODEL_ID,
            "model_revision": model["revision"], "adapter_run": adapter["run"],
            "checkpoint": adapter["checkpoint"], "checkpoint_step": adapter["checkpoint_step"],
            "adapter_sha256": manifest["files"]["adapter_model.safetensors"]["sha256"],
            "adapter_config_sha256": manifest["files"]["adapter_config.json"]["sha256"],
            "model_files_sha256": actual_hashes, "validation_loss": value,
            "training_snapshot_manifest_sha256": fingerprints["snapshot_manifest.json"],
            "snapshot_identity_sha256": model["snapshot_identity_sha256"],
            "selection_rule": selection["rule"], "device": "cuda:0", "dtype": "bf16", "quantization": "none",
            "deployment_format": FORMAT, "deployment_manifest_sha256": file_digest(bundle / "deployment_manifest.json")}


def export_bundle(experiment: Path, model_dir: Path, output: Path, archive: Path) -> dict:
    """Export a fresh, verified deployment directory and ZIP from validation selection."""
    output, archive = output.resolve(), archive.resolve()
    if output.exists() or archive.exists():
        raise ValueError("Export destination exists; choose a fresh directory and ZIP to preserve artifacts")
    if archive == output or archive.is_relative_to(output):
        raise ValueError("ZIP archive must be outside the bundle directory")
    chosen = select_adapter(experiment)
    identity = verify_artifacts(chosen, model_dir)
    if not UPSTREAM_LICENSE.is_file() or file_digest(UPSTREAM_LICENSE) != UPSTREAM_LICENSE_SHA256:
        raise ValueError("Pinned upstream model license is missing or has changed")
    config = read_json(chosen.checkpoint / "adapter_config.json")
    # PEFT needs this identifier but no original checkout path when a base model
    # is explicitly supplied by the caller. Preserve every other adapter setting.
    config["base_model_name_or_path"] = MODEL_ID
    _portable_json(config)
    output.mkdir(parents=True)
    shutil.copyfile(chosen.checkpoint / "adapter_model.safetensors", output / "adapter_model.safetensors")
    shutil.copyfile(UPSTREAM_LICENSE, output / "LICENSE")
    notice = ("Evidoc LoRA deployment bundle\n\n"
        "Upstream base model: Qwen/Qwen3.5-2B\n"
        "Source: https://huggingface.co/Qwen/Qwen3.5-2B\n"
        f"Pinned revision: {identity['model_revision']}\n"
        "Upstream copyright: Copyright 2026 Alibaba Cloud\n"
        "The unmodified upstream Apache License 2.0 text is included in LICENSE.\n\n"
        "Modification: Evidoc trained the supplied LoRA adapter for short Chinese document QA.\n"
        f"Selected run: {identity['adapter_run']}\n"
        f"Selected checkpoint: {identity['checkpoint']}\n"
        f"Selection: minimum validation loss ({identity['validation_loss']}); test scores were not used.\n\n"
        "Training-data attribution: Cui et al., A Span-Extraction Dataset for Chinese Machine Reading\n"
        "Comprehension, EMNLP-IJCNLP 2019 (CMRC2018).\n"
        "Source: https://github.com/ymcui/cmrc2018\n"
        "Dataset mirror: https://huggingface.co/datasets/hfl/cmrc2018\n"
        f"Dataset revision: {chosen.manifest.get('dataset_revision', 'not recorded')}\n"
        "The dataset card declares CC-BY-SA-4.0: https://creativecommons.org/licenses/by-sa/4.0/\n"
        "This bundle includes no dataset examples, evaluation questions, or optimizer state.\n\n"
        "The Evidoc software-code license does not replace the upstream model or dataset terms,\n"
        "and does not independently license model weights. Consult the cited upstream terms.\n")
    (output / "NOTICE").write_text(notice, encoding="utf-8", newline="\n")
    (output / "adapter_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = {name: {"sha256": file_digest(output / name), "size": (output / name).stat().st_size}
             for name in ("adapter_model.safetensors", "adapter_config.json", "LICENSE", "NOTICE")}
    snapshot = snapshot_identity(read_json(model_dir / "snapshot_manifest.json"))
    manifest = {"format": FORMAT, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"id": MODEL_ID, "revision": identity["model_revision"], "dtype": "bf16", "quantization": "none",
                  "files_sha256": identity["model_files_sha256"], "snapshot_identity": snapshot,
                  "snapshot_identity_sha256": snapshot_fingerprint(snapshot),
                  "snapshot_hash_policy": "Local snapshot_manifest.json uses immutable JSON identity; original training byte hash retained above. All other files require exact byte hashes."},
        "adapter": {"run": identity["adapter_run"], "checkpoint": identity["checkpoint"],
                    "checkpoint_step": identity["checkpoint_step"]},
        "selection": {"metric": "validation_loss", "value": identity["validation_loss"], "rule": identity["selection_rule"],
                      "dataset_manifest_sha256": chosen.manifest["dataset_manifest_sha256"],
                      "eval_samples": chosen.manifest["arguments"]["eval_samples"]},
        "training": {"max_length": 512, "prompt_sha256": prompt_fingerprint(), "serving_limits": Limits().as_dict()},
        "files": files}
    _portable_json(manifest)
    (output / "deployment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    deployed_identity = verify_bundle(output, model_dir)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for name in sorted(MEMBERS):
            zipped.write(output / name, arcname=name)
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None or set(zipped.namelist()) != MEMBERS:
            raise ValueError("Exported archive failed its roundtrip CRC/member check")
    return {"bundle": str(output), "archive": str(archive), "archive_size": archive.stat().st_size,
            "archive_sha256": file_digest(archive), "model_info": deployed_identity}
