"""Download attributed CMRC2018 and create document-disjoint experiment splits."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import urllib.request

from .data import add_missing_context, digest, normalize_row, split_documents, write_jsonl

DATASET_ID = "hfl/cmrc2018"
ORIGINAL_REPO = "ymcui/cmrc2018"


def fetch_json(url: str) -> tuple[dict, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "EviDoc-Training/1.0", "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(request, timeout=40) as response:
        content = response.read(40 * 1024 * 1024 + 1)
        compressed = response.headers.get("Content-Encoding") == "gzip"
    if compressed:
        content = gzip.decompress(content)
    if len(content) > 40 * 1024 * 1024:
        raise ValueError("Dataset response exceeds 40 MB")
    return json.loads(content.decode("utf-8-sig")), digest(content)


def original_rows(payload: dict, split: str):
    for article in payload["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                yield {**qa, "context": paragraph["context"], "title": article.get("title", "")}


def hf_viewer_rows(split: str) -> tuple[list[dict], str]:
    """Public HF viewer fallback without pyarrow; verify every row/page."""
    base = f"https://datasets-server.huggingface.co/rows?dataset=hfl%2Fcmrc2018&config=default&split={split}&length=100&offset="
    first, _ = fetch_json(base + "0")
    total = first["num_rows_total"]

    def page(offset):
        payload = first if offset == 0 else fetch_json(base + str(offset))[0]
        if payload.get("partial") or payload["num_rows_total"] != total:
            raise ValueError("HF viewer returned partial or changed data")
        received = payload["rows"]
        if [item["row_idx"] for item in received] != list(range(offset, min(offset + 100, total))):
            raise ValueError("HF viewer page has missing or unexpected row indices")
        if any(item.get("truncated_cells") for item in received):
            raise ValueError("HF viewer truncated a field; cannot use truncated supervision")
        return [item["row"] for item in received]

    with ThreadPoolExecutor(max_workers=8) as executor:
        pages = list(executor.map(page, range(0, total, 100)))
    rows = [row for page_rows in pages for row in page_rows]
    return rows, digest(json.dumps(rows, ensure_ascii=False, sort_keys=True))


def hf_parquet_rows(output: Path, revision: str | None):
    """Pin and hash actual official HF Parquet files; pyarrow is optional otherwise."""
    from scripts.download_models import download_parallel

    metadata, _ = fetch_json(f"https://huggingface.co/api/datasets/{DATASET_ID}/revision/{revision or 'main'}?blobs=true")
    revision = metadata["sha"]
    files = {"train": [], "validation": []}
    hashes = {}
    for item in metadata["siblings"]:
        name = item["rfilename"]
        split = next((split for split in files if name.startswith(f"data/{split}-") and name.endswith(".parquet")), None)
        if split is None:
            continue
        checksum = item.get("lfs", {}).get("sha256")
        if not checksum or not item.get("size"):
            raise ValueError(f"Parquet file lacks expected upstream LFS checksum: {name}")
        destination = output / "raw" / Path(name).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{revision}/{name}"
        print(f"Downloading verified HF Parquet {name}", flush=True)
        download_parallel(url, destination, checksum, item["size"], connections=2)
        files[split].append(destination)
        hashes[name] = checksum
    if any(not paths for paths in files.values()):
        raise ValueError("Official HF train/validation Parquet files were not found")
    import pyarrow.parquet as parquet

    return {split: [row for path in paths for row in parquet.read_table(path).to_pylist()]
            for split, paths in files.items()}, revision, hashes


def prepare(source: str, revision: str | None, output: Path, seed: int,
            validation_fraction: float, missing_fraction: float) -> dict:
    if (output / "manifest.json").exists():
        raise ValueError(f"Dataset already exists at {output}; choose another output to preserve provenance.")
    rows, input_counts, raw_hashes = [], {}, {}
    invalid = 0
    if source == "github":
        if not revision:
            commit, _ = fetch_json(f"https://api.github.com/repos/{ORIGINAL_REPO}/git/ref/heads/master")
            revision = commit["object"]["sha"]
        for split, filename in (("train", "cmrc2018_train.json"), ("validation", "cmrc2018_dev.json")):
            url = f"https://raw.githubusercontent.com/{ORIGINAL_REPO}/{revision}/squad-style-data/{filename}"
            print(f"Downloading {split} from commit {revision}", flush=True)
            try:
                payload, raw_hashes[split] = fetch_json(url)
            except (TimeoutError, OSError):
                mirror = f"https://cdn.jsdelivr.net/gh/{ORIGINAL_REPO}@{revision}/squad-style-data/{filename}"
                print("Direct origin unavailable; trying the same commit via jsDelivr CDN", flush=True)
                payload, raw_hashes[split] = fetch_json(mirror)
            raw_rows = list(original_rows(payload, split))
            input_counts[split] = len(raw_rows)
            for row in raw_rows:
                normalized = normalize_row(row, split)
                if normalized:
                    rows.append(normalized)
                else:
                    invalid += 1
    elif source == "hf-parquet":
        raw_partitions, revision, raw_hashes = hf_parquet_rows(output, revision)
        for split, raw_rows in raw_partitions.items():
            input_counts[split] = len(raw_rows)
            for row in raw_rows:
                normalized = normalize_row(row, split)
                if normalized:
                    rows.append(normalized)
                else:
                    invalid += 1
    elif source == "hf-api":
        metadata_url = f"https://huggingface.co/api/datasets/{DATASET_ID}"
        metadata, _ = fetch_json(metadata_url)
        current_revision = metadata["sha"]
        if revision and revision != current_revision:
            raise ValueError("HF viewer serves the current revision only; use --source hf for an older pinned revision")
        revision = current_revision
        for split in ("train", "validation"):
            print(f"Downloading HF viewer {split} at observed revision {revision}", flush=True)
            raw_rows, raw_hashes[split] = hf_viewer_rows(split)
            input_counts[split] = len(raw_rows)
            for row in raw_rows:
                normalized = normalize_row(row, split)
                if normalized:
                    rows.append(normalized)
                else:
                    invalid += 1
        after, _ = fetch_json(metadata_url)
        if after["sha"] != revision:
            raise ValueError("Dataset changed during download; rerun preparation to obtain a consistent snapshot")
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi

        revision = HfApi().dataset_info(DATASET_ID, revision=revision or "main").sha
        dataset = load_dataset(DATASET_ID, revision=revision)
        for split in ("train", "validation"):
            input_counts[split] = len(dataset[split])
            raw_hashes[split] = dataset[split]._fingerprint
            for row in dataset[split]:
                normalized = normalize_row(row, split)
                if normalized:
                    rows.append(normalized)
                else:
                    invalid += 1
    partitions, split_report = split_documents(rows, seed, validation_fraction)
    if any(not partitions[split] for split in partitions):
        raise ValueError("A split is empty; review data parsing and group split parameters")
    partitions["train"] = add_missing_context(partitions["train"], missing_fraction, seed)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": "CMRC2018",
        "source": source, "repository": ORIGINAL_REPO if source == "github" else DATASET_ID,
        "revision": revision, "seed": seed, "validation_fraction": validation_fraction,
        "license": "CC-BY-SA-4.0 (as declared on the hfl/cmrc2018 dataset card)",
        "attribution": "Cui et al., A Span-Extraction Dataset for Chinese Machine Reading Comprehension, EMNLP-IJCNLP 2019",
        "source_url": "https://github.com/ymcui/cmrc2018",
        "transport_note": "GitHub raw at pinned commit, with jsDelivr gh CDN fallback at the identical commit; content SHA256 recorded.",
        "dataset_card_url": "https://huggingface.co/datasets/hfl/cmrc2018",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "paper_url": "https://aclanthology.org/D19-1600/",
        "split_policy": "Group source document/title and exact normalized context; official development is our held-out test; remove train examples from intersecting groups.",
        "evaluation_scope": "Custom document-disjoint experiment; not official CMRC leaderboard. Does not rule out base-model pretraining contamination or near-duplicate articles.",
        "missing_context_training_fraction_requested": missing_fraction,
        "missing_context_policy": "Controlled empty-context augmentation only in training; never reuse synthetic RAG demo evaluation.",
        "input_rows": input_counts, "raw_hashes_or_hf_fingerprints": raw_hashes,
        "invalid_rows_removed": invalid, **split_report, "splits": {},
    }
    for split, members in partitions.items():
        path = output / f"{split}.jsonl"
        manifest["splits"][split] = {"file": path.name, "sha256": write_jsonl(path, members),
                                      "rows": len(members), "groups": len({row["group_id"] for row in members}),
                                      "missing_context_rows": sum(row["is_missing_context"] for row in members)}
    if source == "hf-api":
        manifest["transport_note"] = "HF dataset viewer API; repository revision checked before and after download; pages validated for row continuity and nontruncation; reconstructed raw-row SHA256 recorded. Viewer does not expose a per-page commit, so this is not a byte-identical Parquet snapshot guarantee."
    elif source == "hf-parquet":
        manifest["transport_note"] = "Official HF Parquet files at pinned commit, verified against upstream LFS SHA256 and exact byte lengths; read with pyarrow."
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("github", "hf", "hf-api", "hf-parquet"), default="hf-parquet")
    parser.add_argument("--revision", help="Pinned upstream commit; resolves the current commit when omitted")
    parser.add_argument("--output", type=Path, default=Path("data/training/cmrc2018"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=.1)
    parser.add_argument("--missing-context-fraction", type=float, default=.05)
    args = parser.parse_args()
    manifest = prepare(args.source, args.revision, args.output, args.seed,
                       args.validation_fraction, args.missing_context_fraction)
    print(json.dumps({"output": str(args.output), "revision": manifest["revision"], "splits": manifest["splits"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
