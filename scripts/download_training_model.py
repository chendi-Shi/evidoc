"""Download a version-locked Qwen training snapshot, checking official hashes."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
from scripts.download_models import download_parallel

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "2B": {"repo": "Qwen/Qwen3.5-2B", "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "size": 4548221488, "sha256": "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"},
    "0.8B": {"repo": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "size": 1746942600, "sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=MODELS, default="2B")
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--mirror", choices=["huggingface", "modelscope"], default="modelscope")
    args = parser.parse_args()
    spec = MODELS[args.size]
    directory = ROOT / "models" / ("Qwen3.5-" + args.size)
    directory.mkdir(parents=True, exist_ok=True)
    api = f"https://huggingface.co/api/models/{spec['repo']}/revision/{spec['revision']}?blobs=true"
    with urlopen(Request(api, headers={"User-Agent": "Evidoc/0.1"}), timeout=60) as response:
        metadata = json.load(response)
    files = []
    for item in metadata["siblings"]:
        name = item["rfilename"]
        if "/" in name or not name.endswith((".json", ".safetensors", ".txt", ".model", ".jinja")):
            continue
        if name.endswith(".safetensors"):
            lfs = item.get("lfs", {})
            if lfs.get("sha256") != spec["sha256"] or item["size"] != spec["size"]:
                raise ValueError("Model weight metadata does not match pinned official checksum.")
            url = (f"https://modelscope.cn/models/{spec['repo']}/resolve/master/{name}" if args.mirror == "modelscope"
                   else f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{name}")
            download_parallel(url, directory / name, spec["sha256"], spec["size"], args.connections)
        else:
            url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{name}"
            with urlopen(Request(url, headers={"User-Agent": "Evidoc/0.1"}), timeout=90) as response:
                data = response.read()
            if len(data) != item["size"]:
                raise ValueError(f"Incomplete metadata file: {name}")
            lfs_hash = item.get("lfs", {}).get("sha256")
            blob_hash = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if ((lfs_hash and hashlib.sha256(data).hexdigest() != lfs_hash)
                or (not lfs_hash and item.get("blobId") and blob_hash != item["blobId"])):
                raise ValueError(f"Metadata checksum mismatch: {name}")
            (directory / name).write_bytes(data)
        files.append(name)
        print("Verified " + name, flush=True)
    (directory / "snapshot_manifest.json").write_text(json.dumps({**spec, "files": files}, indent=2), encoding="utf-8")
    print(f"Training snapshot ready: {directory}", flush=True)


if __name__ == "__main__":
    main()
