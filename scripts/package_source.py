"""Build an explicit source-only delivery archive, excluding all private/runtime data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ("README.md", "THIRD_PARTY.md", "LICENSE", ".gitignore", ".gitattributes", ".python-version",
              "pyproject.toml", "requirements.txt", "start.ps1", "start-adapter.ps1",
              "pause-training.ps1", "resume-training.ps1")
TREES = ("app", "static", "training", "inference", "scripts", "tests", "docs", "evaluation", "data/demo", ".github")
EXTENSIONS = {".py", ".ps1", ".md", ".txt", ".html", ".css", ".js", ".json", ".csv", ".yaml", ".yml", ".toml"}
IMAGES = {".jpg", ".png"}


def source_paths(root: Path = ROOT) -> list[Path]:
    paths = {root / name for name in ROOT_FILES if (root / name).is_file()}
    for name in TREES:
        for path in (root / name).rglob("*"):
            permitted_image = path.suffix in IMAGES and path.parent == root / "docs/images"
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts and (path.suffix in EXTENSIONS or permitted_image):
                paths.add(path)
    if (root / "models/manifest.json").is_file():
        paths.add(root / "models/manifest.json")
    for path in paths:
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Source path escapes project root")
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError(f"Unexpected large source file: {path.relative_to(root)}")
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/Evidoc-source.zip")
    args = parser.parse_args()
    paths = source_paths()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            # Publication candidates must be readable text, not renamed weights.
            if path.suffix not in IMAGES:
                data.decode("utf-8-sig")
            archive.writestr(relative, data)
            manifest.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(args.output), "files": len(paths), "bytes": args.output.stat().st_size,
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
