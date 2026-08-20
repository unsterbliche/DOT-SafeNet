from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


DEFAULT_PART_SIZE = 48 * 1024 * 1024


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Split DOT-SafeNet Zenodo archives into deterministic consecutive parts.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--part-size-mib", type=int, default=48)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    part_size = args.part_size_mib * 1024 * 1024
    output.mkdir(parents=True, exist_ok=True)

    manifest = {"part_size_bytes": part_size, "archives": []}
    part_hashes: list[tuple[str, str]] = []
    for archive in sorted(source.glob("*.tar.gz")):
        parts = []
        with archive.open("rb") as handle:
            index = 0
            while True:
                data = handle.read(part_size)
                if not data:
                    break
                part = output / f"{archive.name}.part-{index:03d}"
                part.write_bytes(data)
                checksum = hashlib.sha256(data).hexdigest()
                part_hashes.append((checksum, part.name))
                parts.append({"name": part.name, "bytes": len(data), "sha256": checksum})
                index += 1
        manifest["archives"].append({"name": archive.name, "bytes": archive.stat().st_size, "sha256": sha256(archive), "parts": parts})

    for name in ["README.md", "SHA256SUMS.txt", "archive_manifest.json", "weights_manifest.tsv", "inference_assets_manifest.csv"]:
        shutil.copy2(source / name, output / name)
    (output / "PARTS_SHA256SUMS.txt").write_text("".join(f"{checksum}  {name}\n" for checksum, name in part_hashes), encoding="utf-8")
    (output / "multipart_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "parts": len(part_hashes), "output": str(output)}))


if __name__ == "__main__":
    main()
