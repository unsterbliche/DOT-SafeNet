#!/usr/bin/env python3
"""Build deterministic Zenodo archives for DOT-SafeNet model inference."""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import tarfile
from pathlib import Path


FAMILIES = ("GPCR", "IonChannel", "Enzyme", "Kinase", "NHR", "Transporter", "Other")
VERSION = "1.0.0"


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_checkpoint_manifest(release_root: Path) -> list[dict[str, str]]:
    path = release_root / "weights" / "weights_manifest.tsv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 37:
        raise RuntimeError(f"Expected 37 checkpoint records, found {len(rows)}")
    return rows


def checkpoint_files(project_root: Path, release_root: Path) -> list[tuple[Path, Path]]:
    files: list[tuple[Path, Path]] = []
    for row in read_checkpoint_manifest(release_root):
        relative = Path(row["path"])
        source = project_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        actual_size = source.stat().st_size
        if actual_size != int(row["bytes"]):
            raise RuntimeError(f"Size mismatch for {relative}: {actual_size} != {row['bytes']}")
        actual_hash = sha256(source)
        if actual_hash.lower() != row["sha256"].lower():
            raise RuntimeError(f"SHA256 mismatch for {relative}")
        files.append((source, relative))
    files.extend(
        [
            (release_root / "weights" / "weights_manifest.tsv", Path("weights_manifest.tsv")),
            (release_root / "weights" / "README.md", Path("README.md")),
        ]
    )
    return files


def inference_files(project_root: Path) -> list[tuple[Path, Path]]:
    base = project_root / "PreMOTA" / "dataset_reg_multitask"
    files: list[tuple[Path, Path]] = []
    protein_count = 0
    for family in FAMILIES:
        family_root = base / family
        protein_file = family_root / "proteins.txt"
        embedding_root = family_root / "esm2"
        if not protein_file.is_file() or not embedding_root.is_dir():
            raise FileNotFoundError(f"Missing inference assets for {family}: {family_root}")
        protein_dict = ast.literal_eval(protein_file.read_text(encoding="utf-8"))
        if not isinstance(protein_dict, dict):
            raise TypeError(f"Expected a protein dictionary in {protein_file}")
        protein_count += len(protein_dict)
        for name in ("proteins.txt", "max_length.txt"):
            source = family_root / name
            if source.is_file():
                files.append((source, source.relative_to(project_root)))
        for source in sorted(path for path in embedding_root.rglob("*") if path.is_file()):
            files.append((source, source.relative_to(project_root)))
    if protein_count != 194:
        raise RuntimeError(f"Expected 194 proteins, found {protein_count}")
    return files


def normalized_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o644
    return info


def write_archive(output: Path, files: list[tuple[Path, Path]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for source, relative in sorted(files, key=lambda item: item[1].as_posix()):
                    archive.add(source, arcname=relative.as_posix(), recursive=False, filter=normalized_info)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    release_root = args.release_root.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint_archive = output_dir / f"dotsafenet-model-checkpoints-v{VERSION}.tar.gz"
    inference_archive = output_dir / f"dotsafenet-ot-profilenet-inference-assets-v{VERSION}.tar.gz"

    write_archive(checkpoint_archive, checkpoint_files(project_root, release_root))
    write_archive(inference_archive, inference_files(project_root))

    records = []
    for path in (checkpoint_archive, inference_archive):
        records.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    sums = output_dir / "SHA256SUMS.txt"
    sums.write_text("".join(f"{row['sha256']}  {row['name']}\n" for row in records), encoding="utf-8")
    (output_dir / "archive_manifest.json").write_text(
        json.dumps({"version": VERSION, "archives": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", "archives": records}, indent=2))


if __name__ == "__main__":
    main()
