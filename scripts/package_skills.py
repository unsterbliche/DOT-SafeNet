#!/usr/bin/env python3
"""Create deterministic distributable archives for the release skills."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "skill"
DIST = ROOT / "dist"
FIXED_TIME = (2026, 8, 18, 0, 0, 0)


def package(skill: Path) -> Path:
    output = DIST / f"{skill.name}.skill.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(skill.rglob("*"), key=lambda value: value.as_posix()):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            relative = Path(skill.name) / path.relative_to(skill)
            info = zipfile.ZipInfo(relative.as_posix(), FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return output


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    DIST.mkdir(parents=True, exist_ok=True)
    archives = [package(skill) for skill in sorted(SKILL_ROOT.iterdir()) if (skill / "SKILL.md").is_file()]
    lines = [f"{sha256(path)}  {path.name}" for path in archives]
    (DIST / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path in archives:
        print(path.relative_to(ROOT), path.stat().st_size, sha256(path))


if __name__ == "__main__":
    main()
