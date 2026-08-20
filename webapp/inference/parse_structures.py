from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import MolStandardize


def _canonical(mol) -> str | None:
    if mol is None:
        return None
    chooser = MolStandardize.fragment.LargestFragmentChooser()
    return Chem.MolToSmiles(chooser.choose(mol), isomericSmiles=True)


def _parse_sdf(path: Path) -> list[dict]:
    supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
    items = []
    for index, mol in enumerate(supplier, start=1):
        smiles = _canonical(mol)
        if not smiles:
            continue
        name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") and mol.GetProp("_Name").strip() else f"molecule_{index}"
        items.append({"name": name, "smiles": smiles})
    return items


def _parse_mol2(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = [block for block in text.split("@<TRIPOS>MOLECULE") if block.strip()]
    items = []
    for index, block in enumerate(blocks, start=1):
        full_block = "@<TRIPOS>MOLECULE" + block
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        name = lines[0] if lines else f"molecule_{index}"
        mol = Chem.MolFromMol2Block(full_block, sanitize=True, removeHs=False)
        smiles = _canonical(mol)
        if smiles:
            items.append({"name": name, "smiles": smiles})
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--filename", required=True)
    args = parser.parse_args()
    path = Path(args.input)
    suffix = Path(args.filename).suffix.lower()
    if suffix == ".sdf":
        items = _parse_sdf(path)
    elif suffix == ".mol2":
        items = _parse_mol2(path)
    else:
        raise ValueError("Only SDF and MOL2 structure parsing is handled by RDKit.")
    print(json.dumps({"items": items}, ensure_ascii=True))


if __name__ == "__main__":
    main()
