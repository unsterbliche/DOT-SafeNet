from __future__ import annotations

import argparse
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import MolStandardize


def depict(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES.")
    chooser = MolStandardize.fragment.LargestFragmentChooser()
    mol = chooser.choose(mol)
    Chem.rdDepictor.Compute2DCoords(mol)
    drawer = Draw.MolDraw2DSVG(440, 300)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.padding = 0.08
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(depict(args.smiles), encoding="utf-8")


if __name__ == "__main__":
    main()
