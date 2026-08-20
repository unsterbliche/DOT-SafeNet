from __future__ import annotations

from dataclasses import dataclass


PAPER_CASE_GROUP = "clinical_dose_target_attribution_cases"


@dataclass(frozen=True)
class Sample:
    key: str
    name: str
    smiles: str
    dose_panel_mg: tuple[float, ...]
    highlighted_soc: str
    highlighted_target: str
    paper_case_group: str = PAPER_CASE_GROUP

    def to_api(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "smiles": self.smiles,
            "dose_panel_mg": list(self.dose_panel_mg),
            "highlighted_soc": self.highlighted_soc,
            "highlighted_target": self.highlighted_target,
            "paper_case_group": self.paper_case_group,
        }


SAMPLES: dict[str, Sample] = {
    "meclofenamic_acid": Sample(
        key="meclofenamic_acid",
        name="Meclofenamic acid",
        smiles="Cc1ccc(Cl)c(Nc2ccccc2C(=O)O)c1Cl",
        dose_panel_mg=(75, 150, 300),
        highlighted_soc="GAS",
        highlighted_target="PTGS1",
    ),
    "citalopram": Sample(
        key="citalopram",
        name="Citalopram",
        smiles="CN(C)CCCC1(c2ccc(F)cc2)OCc2cc(C#N)ccc21",
        dose_panel_mg=(10, 20, 40),
        highlighted_soc="CAR",
        highlighted_target="KCNH2",
    ),
    "spironolactone": Sample(
        key="spironolactone",
        name="Spironolactone",
        smiles="CC(=O)S[C@@H]1CC2=CC(=O)CC[C@]2(C)[C@H]2CC[C@@]3(C)[C@@H](CC[C@@]34CCC(=O)O4)[C@H]12",
        dose_panel_mg=(12.5, 25, 50),
        highlighted_soc="REP",
        highlighted_target="AR",
    ),
    "candesartan_cilexetil": Sample(
        key="candesartan_cilexetil",
        name="Candesartan cilexetil",
        smiles="CCOc1nc2cccc(C(=O)OC(C)OC(=O)OC3CCCCC3)c2n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1",
        dose_panel_mg=(8, 16, 32),
        highlighted_soc="VAS/REN",
        highlighted_target="AGTR1",
    ),
}


def list_samples() -> list[dict]:
    return [sample.to_api() for sample in SAMPLES.values()]


def get_sample(key: str) -> Sample | None:
    return SAMPLES.get(key.lower())
