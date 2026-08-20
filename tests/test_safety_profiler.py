from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safety_profiler.core import assemble_dotsafenet_features, empirical_percentile, load_contract, standardize_input_table


class SafetyProfilerTest(unittest.TestCase):
    def test_reference_contract(self):
        contract = load_contract(ROOT)
        self.assertEqual(len(contract.target_ids), 194)
        self.assertEqual(len(contract.feature_columns), 389)
        reference = pd.read_csv(ROOT / "data/reference/target_feature_reference.csv")
        self.assertEqual(len(reference), 194)
        self.assertEqual(set(reference["reference_source"]), {"base_random_train_X"})

    def test_input_schema(self):
        frame = pd.DataFrame({"smiles": ["CCO"], "dose": [10]})
        result = standardize_input_table(frame, canonicalize=False)
        self.assertEqual(result.loc[0, "compound_id"], "CMPD0001")
        self.assertEqual(result.loc[0, "dose_mg_day"], 10)
        with self.assertRaises(ValueError):
            standardize_input_table(pd.DataFrame({"smiles": ["CCO"], "dose": [0]}), canonicalize=False)

    def test_feature_assembly_uses_published_transform(self):
        contract = load_contract(ROOT)
        input_df = pd.DataFrame({
            "compound_id": ["X"], "name": ["Example"], "smiles": ["CCO"],
            "dose_mg_day": [10.0], "route": ["oral"], "molecular_weight": [100.0],
        })
        ot = {"CCO": {target: 7.0 for target in contract.target_ids}}
        ppb = pd.DataFrame({
            "smiles": ["CCO"], "pred_ppb_mean": [0.5], "pred_fu_mean": [0.5],
            "pred_ppb_std": [0.0], "pred_fu_std": [0.0],
        })
        cmax = pd.DataFrame({
            "compound_id": ["X"], "predicted_log10_cmax_ug_ml_mean": [0.0],
            "predicted_log10_cmax_ug_ml_sd": [0.0], "predicted_cmax_ug_ml_mean": [1.0],
            "predicted_cmax_ug_ml_sd": [0.0],
        })
        features, exposure, margins = assemble_dotsafenet_features(input_df, ot, ppb, cmax, contract)
        first = contract.target_ids[0]
        self.assertAlmostEqual(features.loc[0, first], -1.0)
        self.assertAlmostEqual(exposure.loc[0, "free_cmax_uM"], 5.0)
        self.assertAlmostEqual(features.loc[0, f"{first}*Cmax"], -np.log10(5.0))
        self.assertAlmostEqual(margins.loc[0, "margin_log10_AC50_over_free_Cmax"], -1.0 - np.log10(5.0))

    def test_background_percentile(self):
        grid = pd.read_csv(ROOT / "data/reference/soc_background_quantiles.csv.gz")
        subset = grid[(grid["adr_abbr"] == "CAR") & (grid["label_subset"] == "all")]
        median_score = float(subset.loc[np.isclose(subset["percentile"], 50.0), "score"].iloc[0])
        self.assertAlmostEqual(empirical_percentile(median_score, grid, "CAR"), 50.0, places=6)

    def test_paper_case_replay(self):
        with tempfile.TemporaryDirectory(prefix="multisafe_profiler_") as directory:
            subprocess.run([
                sys.executable, str(ROOT / "scripts/profile_molecule.py"), "replay", "--output", directory
            ], cwd=ROOT, check=True)
            output = Path(directory)
            adr = pd.read_csv(output / "adr_predictions.csv")
            self.assertEqual(adr["compound_id"].nunique(), 4)
            self.assertEqual(len(adr), 72)
            self.assertTrue((output / "report.html").exists())
            self.assertTrue((output / "report.json").exists())

    def test_distributable_skill_wrapper(self):
        with tempfile.TemporaryDirectory(prefix="multisafe_skill_") as directory:
            subprocess.run([
                sys.executable,
                str(ROOT / "skills/dotsafenet-safety-profiler/scripts/profile.py"),
                "--release-root", str(ROOT), "replay", "--input",
                str(ROOT / "tests/fixtures/paper_citalopram_20mg.csv"),
                "--output", directory,
            ], cwd=Path(directory), check=True)
            adr = pd.read_csv(Path(directory) / "adr_predictions.csv")
            self.assertEqual(len(adr), 18)
            self.assertEqual(set(adr["compound_id"]), {"PAPER_CITALOPRAM_20"})

    def test_full_pipeline_smoke_record(self):
        root = ROOT / "data/examples/citalopram_full_inference"
        adr = pd.read_csv(root / "adr_predictions.csv")
        margins = pd.read_csv(root / "target_margins.csv")
        attribution = pd.read_csv(root / "target_attribution.csv")
        self.assertEqual(len(adr), 18)
        self.assertEqual(len(margins), 194)
        self.assertEqual(len(attribution), 18 * 194)
        self.assertTrue(np.isfinite(adr["probability_mean"]).all())
        self.assertTrue(np.isfinite(margins["margin_log10_AC50_over_free_Cmax"]).all())


if __name__ == "__main__":
    unittest.main()
