from __future__ import annotations

import csv
import importlib.util
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VALIDATE = load_module("release_validate", ROOT / "scripts" / "validate_release.py")
METRICS = load_module("release_metrics", ROOT / "scripts" / "reproduce_metrics.py")


class ReleaseContractTest(unittest.TestCase):
    def test_package_contract(self):
        report = VALIDATE.validate_package(check_hashes=False)
        self.assertEqual(report["status"], "passed", report["errors"])
        self.assertEqual(report["targets"], 194)
        self.assertEqual(report["soc_tasks"], 18)
        self.assertEqual(report["manuscript_figures"], 14)

    def test_figure_order(self):
        with (ROOT / "data" / "manifests" / "figure_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
            names = [row["figure"] for row in csv.DictReader(handle)]
        self.assertEqual(names[:6], [f"Figure {i}" for i in range(1, 7)])
        self.assertEqual(names[6:], [f"Supplementary Figure {i}" for i in range(1, 9)])

    def test_soft_pseudo_dose_schedule(self):
        config = yaml.safe_load((ROOT / "configs" / "dotsafenet_base.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["positive_soft_pseudo_labels"], "0.30103:0.92:0.60,0.60206:0.96:0.80,-0.30103:0.75:0.20")
        self.assertEqual(config["negative_soft_pseudo_labels"], "-0.60206:0.02:0.80,-0.30103:0.04:0.60,0.30103:0.18:0.20,0.60206:0.30:0.10")
        self.assertEqual(config["monotonicity"]["loss_weight"], 0.5)

    def test_clinical_metrics_recalculate_exactly(self):
        calculated, difference = METRICS.clinical_57_metrics()
        self.assertEqual(len(calculated), 10)
        self.assertLessEqual(difference, 1e-12)
        means = calculated.groupby("stage")[["AUROC", "AUPRC"]].mean()
        self.assertAlmostEqual(means.loc["Before", "AUROC"], 0.6120178468624065, places=12)
        self.assertAlmostEqual(means.loc["After", "AUROC"], 0.6222087890999808, places=12)

    def test_per_soc_roc_contains_all_tasks(self):
        table = METRICS.per_soc_auroc()
        self.assertEqual(len(table), 18)
        self.assertTrue(table["auroc"].between(0.5, 1.0).all())


if __name__ == "__main__":
    unittest.main()
