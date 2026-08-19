# Release contract

The manuscript contains six main figures and eight supplementary figures. `data/manifests/figure_manifest.csv` is the authoritative figure map.

Expected package counts:

- 194 target activity features.
- 194 target–exposure product features.
- one free-Cmax feature.
- 389 DOT-SafeNet numerical features.
- 18 SOC tasks.
- 57 clinical-dose test drugs and 907 observed drug–SOC labels.
- 14 final 600 dpi TIFF files.
- 12 code-derived figure entrypoints.
- 37 checkpoint files: 7 OT-ProfileNet, 5 PlasmaBindNet-Fu, 5 DoseExpoNet, and 20 DOT-SafeNet TensorFlow members.
- Checkpoint and inference-asset DOI: `10.5281/zenodo.22010299`.

Final manuscript mapping:

1. Figure 1: exposure-aware framework.
2. Figure 2: OT-ProfileNet data and performance.
3. Figure 3: PPB and dose–Cmax models.
4. Figure 4: ADR–target evidence and DOT-SafeNet training strategy.
5. Figure 5: DOT-SafeNet comparison, clinical fine-tuning, ROC, heatmap, and UMAP results.
6. Figure 6: four clinical-dose target-attribution cases.
7. Supplementary Figure 1: OT-ProfileNet architecture.
8. Supplementary Figures 2–4: OT-ProfileNet data and additional results.
9. Supplementary Figure 5: PlasmaBindNet-Fu and DoseExpoNet architecture.
10. Supplementary Figures 6–8: dataset characteristics, additional dose–Cmax cases, and complete SOC UMAP results.

Submitted TIFF checksums and dimensions are stored in `data/manifests/final_figure_files.csv`.
