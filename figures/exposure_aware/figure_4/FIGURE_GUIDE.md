# Figure 4 guide

## Scientific content

Figure 4 contains three main-text panels:

- ROC curves for 18 SOCs.
- Observed-label, low-exposure (20 mg/day), and high-exposure (250 mg/day) heatmaps.
- GAS-VAS and NER-PSY UMAP comparisons.

An additional horizontal strip contains the model comparison, the original 921-drug test-set AUROC before and after clinical-dose fine-tuning, and the AUROC for 57 CT-ADE-matched test drugs at their clinical-dose inputs. The clinical-dose subset contains 907 labeled drug-SOC pairs. Error bars denote SD across five matched folds. Fold points and connecting lines are omitted. Significance stars report two-sided paired t-tests (* p<0.05, ** p<0.01, *** p<0.001); exact p-values remain in the source tables.

The main composite uses a tall 6 x 3 ROC grid on the left, three slim vertical heatmaps on the right, and a single row of four UMAP panels below. GAS-VAS use one teal palette; NER-PSY use one purple palette. All legends remain separate.

The random-split model comparison and complete 18-SOC UMAP are stored together
under `supplementary_figure_8/`.

## Rendering

Run `python run.py` from this directory.

## Inputs

Source tables are stored under `data/`. ROC calculations, heatmap ordering,
UMAP coordinates, and the prediction threshold are unchanged during style
revisions.

## Outputs

Main panels:

- `outputs/figure4_model_comparison_and_clinical_finetuning.*`
- `outputs/figure4_finetune_auroc.*`
- `outputs/figure4_finetune_auprc.*`
- `outputs/figure4_original_test_finetune_auroc.*`
- `outputs/figure4_original_test_finetune_auprc.*`
- `outputs/figure4_clinical_dose_finetuning_statistics.csv`

- `outputs/figure4a_roc_18soc.*`
- `outputs/figure4b_observed_low_high_dose_heatmap.*`
- `outputs/figure4c_soc_relationship_umap.*`
- `outputs/figure_4.png` and `outputs/figure_4.pdf`

Separate legends:

- `outputs/legends/figure4a_roc_legend.*`
- `outputs/legends/figure4b_heatmap_legend.*`
- `outputs/legends/figure4c_umap_legend.*`

Typography, colors, dimensions, and spacing are controlled by `params.yaml`.

## Dose heatmap calculation

The held-out test input is the historical 100 mg/day reference. For 20 and 250 mg/day, log10 free Cmax is shifted by log10(dose/100), and all 194 target-by-exposure interaction columns are recomputed before five-fold inference. Drug order and SOC order are identical across the three heatmaps.
