# Supplementary Figure 2 — pretraining dataset statistics

| Panel | Content | Source-data status |
|---|---|---|
| a | Protein sequence-length distribution | Complete, 7,258 proteins |
| b | Compound counts per protein target | Complete, 7,258 targets |
| c | Positive and negative interaction counts | Complete |
| d | Targets, compounds and interaction pairs | Complete |

Panel b was reconstructed from the archived V100 pretraining pair table. The source table contains 2,102,767 rows for 7,258 UniProt targets. The derivation and integrity checks are implemented in `scripts/prepare_panel_b_data.py`.