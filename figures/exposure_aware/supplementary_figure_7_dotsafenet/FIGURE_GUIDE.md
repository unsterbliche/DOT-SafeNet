# Figure 4 source panels: DOT-SafeNet strategy and ADR-target association

Panel A is the reproducible DOT-SafeNet model architecture diagram. It shows the 389-feature drug encoder, 18-SOC embedding encoder, absolute-difference pair representation, SOC-specific task head, soft pseudo-dose/monotone training objective, CT-ADE task-head fine-tuning and five-fold ensemble.

Panel B shows the 194-target ADR-association matrix grouped by M1-M9 and U. Direct evidence is dark blue and secondary evidence is translucent orange. No target or SOC label is highlighted in red. The matrix is descriptive evidence organization and is not presented as model attribution.

Render the generated panels with `python run.py`. The final editable composition is slide 4 of `paper_figures/figure.pptx`.

Outputs:

- `outputs/supplementary_figure_7a_dotsafenet_architecture.png/.pdf/.svg`

- `outputs/supplementary_figure_7b_adr_target_association.png/.pdf/.svg`
- `outputs/supplementary_figure_7b_adr_target_legend.png/.pdf/.svg`

The additional PPB/Cmax results are the final Supplementary Figure 7 and are stored in `../supplementary_figure_7/`.
