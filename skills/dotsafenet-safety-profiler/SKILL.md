---
name: dotsafenet-safety-profiler
description: Predict exposure-aware safety profiles for new small molecules with the published DOT-SafeNet framework. Use when a user supplies a SMILES string and oral daily dose, requests 194-target activity, PPB, total and free Cmax, 18-SOC ADR scores, development-background percentiles, target safety margins, target-replacement attribution, or reproduction of the deposited paper examples.
---

# DOT-SafeNet Safety Profiler

Use the release entrypoint `scripts/profile_molecule.py`. Keep the target order, SOC order, units and transformations fixed.

## Select an operation

- For a new molecule, read [references/input_output.md](references/input_output.md), then run `analyze` with the deposited checkpoint and inference-asset roots.
- For a paper example or installation check, run `replay`. This uses the four deposited clinical cases and does not require model checkpoints.
- For environment preparation or missing-file errors, read [references/installation.md](references/installation.md).
- For scientific interpretation, read [references/interpretation.md](references/interpretation.md).

## Required model contract

1. Standardize the SMILES and calculate molecular weight with RDKit.
2. Use the second OT-ProfileNet regression output as pAC50 for each of 194 targets.
3. Predict free fraction with the five-member PlasmaBindNet-Fu ensemble.
4. Predict total Cmax from daily dose with the five-member DoseExpoNet ensemble.
5. Calculate free Cmax in micromolar and take `log10`.
6. Construct 389 DOT-SafeNet inputs in the deposited target order: one log10 free-Cmax value, 194 values of `6 - pAC50`, and 194 target–exposure products.
7. Average the five clinically fine-tuned DOT-SafeNet members for each of 18 SOC tasks.
8. Report raw scores together with within-SOC empirical background percentiles. Do not describe either value as ADR incidence.
9. For target replacement, replace the selected target feature with its training-set reference and recalculate the matching target–exposure product.

Run `validate` before full inference. Stop if checkpoints, the seven OT-ProfileNet protein-family resources, the 194-target order or the 18-SOC order fail validation.

## Outputs

Return the generated `report.html` and `report.json`, followed by the principal CSV tables. Separate model predictions from direct and secondary ADR–target evidence. State the supplied dose and route in the result.

The report covers computational off-target, exposure and SOC-level ADR assessment. Do not describe it as a clinical safety determination.
