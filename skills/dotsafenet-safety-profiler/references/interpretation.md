# Interpretation

## SOC results

`probability_mean` is the five-member DOT-SafeNet score. `background_percentile` is the score's empirical position among original drug–SOC samples from the model development set. It controls for the different score distributions of the 18 SOC heads. Neither value estimates clinical event incidence.

## Target activity and margin

OT-ProfileNet pAC50 is converted to predicted AC50 in micromolar. The target margin is:

`log10(predicted AC50 in micromolar) - log10(predicted free Cmax in micromolar)`.

Lower values indicate that predicted free exposure approaches or exceeds the predicted target activity concentration. The margin is a model-derived comparison and does not include tissue exposure, active metabolites or time-dependent pharmacology.

## Target attribution

`delta_probability` is the original SOC score minus the score after replacing one target feature with its training reference and recalculating the corresponding target–exposure product. Positive values identify target inputs that increase the selected SOC score for that molecule and regimen. Fold standard deviation and positive-fold fraction report ensemble consistency.

Evidence levels:

- `direct`: the target has direct SOC association evidence in the deposited ADR–target table.
- `secondary`: the target was added through the predefined secondary evidence procedure.
- `none`: the target is not currently linked to that SOC in the deposited table.

Attribution describes model dependence on an input feature. It is not a binding assay or causal-mechanism experiment.
