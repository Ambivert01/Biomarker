# Model Selection — Bake-off Results & Decision Rationale

**All numbers below come from 5-fold stratified cross-validation on the 85% dev pool.
The 15% held-out test set was not touched to produce this table or make this decision** —
consistent with the brief's "never evaluate on the test set until the final model."

| Rank | Model | CV Macro-F1 (±std) | CV Bal. Acc. | Dementia Recall | Dementia Precision | MCI Recall | Control Recall |
|---|---|---|---|---|---|---|---|
| 1 | MLP | 0.6930 (±0.095) | 0.692 | 0.811 | 0.859 | 0.496 | 0.770 |
| 2 | Extra Trees | 0.6911 (±0.031) | 0.698 | 0.832 | 0.791 | 0.485 | 0.777 |
| 3 | Logistic Regression | 0.6880 (±0.053) | 0.692 | 0.821 | 0.826 | 0.474 | 0.780 |
| 4 | **LightGBM** | **0.6871 (±0.015)** | 0.691 | **0.832** | 0.803 | **0.507** | 0.735 |
| 5 | Random Forest | 0.6792 (±0.016) | 0.685 | 0.827 | 0.818 | 0.451 | 0.777 |
| 6 | SVM | 0.6749 (±0.067) | 0.681 | 0.786 | 0.842 | 0.410 | 0.845 |
| 7 | CatBoost | 0.6691 (±0.009) | 0.674 | 0.796 | 0.813 | 0.440 | 0.787 |
| 8 | XGBoost | 0.6635 (±0.020) | 0.664 | 0.791 | 0.820 | 0.485 | 0.715 |
| 9 | HistGradientBoosting | 0.6598 (±0.014) | 0.664 | 0.827 | 0.806 | 0.478 | 0.687 |

## Why LightGBM was selected as the final model, not the nominal #1 (MLP)

The top four models (MLP, Extra Trees, Logistic Regression, LightGBM) are separated by only
**0.006 macro-F1** — well within fold-to-fold noise for an 889-patient dataset — so picking
"the highest mean" alone would be selecting noise, not signal. Three additional,
pre-specified criteria from the project brief broke the tie:

1. **Reliability / reproducibility is an explicit requirement for this medical AI system.**
   MLP's cross-validation standard deviation (±0.095) is **6–10x higher** than LightGBM's
   (±0.015) or CatBoost's (±0.009) — meaning MLP's apparent #1 ranking is driven by
   fold-to-fold instability (small multilayer perceptrons on n≈600 training rows are known to
   be sensitive to initialization/fold composition), not a dependably better fit. A model
   whose performance swings by ±9.5 points across folds is a poor candidate for a clinical
   tool that needs consistent behavior on new patients.
2. **The brief explicitly prioritizes Dementia recall over marginal overall-accuracy gains.**
   LightGBM and Extra Trees tie for the **highest Dementia recall (0.832)** of all 9 models,
   both comfortably clearing the proposal document's own >80% target.
3. **Tie-break between LightGBM and Extra Trees:** LightGBM has roughly **half the
   variance** of Extra Trees (±0.015 vs ±0.031), slightly better Dementia precision (0.803 vs
   0.791 — fewer false Dementia alarms at the same recall), and materially better **MCI
   recall** (0.507 vs 0.485) — MCI is the clinically hardest class in this dataset (see EDA:
   it sits at the ambiguous CN/Dementia boundary) and every model in this bake-off struggles
   with it, so LightGBM's edge there is a meaningful, not marginal, advantage.

**Decision: LightGBM is the primary production model.** Its tuned hyperparameters (from
Optuna, 25 trials / 5-fold CV) are in `artifacts/optuna_studies/lightgbm_best_params.json`.
Extra Trees is retained as the documented runner-up / sensitivity check in the final report,
since it is a structurally different algorithm (bagged trees vs. boosted trees) that reaches
a similar operating point — useful corroboration that the signal is real and not an artifact
of one particular algorithm family.

This model is now refit on the **full 85% dev pool** (train+val combined, using the same
tuned hyperparameters — no further tuning) and evaluated **exactly once** on the untouched
15% test set in the next script.
