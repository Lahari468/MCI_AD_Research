# Research Objectives

## Primary Research Question
Does baseline volumetric structural 3D T1-weighted MRI provide statistically significant incremental predictive power when added to clinical cognitive testing and subject-space FreeSurfer morphometry for predicting 24-month MCI-to-AD progression?

## Key Scientific Hypotheses
1. **Hypothesis 1 (Morphometric Value)**: Incorporating regional morphometric volumes from automated segmentation pipelines (FreeSurfer) will statistically improve prediction accuracy over clinical cognitive scores alone.
2. **Hypothesis 2 (Incremental Deep MRI Value)**: Directly encoding preprocessed 3D MRI volumes using a 3D CNN and cross-modal attention will extract residual morphological or textural features not captured by tabular summary statistics.
3. **Hypothesis 3 (Subcortical Contrast Value)**: Regional intensity contrast and tissue heterogeneity extracted from vulnerable subcortical structures (hippocampus, amygdala, lateral ventricles) will capture tissue-level alterations complementary to macro-structural volume loss.

## Scientific Evaluation Principles
- **Reproducibility**: Complete deterministic reproduction across random seeds.
- **Leakage Prevention**: Strict separation of training and validation folds; fitting all normalizations, imputations, and hyperparameter selections solely on inner folds.
- **Fair Paired Comparisons**: Evaluating competing models on identical participants, identical fold partitions, and identical bootstrap resamples.
- **Statistical Rigor**: Utilizing participant-level bootstrap confidence intervals and paired difference testing rather than naive point estimate comparisons.
