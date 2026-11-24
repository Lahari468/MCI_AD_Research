# Model Architecture

## 1.Multimodal Cross-Attention Network
The architecture fuses tabular clinical scores, FreeSurfer regional morphometry, and 3D volumetric MRI tensors:

```
[Clinical 10] ──> MLPEnc (10 -> 128 -> 256 -> 32) ──┐
                                                    │
[FS29 Morph]  ──> MLPEnc (29 -> 128 -> 256 -> 32) ──┼──> Cross-Attention Fusion
                                                    │    (Multi-Head Self-Attention,
[3D MRI Vol]  ──> 4-Stage 3D CNN (1 -> 16..96 -> 32)─┘     h=2 heads, d=32)
                                                                 │
                                                                 ▼
                                                        [Missingness Conditioning]
                                                                 │
                                                                 ▼
                                                         MLP Head ──> Logit
```

### Parameter Budgets
- **Model A (MRI Excluded)**: 159,105 parameters.
- **Model B (MRI Included)**: 509,921 parameters.

## 2.L2-Regularized Logistic Regression
- **Primary Linear Model**: Evaluates feature spaces (Clinical, Clinical + FS29, Clinical + FS29 + 16 Subcortical ROIs).
- **Hyperparameter Optimization**: L2 penalty parameter $C \in \{0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0\}$ chosen by maximizing AUROC on the inner 15% validation split.
- **Operating Threshold Selection**: Classification threshold $t \in [0.10, 0.90]$ chosen by maximizing Youden's $J = \text{Sensitivity} + \text{Specificity} - 1$ on the inner validation split.
- **Secondary Non-linear Model**: HistGradientBoostingClassifier evaluated to detect potential non-linear feature interactions.
