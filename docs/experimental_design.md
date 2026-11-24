# Experimental Design

## Validation Strategy: Multi-Repeat K-Fold Cross-Validation

To prevent fold-partition bias and obtain stable performance estimates, a $3\text{-Repeat} \times 5\text{-Fold}$ stratified cross-validation protocol was employed:

- **Total outer folds per model:** 15.
- **Stratification:** Maintained conversion outcome ratio across all folds.
- **Partition Stability:** Exactly identical fold splits were evaluated by all competing models.

## Paired Bootstrap Inference ($B = 2,000$, Seed 42)

Statistical comparisons between competing models (e.g., Model B vs. Model A) were conducted using paired participant-level bootstrap resampling.

**Step 1: Bootstrap Sampling**

In iteration $b$, sample $N$ participants with replacement using index vector $\mathbf{idx}^{(b)}$.

**Step 2: Metric Difference**

Compute the AUROC difference on the paired bootstrap sample:

$$
\Delta_{\mathrm{AUROC}}^{(b)} = \mathrm{AUROC}(\mathbf{y}[\mathbf{idx}^{(b)}], \bar{\mathbf{p}}_{\mathrm{test}}[\mathbf{idx}^{(b)}]) - \mathrm{AUROC}(\mathbf{y}[\mathbf{idx}^{(b)}], \bar{\mathbf{p}}_{\mathrm{ref}}[\mathbf{idx}^{(b)}])
$$

**Step 3: Confidence Interval**

Construct the 95% empirical percentile confidence interval using the 2.5th and 97.5th percentiles of the bootstrap distribution.

**Step 4: Empirical Bootstrap-tail Proportion**

Calculate the two-sided empirical bootstrap-tail proportion:

$$
p = 2 \times \min\left( \frac{1}{B}\sum_{b=1}^{B}\mathbb{I}(\Delta^{(b)} \leq 0), \frac{1}{B}\sum_{b=1}^{B}\mathbb{I}(\Delta^{(b)} \geq 0) \right)
$$
