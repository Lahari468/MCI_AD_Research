# Experimental Design

## Validation Strategy: Multi-Repeat K-Fold Cross-Validation
To prevent fold-partition bias and obtain stable performance estimates, a $3\text{-Repeat} \times 5\text{-Fold}$ stratified cross-validation protocol was employed:
- Total outer folds per model: 15.
- Stratification: Maintained conversion outcome ratio across all folds.
- Partition Stability: Exactly identical fold splits were evaluated by all competing models.

## Participant-Level Probability Aggregation
For each participant $i$, predictions from the 3 independent repeats were averaged:
$$\bar{p}_i = \frac{1}{3} \sum_{r=0}^{2} p_{i, r}$$
This creates a single, stable out-of-fold prediction vector of size $N$ for downstream statistical comparison, eliminating multi-repeat pseudo-replication.

## Paired Bootstrap Inference ($B = 2,000$, Seed 42)
Statistical comparisons between competing models (e.g., Model B vs. Model A) were conducted using paired participant-level bootstrap resampling:
1. In iteration $b$, sample $N$ participants with replacement using index vector $\mathbf{idx}^{(b)}$.
2. Compute the metric difference on the paired resample:
   $$\Delta_{\text{AUROC}}^{(b)} = \text{AUROC}(\mathbf{y}[\mathbf{idx}^{(b)}], \bar{\mathbf{p}}_{\text{test}}[\mathbf{idx}^{(b)}]) - \text{AUROC}(\mathbf{y}[\mathbf{idx}^{(b)}], \bar{\mathbf{p}}_{\text{ref}}[\mathbf{idx}^{(b)}])$$
3. Construct the 95% empirical percentile confidence interval from the $2.5^{\text{th}}$ and $97.5^{\text{th}}$ percentiles.
4. Calculate the two-sided empirical bootstrap-tail proportion:
   $$p = 2 \times \min\left(\frac{1}{B} \sum_{b=1}^{B} \mathbb{I}(\Delta^{(b)} \le 0), \frac{1}{B} \sum_{b=1}^{B} \mathbb{I}(\Delta^{(b)} \ge 0)\right)$$
