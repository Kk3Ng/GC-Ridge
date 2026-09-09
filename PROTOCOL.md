# Study and execution protocol

## Objective

For a frozen feature vector z and a MOS target standardized using the pooled training mean and population standard deviation, fit one shared slope w and intercept b. With residual r = w'z + b − y and generator k:

```
L_within = sum_k sum_(i in k) (r_i - mean_k(r))^2
L_between = sum_k n_k mean_k(r)^2
L = sum_i r_i^2 + lambda * L_within + alpha * ||w||^2
  = (1 + lambda) * L_within + L_between + alpha * ||w||^2
```

The intercept is unpenalized. GC-Ridge gives every image unit weight in the centered term. The Equal-generator alternative multiplies each generator's centered term by N/(K n_k). Prediction uses one common head and restores the training MOS scale; no generator identity is needed at inference. Additive moments permit aggregation, but these experiments do not establish formal privacy guarantees or deployment in a real federated network.

The portable implementation receives only training arrays when computing moments and target scaling. It casts the frozen, already normalized float32 feature values to float64 without renormalizing them. Cholesky fitting is on CPU.

## Data partitions

| Protocol | Images | Primary train/val/test images | Primary train/val/test prompts |
|---|---:|---|---|
| AGIQA-3K full | 2,982 | 2,016 / 248 / 718 | 203 / 25 / 72 |
| AGIQA-S subset | 2,190 | 1,487 / 174 / 529 | 203 / 25 / 72 |
| AIGCIQA2023 | 2,400 | 1,680 / 360 / 360 | 70 / 15 / 15 |

All protocols have six generators. The 15 included CSVs define the exact frozen assignments and row IDs for seeds 42–46. Seed 42 is primary. Train, validation and test prompts are disjoint within each core split. Splits 43–46 reuse images and assess sensitivity; they are not independent replications. The AGIQA subset and full protocol overlap.

In archived LOGO evaluation, one generator is excluded from training and validation and all its images are tested. Prompts may still occur in other training generators, so this is not simultaneously an unseen-prompt test. Cross-dataset transfer selects parameters and target scaling on the source dataset's training/validation sets and evaluates the target primary test set. Its P and W can move in opposite directions.

## Selection

The exact grids are in `configs/experiment.json`. GC-Ridge uses seven alpha values times nine lambda values, including zero lambda. Ridge-63 includes the 58 distinct alpha/(1+lambda) values and five successive geometric midpoints in the largest log gaps; ties choose the lower interval. Ridge-15 is a separate historical grid: the union of the seven alpha values and 10/(1+lambda).

Let P0 be the validation pooled SRCC for Ridge alpha=10. Retain candidates with P >= P0−0.01, then maximize (P+W)/2. Ties prefer higher W, higher P, lower lambda, then higher alpha. The per-configuration hyperparameter search uses validation targets, not test targets. `results` checks 148 marked choices over the 37 archived core/LOGO search configurations and four historical method families; `tune` reruns the GC/Ridge-63 searches for the 25 core configurations.

The method family and research framing evolved while these benchmarks were being analyzed. Per-configuration validation-only selection does not make the overall study an untouched-test, prospectively registered confirmation. This scope is retained in the manuscript's confidence-interval statement. No new independent test dataset is claimed by this release.

## Metrics and inference

P is pooled Spearman rank correlation. To compute W, rank targets and predictions separately within each generator with average ties, subtract each group's mean rank, then correlate the concatenated centered ranks. W is different from an unweighted mean of per-generator SRCCs. Without ties, its group contributions scale as n(n²−1). Macro SRCC reports the unweighted per-generator mean.

Constant predictions have undefined correlations. The evaluator preserves N/A rather than substituting zero or omitting an undefined group from the macro average. It checks labels and IDs against the fixed split before evaluating predictions. The contextual Generator-specific Ridge comparison uses the true test generator and is labeled separately from image-only methods.

Paired primary CIs use 10,000 prompt-cluster bootstrap draws with reranking of sampled, potentially duplicated images. Their seed is 2026082600 plus 100 times the dataset index in [AGIQA-S, AIGCIQA2023, full AGIQA]. All methods share each resample. The reported bounds are the 2.5th and 97.5th percentiles of paired differences. These CIs condition on fixed fitted models and the primary split and do not include model retraining or method-development uncertainty.

## Numerical reproducibility and scope

The CPU release was checked with Python 3.13.12, NumPy 2.4.4, SciPy 1.17.1 and pandas 3.0.2. Original feature extraction used a separate GPU environment. Fixed split files, target/feature fingerprints and per-image output comparisons are supplied. Human targets are imported from official local annotations; image features are local inputs. Small floating-point differences across BLAS implementations are expected. A prediction difference above 1e-10 is reported as a numerical mismatch; `--strict-archive` additionally fails the command and requires the archived feature fingerprint. Actual checks and their coverage are recorded in `VERIFICATION.json`.

The associated manuscript is v26, by Le Gao, Qingbing Sang, and Yue Fang. Its research text, tables and figure assets are unchanged from v23. The tagged v1.0.0 release and `VERIFICATION.json` retain the original v23 verification record; the current citation metadata is in `CITATION.cff`. The complete author-side evidence package is retained locally; this repository exposes the code, fixed study assignments and experimental outputs while leaving dataset acquisition to the user.
