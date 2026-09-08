# GC-Ridge

**Generator-Centered Ridge Regression for AI-Generated Image Quality Assessment**

GC-Ridge trains one linear quality predictor on frozen CLIP image features. Its objective increases the weight of residual variation within each training generator while retaining pooled score-level fitting. This targets a practical distinction in AI-generated image quality assessment: ranking images within a generator and ranking images pooled across generators can favor different predictors.

The model uses generator labels during training and validation. Inference needs only image features and a fitted head. Fitting uses a closed-form, regularized least-squares solver on CPU, with an unpenalized intercept and training-only target normalization.

```text
Local images -> frozen CLIP -> normalized features -> one quality score
                                  |
                  training: pooled + generator-centered residual loss
                  selection: validation-only pooled-retention rule
```

Associated manuscript: *Generator-Centered Ridge Regression for AI-Generated Image Quality Assessment*, Le Gao and Qingbing Sang, manuscript v23. Citation metadata is in `CITATION.cff`.

## Quick start

Use Python 3.13. From this directory:

```bash
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
```

The 19 tests run without datasets, GPUs or checkpoints. To reproduce the paper tables from the supplied predictions, first obtain the two official annotation files described in [DATA.md](DATA.md):

```bash
python prepare_data.py --agiqa-csv path/to/data.csv --aigciqa-mos path/to/moz1.txt --out local_data
python reproduce.py results --data local_data --out reproduced/results
```

The result command evaluates 257,542 prediction rows, rebuilds all four paper tables and the numeric points for Figure 2, and checks the recorded validation selections. Human annotations are joined locally by sample ID; they are not included in this repository.

## Fit the method

Follow [DATA.md](DATA.md) to extract CLIP RN50 or ViT-B/32 features from your local images, then run:

```bash
python reproduce.py tune --dataset AGIQA-3K-full --backbone RN50 --seed 42 --data local_data --features local_data/features --out reproduced/primary
```

This searches 63 GC-Ridge candidates and 63 matched Ridge candidates on validation data. The selection rule retains candidates whose pooled validation SRCC is within 0.01 of the fixed Ridge-10 reference, then maximizes the mean of pooled and within-generator SRCC. Test targets do not enter the search function.

For the 25 core configurations, use `tune --all`. The command also refits the archived independently tuned Equal-generator baseline and fixed-parameter component controls, totaling 175 selected fits. The Equal-generator search is verified from its validation trace; it is not rerun by this command.

Fit outputs include prediction heads in NPZ format and comparisons against archived predictions. For example:

```bash
python reproduce.py predict --model reproduced/primary/agiqa_full_RN50_42_sample_joint.npz --input-features local_data/features/agiqa_full_RN50.npz --out reproduced/predict
python bootstrap.py --data local_data --out reproduced/bootstrap
```

The predictor outputs raw MOS units and does not request generator identities. Bootstrap recomputes 10,000 prompt-cluster draws for each primary protocol and all 24 reported GC-Ridge paired interval entries.

## Reported results

Primary RN50 results, seed 42. P is pooled SRCC and W is the correlation of concatenated, within-generator centered ranks; higher is better for both.

| Protocol | Ridge-63 P | Ridge-63 W | GC-Ridge P | GC-Ridge W |
|---|---:|---:|---:|---:|
| AGIQA-3K full | 0.8446 | 0.6054 | **0.8471** | **0.6142** |
| AGIQA-S subset | **0.8655** | 0.6089 | 0.8653 | **0.6380** |
| AIGCIQA2023 | **0.7990** | **0.4488** | 0.7988 | 0.4475 |

Across the 25 core dataset/backbone/split configurations, GC-Ridge improves W over Ridge-63 in 24. These overlapping configurations measure sensitivity, not 25 independent replications. AGIQA-S is the fixed 2,190-image subset of the 2,982-image full protocol. Full tables, contextual comparisons, component controls and generalization outputs remain in `evidence/`.

## Reproducibility coverage

| Material | What can be rerun |
|---|---|
| 15 fixed split CSVs | Sample alignment, local target fingerprints and prompt separation |
| 551 stored experimental runs | Metrics, 108 table values and 25 Figure 2 points, from local annotations |
| Core GC-Ridge and Ridge-63 | Complete validation searches across 25 configurations |
| Selected core models and components | 175 fits from local features |
| Primary paired intervals | Three sets of 10,000 bootstrap draws |
| Local image encoding | Optional OpenAI CLIP extraction script; validation coverage is recorded separately |

See [PROTOCOL.md](PROTOCOL.md) for the objective, grids, metrics and study scope, and `VERIFICATION.json` for the actual executed checks. The new RN50 extraction smoke test runs, but does not meet the archived-feature regression threshold; details are in [DATA.md](DATA.md). Core numerical agreement was verified separately with the authors' archived local features. The fitter reports real differences and supports `--strict-archive` for exact archived-input regression. Neural baseline training, Q-Align inference and LOGO/transfer retraining are outside this version's training coverage; their supplied predictions can still be evaluated. Confidence intervals condition on the fitted models and primary split and exclude uncertainty from method development on these benchmarks.

## Files and names

- `gc_ridge.py`: training moments, solver, evaluation metrics and validation selection.
- `prepare_data.py`, `data_io.py`: local annotation import, alignment and fingerprints.
- `extract_features.py`: optional image-to-feature preparation.
- `reproduce.py`, `bootstrap.py`: fitting, evaluation, prediction and interval reconstruction.
- `configs/`, `splits/`, `evidence/`: experiment settings, fixed assignments and experimental outputs.
- `tests/`: mathematical and input-interface checks.

Historical output IDs remain traceable: `sample_joint` means GC-Ridge; `ridge_matched` means Ridge-63; `cc_joint` means independently tuned Equal-generator. In fixed component files, `equal_generator` instead uses the GC-selected parameters. The historical `uncentered_joint` control uses independently tuned equal-generator auxiliary weighting and is different from the unit-image-weight `uncentered` component in Table 4. See the protocol before combining rows from different experimental regimes.

## License and data sources

The project-owned implementation is distributed under the [MIT License](LICENSE), copyright 2026 Le Gao. Third-party notices and official dataset links are in [THIRD_PARTY.md](THIRD_PARTY.md). Obtain dataset annotations, images and pretrained encoders under their applicable terms. This repository contains no human MOS, prompt texts, original images, image features or pretrained/fitted model weights. Keep your generated local data and outputs out of commits.
