# Prepare local inputs

This repository supplies code, fixed split assignments, model predictions and aggregate results. Obtain the original metadata and images from the dataset authors. Human MOS, image files, image features and trained prediction heads are not bundled in this release.

## 1. Obtain official quality annotations

- **AGIQA-3K:** obtain `data.csv` from the [official repository](https://github.com/lcysyzxdxc/AGIQA-3k-Database). The importer uses `name` and `mos_quality`, including all 2,982 rows. The same file covers the fixed 2,190-image subset.
- **AIGCIQA2023:** obtain the quality score file `moz1.txt` from the database download linked in the [official repository](https://github.com/wangjiarui153/AIGCIQA2023). Its 2,400 scores must remain in their original order. The official README calls the quality dimension `mosz1`; authenticity and correspondence are different targets.

```bash
python prepare_data.py --agiqa-csv path/to/data.csv --aigciqa-mos path/to/moz1.txt --out local_data
python reproduce.py results --data local_data --out reproduced/results
```

Each input option can also be supplied separately. All paper tables require both datasets. The importer joins AGIQA by official filename. For AIGCIQA it uses the zero-based image index encoded in the supplied sample ID, divides the original quality score by 100, and uses `float(format(value, '.12f'))` to reproduce the frozen manifest's decimal representation. The complete target fingerprints must match before imported labels are saved. Do not reorder the text file, use another MOS dimension, or rescale scores again.

This produces `local_data/agiqa3k_labels.csv` and `local_data/aigciqa2023_labels.csv`. These contain human annotations and are ignored by Git. The public prediction files contain our `y_pred` values but no `y_true`; `reproduce.py` joins the local targets by sample ID before evaluating them. No artificial targets are substituted when annotations are missing.

## 2. Extract frozen CLIP features

Skip this step when only recomputing the supplied predictions, tables, Figure 2 points or bootstrap intervals. Model fitting requires features from the original images.

Use a separate environment if your machine has a different PyTorch/CUDA installation. `requirements-features.txt` specifies the optional encoder packages and pins the official OpenAI CLIP source to one commit. Follow the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) for a compatible CPU or CUDA build, then install:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -r requirements-features.txt
```

The extractor uses the preprocessing returned by [OpenAI CLIP](https://github.com/openai/CLIP), converts images to RGB, calls `encode_image`, and L2-normalizes float32 features. Named models download their official checkpoint when absent. Alternatively, supply `--checkpoint path/to/RN50.pt` or the official ViT-B/32 checkpoint; its SHA-256 is checked before loading. Model downloads and generated features stay under the chosen local output directory.

For AGIQA, point `--images` to a directory containing the official `.jpg` filenames. Subdirectories are supported, but duplicate filename matches fail explicitly:

```bash
python extract_features.py --dataset AGIQA-3K-full --backbone RN50 --images path/to/agiqa_images
python extract_features.py --dataset AGIQA-3K --backbone ViT-B-32 --images path/to/agiqa_images
```

The first command also writes the RN50 subset features by selecting the same images from the full extraction. The full protocol was evaluated with RN50 only.

For the AIGCIQA `allimg/0.png` through `allimg/2399.png` layout:

```bash
python extract_features.py --dataset AIGCIQA2023 --backbone RN50 --images path/to/allimg
python extract_features.py --dataset AIGCIQA2023 --backbone ViT-B-32 --images path/to/allimg
```

If your official download is organized by generator folders containing names such as `062-2.png`, add `--layout generator` and point `--images` to the parent of those folders. Each path must have an ancestor folder matching the generator name, ignoring case and punctuation. Repeated filenames across different generators are not interchangeable. `--device cpu` selects CPU explicitly; the default uses CUDA when available. The default batch size is 32.

Five local NPZ files are produced: `agiqa_full_RN50`, `agiqa_subset_RN50`, `agiqa_subset_ViT-B-32`, `aigciqa2023_RN50` and `aigciqa2023_ViT-B-32`. Each contains only `sample_ids` and `features`.

## 3. Fit and compare

```bash
python reproduce.py fit --all --data local_data --features local_data/features --out reproduced/fit
python reproduce.py tune --all --data local_data --features local_data/features --out reproduced/tune
python bootstrap.py --data local_data --out reproduced/bootstrap
```

New GPU/CPU feature extraction can differ numerically from the archived extraction. Each extraction report records the feature fingerprint. Fitting always reports the actual prediction differences; tuning records its selected parameters and checks them against the archive. The `passed` field in `fit_audit.json` is true only if predictions and, when searched, selected parameters match the documented tolerances. Successful program execution alone is not a claim of numerical agreement.

Add `--strict-archive` to `fit` or `tune` to require the exact archived feature fingerprints and fail on a parameter or prediction discrepancy. This mode is useful with a verified copy of the original features; it is not required to evaluate newly extracted features. No result or parameter is silently replaced with an archived value to make a comparison pass.

The public archive's feature fingerprints let users compare inputs, but do not contain the original features. In a new RN50 extraction check on seven AGIQA example images, the extractor ran successfully but the maximum difference from the archived features was about `4.54e-4`, above the `1e-5` regression threshold. Exact archive agreement was therefore **not established** for this extraction environment. The cause has not been isolated; changes in encoding precision alone did not resolve it. The archive-based core fits pass separately and must not be represented as a complete new image-to-result reproduction.

The validation report distinguishes tests with the authors' archived local features from tests of new extraction. Neural baseline training, Q-Align inference and LOGO/transfer retraining are outside the executable training coverage of this version; their supplied predictions can be evaluated from local annotations.
