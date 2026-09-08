"""本地标注和特征接口；公开仓库仅提供样本划分及内容指纹。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SLUGS = json.loads((ROOT/'configs/experiment.json').read_text(encoding='utf-8'))['dataset_slugs']


def split_ids(dataset, seed=42):
    return pd.read_csv(ROOT/'splits'/f'{SLUGS[dataset]}_seed{seed}.csv')


def target_fingerprint(frame):
    text = ''.join(f'{r.sample_id}\t{r.mos:.12f}\n' for r in frame.sort_values('sample_id').itertuples())
    return hashlib.sha256(text.encode()).hexdigest()


def validate_labels(frame, dataset):
    required = {'sample_id','mos'}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError('Labels require non-missing sample_id and mos columns.')
    if frame.sample_id.duplicated().any() or not np.isfinite(frame.mos.to_numpy(dtype=float)).all():
        raise ValueError('Label IDs must be unique and MOS values must be finite.')
    wanted = split_ids(dataset).sample_id
    if not set(wanted).issubset(set(frame.sample_id)):
        raise ValueError(f'Missing labels for {dataset}. Use the official quality target.')
    selected = frame.set_index('sample_id').loc[wanted].reset_index()[['sample_id','mos']]
    records = json.loads((ROOT/'configs/target_fingerprints.json').read_text())
    if target_fingerprint(selected) != records[dataset]['sha256_at_12_decimals']:
        raise ValueError(f'{dataset}: target fingerprint differs from the study. Check the quality field, scale and index order.')
    return selected


def read_labels(directory, dataset):
    name = 'aigciqa2023_labels.csv' if dataset == 'AIGCIQA2023' else 'agiqa3k_labels.csv'
    path = Path(directory)/name
    if not path.is_file():
        raise FileNotFoundError(f'Missing local labels: {name}. Run prepare_data.py with official metadata; see DATA.md.')
    frame = pd.read_csv(path, float_precision='round_trip')
    return validate_labels(frame, dataset).set_index('sample_id').mos


def feature_fingerprint(ids, features):
    ids = np.asarray(ids).astype(str)
    order = np.argsort(ids)
    x = np.ascontiguousarray(np.asarray(features)[order], dtype='<f4')
    return hashlib.sha256(('\n'.join(ids[order])+'\n').encode()+x.tobytes()).hexdigest()


def read_features(directory, dataset, backbone):
    key = f'{SLUGS[dataset]}_{backbone}'
    records = json.loads((ROOT/'configs/feature_fingerprints.json').read_text())
    if key not in records:
        raise ValueError('This dataset/backbone combination is outside the reported configurations.')
    path = Path(directory)/f'{key}.npz'
    if not path.is_file():
        raise FileNotFoundError(f'Missing local features: {path.name}. See extract_features.py and DATA.md.')
    with np.load(path, allow_pickle=False) as data:
        ids, x = data['sample_ids'], data['features']
    expected = records[key]
    wanted = split_ids(dataset).sample_id.to_numpy()
    if ids.ndim != 1 or ids.dtype.kind not in 'US' or len(set(ids)) != len(ids):
        raise ValueError('Feature sample_ids must be a unique one-dimensional string array.')
    if x.ndim != 2 or x.shape != (len(ids),expected['dimensions']) or not np.isfinite(x).all():
        raise ValueError('Invalid feature dimensions or values.')
    if set(ids) != set(wanted):
        raise ValueError('Feature IDs must exactly match the selected dataset protocol.')
    if not np.allclose(np.linalg.norm(x,axis=1),1,atol=3e-7,rtol=0):
        raise ValueError('Features must use the L2-normalized frozen CLIP representation.')
    matches = feature_fingerprint(ids,x) == expected['sha256']
    return dict(zip(ids.astype(str),x.astype(np.float64))), matches
