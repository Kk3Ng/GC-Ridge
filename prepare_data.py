"""从使用者自行取得的官方文件生成本地标注，不下载或公开上传数据。"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from data_io import split_ids, validate_labels
from reproduce import output_directory


def agiqa_labels(path):
    source = pd.read_csv(path, float_precision='round_trip')
    if not {'name','mos_quality'}.issubset(source.columns) or source.name.duplicated().any():
        raise ValueError('Expected official AGIQA-3K data.csv with unique name and mos_quality columns.')
    rows = split_ids('AGIQA-3K-full')
    indexed = source.set_index('name').mos_quality
    if not set(rows.official_filename).issubset(indexed.index):
        raise ValueError('Official AGIQA-3K metadata does not cover all 2,982 study images.')
    result = pd.DataFrame({'sample_id':rows.sample_id,'mos':indexed.loc[rows.official_filename].to_numpy()})
    return validate_labels(result, 'AGIQA-3K-full')


def aigciqa_labels(path):
    quality = np.loadtxt(path)
    if quality.shape not in {(2400,), (2400,1)} or not np.isfinite(quality).all():
        raise ValueError('Expected 2,400 finite values in official quality file moz1.txt, in its original order.')
    rows = split_ids('AIGCIQA2023')
    indices = rows.sample_id.str.rsplit('_',n=1).str[-1].astype(int).to_numpy()
    if set(indices) != set(range(2400)):
        raise ValueError('Packaged sample indices are not a one-to-one mapping to 0..2399.')
    # 原研究将 0--100 的质量 MOS 除以 100，并在清单中保留 12 位小数。
    values = np.asarray([float(format(v/100.,'.12f')) for v in quality.reshape(-1)[indices]])
    result = pd.DataFrame({'sample_id':rows.sample_id,'mos':values})
    return validate_labels(result, 'AIGCIQA2023')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agiqa-csv', type=Path)
    parser.add_argument('--aigciqa-mos', type=Path)
    parser.add_argument('--out', type=Path, default=Path('local_data'))
    args = parser.parse_args()
    if not args.agiqa_csv and not args.aigciqa_mos:
        parser.error('Supply --agiqa-csv, --aigciqa-mos, or both.')
    prepared = {}
    if args.agiqa_csv:
        prepared['agiqa3k_labels.csv'] = agiqa_labels(args.agiqa_csv)
    if args.aigciqa_mos:
        prepared['aigciqa2023_labels.csv'] = aigciqa_labels(args.aigciqa_mos)
    out = output_directory(args.out)
    for name, frame in prepared.items():
        frame.to_csv(out/name,index=False)
    print(json.dumps({'prepared':{name:len(frame) for name,frame in prepared.items()},'target_fingerprints_match':True}))


if __name__ == '__main__':
    main()
