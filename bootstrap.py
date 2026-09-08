"""用逐图预测重新生成主划分的配对 prompt-cluster bootstrap。"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import reproduce
from scipy.stats import rankdata
from reproduce import ROOT, read, split_data, write_json, output_directory


def pooled_within(y, predictions, groups):
    yr, pr = rankdata(y,method='average'),rankdata(predictions,axis=0,method='average')
    yr, pr = yr-yr.mean(), pr-pr.mean(axis=0)
    yc, pc = np.empty_like(y), np.empty_like(predictions)
    for group in np.unique(groups):
        ix = groups==group
        center = (int(ix.sum())+1)/2
        yc[ix] = rankdata(y[ix],method='average')-center
        pc[ix] = rankdata(predictions[ix],axis=0,method='average')-center
    with np.errstate(invalid='ignore',divide='ignore'):
        p = (yr@pr)/(np.linalg.norm(yr)*np.linalg.norm(pr,axis=0))
        w = (yc@pc)/(np.linalg.norm(yc)*np.linalg.norm(pc,axis=0))
    return np.stack((p,w),axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('reproduced_bootstrap'))
    parser.add_argument('--data',type=Path,default=ROOT/'local_data')
    args = parser.parse_args()
    reproduce.DATA_DIR = args.data
    args.out = output_directory(args.out)
    all_predictions = read('predictions.csv.gz')
    rows, checks = [], []
    for dataset_index,(dataset,slug) in enumerate([('AGIQA-3K','agiqa3k'),('AIGCIQA2023','aigciqa2023'),('AGIQA-3K-full','agiqa3k_full')]):
        with np.load(ROOT/'evidence'/f'{slug}__rn50__split42_draws.npz',allow_pickle=False) as z:
            names, original, original_point = list(z['names'].astype(str)),z['draws'],z['point']
        split = split_data(dataset,42)
        test = split[split.split=='test']
        frame = all_predictions[(all_predictions.dataset==dataset)&(all_predictions.split_seed==42)&
                                (all_predictions.backbone=='RN50')&(all_predictions.regime=='core')]
        predictions = np.column_stack([frame[frame.method==m].set_index('sample_id').loc[test.sample_id,'y_pred'] for m in names])
        y, prompts, groups = test.mos.to_numpy(),test.prompt_id.to_numpy(),test.client_id.to_numpy()
        clusters = [np.flatnonzero(prompts==p) for p in np.unique(prompts)]
        rng = np.random.default_rng(2026082600+dataset_index*100)
        draws = np.empty_like(original)
        for i in range(len(draws)):
            ix = np.concatenate([clusters[j] for j in rng.integers(len(clusters),size=len(clusters))])
            draws[i] = pooled_within(y[ix],predictions[ix],groups[ix])
            if (i+1)%2500==0: print(f'{dataset}: {i+1}/{len(draws)} resamples',flush=True)
        point = pooled_within(y,predictions,groups)
        if not np.allclose(draws,original,atol=1e-12,rtol=0,equal_nan=True):
            raise AssertionError(f'{dataset}: regenerated draws differ from archive')
        if not np.allclose(point,original_point,atol=1e-12,rtol=0,equal_nan=True):
            raise AssertionError(f'{dataset}: point estimates differ from archive')
        gc_index = names.index('sample_joint')
        for control in ['ridge_matched','ridge_tuned15','cc_joint','uncentered_joint']:
            j = names.index(control)
            for k, metric in enumerate(['P','W']):
                delta = draws[:,gc_index,k]-draws[:,j,k]
                if not np.isfinite(delta).all(): raise AssertionError('Undefined draw in reported comparison')
                lo,hi = np.quantile(delta,[.025,.975])
                rows.append({'dataset':dataset,'control':control,'metric':metric,
                             'effect':point[gc_index,k]-point[j,k],'lo':lo,'hi':hi,'resamples':len(draws)})
        finite = np.isfinite(draws)&np.isfinite(original)
        checks.append({'dataset':dataset,'resamples':len(draws),'methods':len(names),
                       'maximum_draw_error':float(np.max(np.abs(draws[finite]-original[finite])))})
    result = pd.DataFrame(rows)
    keys = ['dataset','control','metric']
    columns = ['effect','lo','hi','resamples']
    ref = read('paired_effects.csv').set_index(keys)
    actual = result.set_index(keys).loc[ref.index]
    if not np.allclose(actual[columns],ref[columns],atol=1e-12,rtol=0):
        raise AssertionError('Paper confidence interval mismatch')
    result.to_csv(args.out/'paired_effects.csv',index=False)
    summary = {'passed':True,'comparisons_checked':len(rows),'checks':checks,
               'scope':'Conditional on fixed fitted predictions; method-development uncertainty is excluded.'}
    write_json(args.out/'bootstrap_audit.json',summary)
    print('All 24 reported GC-Ridge paired interval entries reproduced.',flush=True)


if __name__=='__main__':
    main()
