"""从发布候选目录运行结果复算、模型重新拟合或验证集搜索。"""
import os
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '2')
from pathlib import Path
import argparse
import json
import platform
import numpy as np
import pandas as pd
import scipy
import gc_ridge as gc
from data_io import read_labels, read_features

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT/'local_data'
CONFIG = json.loads((ROOT/'configs/experiment.json').read_text(encoding='utf-8'))
SLUGS = CONFIG['dataset_slugs']
SCORES = ['P','W','balanced','macro','worst','PLCC','RMSE']


def read(name, targets=True):
    frame = pd.read_csv(ROOT/'evidence'/name, float_precision='round_trip')
    if targets and name.endswith('.csv.gz'):
        # 公开预测文件不携带人工 MOS；在本地按样本 ID 恢复真实目标。
        frame['y_true'] = np.nan
        for dataset in frame.dataset.unique():
            ix = frame.dataset == dataset
            labels = read_labels(DATA_DIR,dataset)
            frame.loc[ix,'y_true'] = frame.loc[ix,'sample_id'].map(labels).to_numpy()
        if not np.isfinite(frame.y_true.to_numpy()).all():
            raise ValueError('Predictions contain IDs absent from local quality annotations.')
    return frame


def write_json(path, obj):
    # JSON 使用 null 表示未定义结果，不输出非标准 NaN 字面量。
    def clean(v):
        if isinstance(v, dict): return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v, (list,tuple)): return [clean(x) for x in v]
        if isinstance(v, (float,np.floating)): return float(v) if np.isfinite(v) else None
        if isinstance(v, np.integer): return int(v)
        return v
    path.write_text(json.dumps(clean(obj), indent=2, allow_nan=False)+'\n', encoding='utf-8')


def compare_scores(actual, expected, label, columns=SCORES, tolerance=1e-10):
    a = np.asarray([actual[k] for k in columns], dtype=float)
    b = np.asarray([expected[k] for k in columns], dtype=float)
    if not np.allclose(a,b,atol=tolerance,rtol=0,equal_nan=True):
        raise AssertionError(f'{label}: score mismatch {dict(zip(columns,a-b))}')
    finite = np.isfinite(a)&np.isfinite(b)
    return float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.


def split_data(dataset, seed):
    path = ROOT/'splits'/f'{SLUGS[dataset]}_seed{int(seed)}.csv'
    frame = pd.read_csv(path, float_precision='round_trip')
    frame['mos'] = frame.sample_id.map(read_labels(DATA_DIR,dataset))
    required = ['sample_id','prompt_id','client_id','mos','split']
    if any(c not in frame for c in required) or frame[required].isna().any().any():
        raise AssertionError(f'Missing split fields: {path.name}')
    if not np.isfinite(frame.mos.to_numpy(dtype=float)).all():
        raise AssertionError(f'Non-finite split targets: {path.name}')
    if frame.sample_id.duplicated().any(): raise AssertionError(f'Duplicate IDs: {path.name}')
    if set(frame.split.unique()) != {'train','val','test'}: raise AssertionError('Invalid split labels')
    sets = {s:set(frame.loc[frame.split==s,'prompt_id']) for s in ('train','val','test')}
    if any(sets[a]&sets[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise AssertionError(f'Prompt overlap: {path.name}')
    return frame


def output_directory(path):
    """拒绝将运行输出写入随包提供的输入/源码目录。"""
    path = Path(path).resolve()
    protected = [ROOT/'evidence', ROOT/'splits', ROOT/'configs', ROOT/'models', ROOT/'tests', ROOT/'third_party']
    if path == ROOT.resolve() or any(path.is_relative_to(p.resolve()) for p in protected):
        raise ValueError('Use a separate output directory; packaged input/source directories are protected.')
    path.mkdir(parents=True,exist_ok=True)
    return path


def identity(frame, expected, label, group_field='client_id'):
    if frame.sample_id.duplicated().any(): raise AssertionError(f'{label}: duplicate prediction IDs')
    if set(frame.sample_id) != set(expected.sample_id): raise AssertionError(f'{label}: test IDs mismatch')
    ref = expected.set_index('sample_id').loc[frame.sample_id]
    if not np.allclose(frame.y_true, ref.mos, rtol=0, atol=1e-12):
        raise AssertionError(f'{label}: MOS mismatch')
    if not np.array_equal(frame.prompt_id.to_numpy(),ref.prompt_id.to_numpy()):
        raise AssertionError(f'{label}: prompt IDs mismatch')
    if not np.array_equal(frame[group_field].to_numpy(),ref.client_id.to_numpy()):
        raise AssertionError(f'{label}: generator IDs mismatch')


def table_values(core, components, baselines, logo, transfer):
    datasets = ['AGIQA-3K-full','AGIQA-3K','AIGCIQA2023']
    primary = core[(core.split_seed==42)&(core.backbone=='RN50')]
    def pair(frame, ds, method):
        row = frame[(frame.dataset==ds)&(frame.method==method)].iloc[0]
        return [float(row.P), float(row.W)]
    values = {'main':[], 'context':[], 'generalization':[], 'ablation':[]}
    for method in ['ridge_fixed','ridge_tuned15','ridge_matched','cc_joint','sample_joint']:
        values['main'].append(sum([pair(primary,ds,method) for ds in datasets], []))
    for method in ['qalign_raw','pooled_mlp','matched_centralized_sgd','fedavg','fedprox','scaffold','client_aware_ridge','sample_joint']:
        values['context'].append(sum([pair(primary if method=='sample_joint' else baselines,ds,method) for ds in datasets[1:]], []))
    for ds, kind in [('AGIQA-3K','logo'),('AIGCIQA2023','logo'),('AGIQA-3K','transfer'),('AIGCIQA2023','transfer')]:
        row = []
        for method in ['ridge_matched','sample_joint']:
            if kind == 'logo':
                folds = logo[(logo.dataset==ds)&(logo.method==method)]
                if len(folds) != 6: raise AssertionError('Missing LOGO folds')
                row += [float(np.mean(folds.P.to_numpy())), float(np.min(folds.P.to_numpy()))]
            else: row += pair(transfer,ds,method)
        values['generalization'].append(row)
    p = components[(components.split_seed==42)&(components.backbone=='RN50')]
    for method in ['pooled_only','uncentered','within_only','equal_generator','gc']:
        values['ablation'].append(sum([pair(p,ds,method) for ds in datasets], []))
    return values


def results(out):
    splits = {(ds,s):split_data(ds,s) for ds in SLUGS for s in range(42,47)}
    joint = read('predictions.csv.gz')
    reference = pd.concat([read('core_all.csv'),read('logo_folds.csv'),read('transfer.csv')], ignore_index=True)
    expected = reference.set_index('run_id')
    if expected.index.duplicated().any() or set(joint.run_id.unique()) != set(expected.index):
        raise AssertionError('Missing or duplicated archived runs')
    rows, maximum = [], 0.
    for run_id, frame in joint.groupby('run_id', sort=False):
        row = expected.loc[run_id].to_dict()
        split = splits[row['dataset'],int(row['split_seed'])]
        target = (split[split.client_id==row['heldout']] if row['regime']=='logo'
                  else split[split.split=='test'])
        identity(frame,target,run_id)
        score = gc.metrics(frame.y_true.to_numpy(),frame.y_pred.to_numpy(),frame.client_id.to_numpy())
        maximum = max(maximum,compare_scores(score,row,run_id))
        if set(frame.test_identity) != {'none'}: raise AssertionError('Unexpected inference identity')
        rows.append({'run_id':run_id,**row,**score})
    recomputed = pd.DataFrame(rows)
    component_predictions = read('fixed_component_predictions.csv.gz')
    component_reference = read('fixed_component_results.csv').set_index(['dataset','backbone','split_seed','method'])
    components = []
    for key, frame in component_predictions.groupby(['dataset','backbone','split_seed','method'], sort=False):
        split = splits[key[0],int(key[2])]
        identity(frame,split[split.split=='test'],str(key),'generator')
        score = gc.metrics(frame.y_true.to_numpy(),frame.y_pred.to_numpy(),frame.generator.to_numpy())
        row = component_reference.loc[key].to_dict()
        maximum = max(maximum,compare_scores(score,row,str(key)))
        components.append(dict(zip(['dataset','backbone','split_seed','method'],key))|row|score)
    if len(components) != len(component_reference): raise AssertionError('Missing component runs')
    components = pd.DataFrame(components)
    baseline_reference = read('baseline_seed_metrics.csv').astype({'model_seed':str}).set_index(['dataset','method','model_seed'])
    bp = read('baseline_predictions.csv.gz').astype({'model_seed':str})
    baseline_rows = []
    for key, frame in bp.groupby(['dataset','method','model_seed'], sort=False):
        split = splits[key[0],42]
        identity(frame,split[split.split=='test'],str(key))
        if frame.run_id.nunique()!=1: raise AssertionError('Baseline mixes experimental regimes')
        score = gc.metrics(frame.y_true.to_numpy(),frame.y_pred.to_numpy(),frame.client_id.to_numpy())
        maximum = max(maximum,compare_scores(score,baseline_reference.loc[key],str(key)))
        baseline_rows.append(dict(zip(['dataset','method','model_seed'],key))|score)
    if len(baseline_rows) != len(baseline_reference): raise AssertionError('Missing baseline seeds')
    baseline_rows = pd.DataFrame(baseline_rows)
    # 任一模型种子未定义时保留 N/A，避免 pandas 默认跳过它后给出偏乐观均值。
    baseline_means = baseline_rows.groupby(['dataset','method'],as_index=False)[SCORES].agg(lambda x: np.mean(x.to_numpy()))
    core = recomputed[recomputed.regime=='core']
    values = table_values(core,components,baseline_means,recomputed[recomputed.regime=='logo'],recomputed[recomputed.regime=='transfer'])
    provenance = json.loads((ROOT/'evidence/paper_table_provenance.json').read_text(encoding='utf-8'))
    cells = 0
    for table in provenance:
        if not np.allclose(values[table['table']],table['values'],rtol=0,atol=1e-10):
            raise AssertionError('Paper table mismatch: '+table['table'])
        cells += np.asarray(table['values']).size
        pd.DataFrame(values[table['table']], index=table['row_labels']).to_csv(out/f"table_{table['table']}.csv")
    gc_rows = core[core.method=='sample_joint'].set_index(['dataset','backbone','split_seed'])
    ridge_rows = core[core.method=='ridge_matched'].set_index(['dataset','backbone','split_seed'])
    points = gc_rows[['P','W','macro']]-ridge_rows[['P','W','macro']]
    point_reference = read('figure_2_points.csv').set_index(['dataset','backbone','split_seed'])
    if set(points.index) != set(point_reference.index) or not np.allclose(
            points.loc[point_reference.index,['P','W']],point_reference[['P','W']],rtol=0,atol=1e-10):
        raise AssertionError('Figure 2 coordinate mismatch')
    points.rename(columns={'P':'delta_P','W':'delta_W','macro':'delta_macro'}).to_csv(out/'figure2_points.csv')
    # 逐个验证候选表的选择规则，不用测试分数选参数。
    trace = read('hyperparameter_search.csv')
    selections_checked = 0
    for key, frame in trace.groupby('key',sort=False):
        base = frame[(frame.family=='cc_joint')&(frame.alpha==10)&(frame['lambda']==0)].iloc[0]
        floor = base.P-CONFIG['delta']
        for family, candidates in frame.groupby('family'):
            chosen = gc.select(candidates.to_dict('records'),floor)
            marked = candidates[candidates.selected.astype(str).str.lower()=='true']
            if len(marked)!=1: raise AssertionError('Missing or duplicate selection mark')
            marked = marked.iloc[0]
            if chosen['alpha']!=marked.alpha or chosen['lambda']!=marked['lambda']:
                raise AssertionError(f'Validation selection mismatch: {key}/{family}')
            selections_checked += 1
    recomputed.to_csv(out/'joint_metrics.csv',index=False)
    components.to_csv(out/'component_metrics.csv',index=False)
    baseline_rows.to_csv(out/'baseline_metrics.csv',index=False)
    summary = {'passed':True, 'portable_splits':len(splits), 'joint_runs':len(rows),
               'joint_predictions':len(joint),'component_runs':len(components),
               'component_predictions':len(component_predictions),'baseline_runs':len(baseline_rows),
               'baseline_predictions':len(bp),'table_cells':cells,'figure2_points':len(points),
               'W_gains_vs_Ridge63':int((points.W>0).sum()),'validation_selections_checked':selections_checked,
               'maximum_metric_absolute_error':maximum,
               'scope':'Result recomputation; model search/refitting and bootstrap regeneration have separate commands and audits.'}
    write_json(out/'results_audit.json',summary)
    print(json.dumps(summary),flush=True)


def fitting(args, out):
    core = read('core_all.csv')
    jobs = core[core.method=='sample_joint'].copy()
    if not args.all:
        jobs = jobs[(jobs.dataset==args.dataset)&(jobs.backbone==args.backbone)&(jobs.split_seed==args.seed)]
    if jobs.empty: raise ValueError('Configuration is not in the 25 reported core settings.')
    joint_pred = read('predictions.csv.gz',targets=False)
    fixed_pred = read('fixed_component_predictions.csv.gz',targets=False)
    feature_maps, feature_matches, checks = {}, {}, []
    parameter_matches = True
    for row in jobs.to_dict('records'):
        ds, back, seed = row['dataset'], row['backbone'], int(row['split_seed'])
        cache_key = (ds,back)
        if cache_key not in feature_maps:
            feature_maps[cache_key],feature_matches[cache_key] = read_features(args.features,ds,back)
        if args.strict_archive and not feature_matches[cache_key]:
            raise AssertionError('Local feature fingerprint differs from the archived extraction. Omit --strict-archive to measure the differences.')
        split = split_data(ds,seed)
        x = np.vstack([feature_maps[cache_key][s] for s in split.sample_id])
        y, groups = split.mos.to_numpy(), split.client_id.to_numpy()
        train, val, test = [(split.split==s).to_numpy() for s in ('train','val','test')]
        p = gc.prepare(x[train],y[train],groups[train])
        job_key = f'{SLUGS[ds]}_{back.replace("-","_")}_{seed}'
        selected = None
        if args.command=='tune':
            selected, trace = gc.tune(p,x[val],y[val],groups[val],CONFIG['alphas'],CONFIG['lambdas'],CONFIG['delta'])
            write_json(out/f'{job_key}_validation.json',{'selected':selected,'candidates':trace})
        matches = core[(core.dataset==ds)&(core.backbone==back)&(core.split_seed==seed)]
        specs = [(m,'equal_generator' if m=='cc_joint' else 'gc',False)
                 for m in ['sample_joint','ridge_matched','cc_joint']]
        specs += [(m,m,True) for m in ['pooled_only','uncentered','within_only','equal_generator']]
        for method, component, fixed in specs:
            parameters = row if fixed else matches[matches.method==method].iloc[0].to_dict()
            alpha, lam = parameters['alpha'],parameters['lambda']
            if selected is not None and method in selected:
                choice = selected[method]
                if not np.allclose([alpha,lam],[choice['alpha'],choice['lambda']],atol=1e-12,rtol=0):
                    parameter_matches = False
                    if args.strict_archive:
                        raise AssertionError(f'{job_key}/{method}: validation did not reproduce archived parameters')
                alpha, lam = choice['alpha'],choice['lambda']
            theta = gc.fit(p,alpha,lam,component)
            pred = gc.predict(x[test],theta,p.target_mean,p.target_std)
            ref = fixed_pred if fixed else joint_pred
            ref = ref[(ref.dataset==ds)&(ref.backbone==back)&(ref.split_seed==seed)&(ref.method==method)]
            if not fixed: ref = ref[ref.regime=='core']
            if ref.sample_id.duplicated().any(): raise AssertionError('Ambiguous reference run')
            want = ref.set_index('sample_id').loc[split.loc[test,'sample_id'],'y_pred'].to_numpy()
            error = float(np.max(np.abs(pred-want)))
            if not np.isfinite(error) or (args.strict_archive and error > 1e-10):
                raise AssertionError(f'{job_key}/{method}: prediction error {error}')
            score = gc.metrics(y[test],pred,groups[test])
            checks.append({'dataset':ds,'backbone':back,'split_seed':seed,'method':method,
                           'alpha':alpha,'lambda':lam,'prediction_error':error,**score})
            if method in ('sample_joint','ridge_matched'):
                np.savez_compressed(out/f'{job_key}_{method}.npz',theta=theta,target_mean=p.target_mean,
                                    target_std=p.target_std,alpha=alpha,lam=lam,dataset=ds,backbone=back)
        print(json.dumps({'completed':job_key,'models_checked':len(specs),'validation_search':selected is not None}),flush=True)
    pd.DataFrame(checks).to_csv(out/'refitted_metrics.csv',index=False)
    prediction_matches = all(r['prediction_error'] <= 1e-10 for r in checks)
    summary = {'passed':prediction_matches and parameter_matches,'execution_completed':True,
               'all_feature_fingerprints_match_archive':all(feature_matches.values()),
               'predictions_match_archive':prediction_matches,
               'selected_parameters_match_archive':parameter_matches if args.command=='tune' else None,
               'configurations':len(jobs),'models':len(checks),
               'maximum_prediction_absolute_error':max(r['prediction_error'] for r in checks),
               'validation_search_configurations':len(jobs) if args.command=='tune' else 0,
               'feature_extraction_rerun':False,'python':platform.python_version(),
               'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__}
    write_json(out/'fit_audit.json',summary)
    print(json.dumps(summary),flush=True)


def main():
    global DATA_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['results','fit','tune','predict'])
    parser.add_argument('--out',type=Path,default=Path('reproduced'))
    parser.add_argument('--data',type=Path,default=ROOT/'local_data')
    parser.add_argument('--features',type=Path,default=ROOT/'local_data/features')
    parser.add_argument('--strict-archive',action='store_true',help='Require archived feature fingerprints, selected parameters and predictions.')
    parser.add_argument('--dataset',choices=list(SLUGS),default='AGIQA-3K-full')
    parser.add_argument('--backbone',choices=['RN50','ViT-B-32'],default='RN50')
    parser.add_argument('--seed',type=int,choices=range(42,47),default=42)
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--model',type=Path)
    parser.add_argument('--input-features',type=Path)
    args = parser.parse_args()
    DATA_DIR = args.data
    args.out = output_directory(args.out)
    if args.command=='results': results(args.out)
    elif args.command in ('fit','tune'): fitting(args,args.out)
    else:
        if not args.model or not args.input_features: parser.error('predict requires --model and --input-features')
        with np.load(args.model,allow_pickle=False) as model, np.load(args.input_features,allow_pickle=False) as data:
            pred = gc.predict(data['features'],model['theta'],float(model['target_mean']),float(model['target_std']))
            ids = data['sample_ids'] if 'sample_ids' in data else np.arange(len(pred))
            if ids.ndim != 1 or len(ids) != len(pred) or len(np.unique(ids)) != len(ids):
                raise ValueError('Input sample_ids must be one-dimensional, aligned and unique.')
        pd.DataFrame({'sample_id':ids,'prediction':pred}).to_csv(args.out/'predictions.csv',index=False)
        print(json.dumps({'predictions':len(pred)}))


if __name__=='__main__':
    main()
