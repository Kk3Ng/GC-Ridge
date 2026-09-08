"""GC-Ridge 的独立 CPU 实现；输入为冻结的图像特征。"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import rankdata


def correlation(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 3:
        return float('nan')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float('nan')
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float('nan')
    a, b = a-a.mean(), b-b.mean()
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denominator) if denominator > 0 else float('nan')


def metrics(y, pred, groups):
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    groups = np.asarray(groups)
    if y.shape != pred.shape or y.shape != groups.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError('Expected aligned, nonempty one-dimensional arrays.')
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('Targets and predictions must be finite.')
    if any(k is None or (isinstance(k, (float, np.floating)) and not np.isfinite(k))
           or (isinstance(k, str) and not k.strip()) for k in groups):
        raise ValueError('Generator labels must be present for every sample.')
    yr, pr, yc, pc = [np.empty(len(y), dtype=np.float64) for _ in range(4)]
    local = []
    for k in np.unique(groups):
        ix = groups == k
        ry, rp = rankdata(y[ix], method='average'), rankdata(pred[ix], method='average')
        yr[ix], pr[ix] = ry-ry.mean(), rp-rp.mean()
        yc[ix] = y[ix]-y[ix].mean() if np.ptp(y[ix]) > 0 else 0.
        pc[ix] = pred[ix]-pred[ix].mean() if np.ptp(pred[ix]) > 0 else 0.
        local.append(correlation(ry, rp))
    p, w = correlation(rankdata(y), rankdata(pred)), correlation(yr, pr)
    # 遇到未定义组时保留 N/A，不悄悄丢弃该组后求平均。
    return {'P':p, 'W':w, 'balanced':(p+w)/2, 'macro':float(np.mean(local)),
            'worst':float(np.min(local)), 'PLCC':correlation(y,pred),
            'RMSE':float(np.sqrt(np.mean((y-pred)**2))), 'centered_PLCC':correlation(yc,pc)}


@dataclass
class Moments:
    h: np.ndarray
    g: np.ndarray
    c: np.ndarray
    q: np.ndarray
    equal_c: np.ndarray
    equal_q: np.ndarray
    target_mean: float
    target_std: float
    feature_mean: np.ndarray


def prepare(x_train, y_train, groups_train):
    """只接收训练集，避免从验证/测试目标计算标准化统计量。"""
    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    groups = np.asarray(groups_train)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or groups.shape != y.shape:
        raise ValueError('Misaligned training arrays.')
    if len(y) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Non-finite or empty training data.')
    if x.shape[1] == 0:
        raise ValueError('Training features must have at least one column.')
    if any(k is None or (isinstance(k, (float, np.floating)) and not np.isfinite(k))
           or (isinstance(k, str) and not k.strip()) for k in groups):
        raise ValueError('Generator labels must be present for every training sample.')
    mean, std = float(y.mean()), float(y.std(ddof=0))
    if std < 1e-12:
        raise ValueError('Training MOS has zero variance.')
    y = (y-mean)/std
    d = x.shape[1]+1
    h, c, ec = [np.zeros((d,d)) for _ in range(3)]
    g, q, eq = [np.zeros(d) for _ in range(3)]
    labels = np.unique(groups)
    for k in labels:
        ix = groups == k
        xx, yy = x[ix], y[ix]
        z = np.column_stack((xx, np.ones(len(xx))))
        h += z.T@z
        g += z.T@yy
        xc, yc = xx-xx.mean(axis=0), yy-yy.mean()
        gram, rhs = xc.T@xc, xc.T@yc
        c[:-1,:-1] += gram
        q[:-1] += rhs
        factor = len(x)/(len(labels)*len(xx))
        ec[:-1,:-1] += factor*gram
        eq[:-1] += factor*rhs
    return Moments(h,g,c,q,ec,eq,mean,std,x.mean(axis=0))


def solve(gram, rhs, alpha):
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be positive and finite.')
    a = gram.copy()
    a.flat[::a.shape[0]+1] += np.r_[np.full(a.shape[0]-1, alpha), 0.0]
    return cho_solve(cho_factor(a, lower=True), rhs)


def fit(moments, alpha, lam, component='gc'):
    p = moments
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be positive and finite.')
    if not np.isfinite(lam) or lam < 0:
        raise ValueError('lambda must be nonnegative and finite.')
    if component == 'gc':
        return solve(p.h+lam*p.c, p.g+lam*p.q, alpha)
    if component == 'equal_generator':
        return solve(p.h+lam*p.equal_c, p.g+lam*p.equal_q, alpha)
    if component == 'pooled_only':
        return solve(p.h, p.g, alpha)
    if component == 'uncentered':
        return solve((1+lam)*p.h, (1+lam)*p.g, alpha)
    if component == 'within_only':
        a = lam*p.c[:-1,:-1] + alpha*np.eye(len(p.q)-1)
        slope = cho_solve(cho_factor(a, lower=True), lam*p.q[:-1])
        # 标准化后训练目标均值为零；截距由汇总训练均值恢复。
        return np.r_[slope, -p.feature_mean@slope]
    raise ValueError(f'Unknown component: {component}')


def predict(x, theta, target_mean, target_std):
    """推理仅需特征与模型参数，不需要生成器身份。"""
    x, theta = np.asarray(x, dtype=np.float64), np.asarray(theta, dtype=np.float64)
    if x.ndim != 2 or theta.shape != (x.shape[1]+1,) or not np.isfinite(x).all() or not np.isfinite(theta).all():
        raise ValueError('Invalid features or incompatible model.')
    mean, std = np.asarray(target_mean), np.asarray(target_std)
    if mean.ndim != 0 or std.ndim != 0 or not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError('Model MOS scaling requires a finite mean and positive finite standard deviation.')
    return (x@theta[:-1]+theta[-1])*target_std+target_mean


def ridge_grid(alphas, lambdas):
    grid = []
    for value in sorted(a/(1+l) for a in alphas for l in lambdas):
        if not grid or not math.isclose(value, grid[-1], rel_tol=1e-12, abs_tol=1e-14):
            grid.append(value)
    if len(grid) != 58:
        raise ValueError('Expected the paper grids with 58 distinct effective penalties.')
    while len(grid) < 63:
        index = max(range(len(grid)-1), key=lambda i: math.log(grid[i+1]/grid[i]))
        grid.insert(index+1, math.sqrt(grid[index]*grid[index+1]))
    return grid


def select(candidates, floor):
    eligible = [r for r in candidates if r['P'] >= floor and math.isfinite(r['W'])]
    if not eligible:
        raise ValueError('No finite candidate satisfies validation retention.')
    return max(eligible, key=lambda r: (r['balanced'],r['W'],r['P'],-r['lambda'],r['alpha']))


def tune(p, x_val, y_val, groups_val, alphas, lambdas, delta=0.01):
    """验证集搜索 GC-Ridge 与 Ridge-63；函数不接收测试目标。"""
    traces, selected = {}, {}
    base = predict(x_val, fit(p, 10., 0.), p.target_mean, p.target_std)
    floor = metrics(y_val, base, groups_val)['P']-delta
    specs = {
        'sample_joint': [(a,l) for a in alphas for l in lambdas],
        'ridge_matched': [(a,0.) for a in ridge_grid(alphas,lambdas)],
    }
    for method, pairs in specs.items():
        rows = []
        for alpha, lam in pairs:
            theta = fit(p,alpha,lam)
            pred = predict(x_val,theta,p.target_mean,p.target_std)
            rows.append({'alpha':alpha,'lambda':lam, **metrics(y_val,pred,groups_val)})
        traces[method] = rows
        selected[method] = select(rows, floor)
    return selected, traces
