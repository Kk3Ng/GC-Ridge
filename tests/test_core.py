"""检验论文公式与实现的对应关系，以及排名边界条件。"""
import json
from pathlib import Path
import unittest
from uuid import uuid4
import numpy as np
from numpy.testing import assert_allclose
import gc_ridge as gc
import reproduce


class CoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(913)
        self.x = rng.normal(size=(21,5))
        self.y = rng.normal(size=21)+np.repeat([0.,5.,-2.],[5,7,9])
        self.groups = np.repeat(['a','b','c'],[5,7,9])

    def test_solver_matches_augmented_least_squares(self):
        p = gc.prepare(self.x,self.y,self.groups)
        y = (self.y-p.target_mean)/p.target_std
        z = np.column_stack((self.x,np.ones(len(y))))
        xc, yc = np.empty_like(self.x), np.empty_like(y)
        for k in np.unique(self.groups):
            ix = self.groups==k
            xc[ix],yc[ix] = self.x[ix]-self.x[ix].mean(0), y[ix]-y[ix].mean()
        alpha, lam = 3.,10.
        design = np.vstack((z,np.sqrt(lam)*np.column_stack((xc,np.zeros(len(y)))),
                            np.sqrt(alpha)*np.column_stack((np.eye(5),np.zeros(5)))))
        target = np.r_[y,np.sqrt(lam)*yc,np.zeros(5)]
        direct = np.linalg.lstsq(design,target,rcond=None)[0]
        assert_allclose(gc.fit(p,alpha,lam),direct,rtol=0,atol=1e-12)

    def test_uncentered_control_is_effective_ridge(self):
        p = gc.prepare(self.x,self.y,self.groups)
        assert_allclose(gc.fit(p,10,3,'uncentered'),gc.fit(p,2.5,0),atol=1e-12)

    def test_group_weighting_agrees_when_balanced(self):
        groups = np.repeat(['a','b','c'],7)
        p = gc.prepare(self.x,self.y,groups)
        assert_allclose(gc.fit(p,3,10),gc.fit(p,3,10,'equal_generator'),atol=1e-12)

    def test_intercept_and_mos_scaling(self):
        p = gc.prepare(self.x,self.y,self.groups)
        p2 = gc.prepare(self.x,2*self.y+100,self.groups)
        first = gc.predict(self.x,gc.fit(p,1,3),p.target_mean,p.target_std)
        second = gc.predict(self.x,gc.fit(p2,1,3),p2.target_mean,p2.target_std)
        assert_allclose(second,2*first+100,atol=1e-12)

    def test_within_rank_can_disagree_with_pooled_rank(self):
        score = gc.metrics([1.,2.,3.,100.,101.,102.], [3.,2.,1.,102.,101.,100.], ['a']*3+['b']*3)
        self.assertGreater(score['P'],0)
        self.assertAlmostEqual(score['W'],-1)
        self.assertAlmostEqual(score['macro'],-1)

    def test_constant_predictions_stay_undefined(self):
        score = gc.metrics(np.arange(718.),np.full(718,0.98568315494),np.repeat(['a','b'],359))
        for name in ['P','W','macro','PLCC','centered_PLCC']:
            self.assertTrue(np.isnan(score[name]),name)
        partial = gc.metrics(np.arange(6.),[1,1,1,3,4,5],['a']*3+['b']*3)
        self.assertTrue(np.isnan(partial['macro']))

    def test_selection_floor_and_tie_order(self):
        rows = [dict(alpha=a,**{'lambda':l},P=p,W=w,balanced=(p+w)/2)
                for a,l,p,w in [(1,0,.1,1.),(1,3,.8,.6),(1,1,.8,.6),(10,1,.8,.6)]]
        selected = gc.select(rows,.79)
        self.assertEqual((selected['alpha'],selected['lambda']),(10,1))

    def test_ridge_grid_matches_frozen_grid(self):
        config = json.loads((Path(__file__).resolve().parents[1]/'configs/experiment.json').read_text())
        assert_allclose(gc.ridge_grid(config['alphas'],config['lambdas']),config['ridge_alphas'],atol=0,rtol=0)

    def test_invalid_model_parameters_are_rejected(self):
        for theta,mean,std in [(np.array([np.nan,0,0]),0.,1.),(np.zeros(3),np.inf,1.),
                               (np.zeros(3),0.,0.),(np.zeros(3),0.,-1.),(np.zeros(3),0.,np.nan)]:
            with self.subTest(mean=mean,std=std), self.assertRaises(ValueError):
                gc.predict(np.ones((2,2)),theta,mean,std)

    def test_integer_rmse_does_not_overflow(self):
        score = gc.metrics(np.arange(6,dtype=np.int64)+10**12,np.arange(6,dtype=np.int64),np.array(['a']*3+['b']*3))
        assert_allclose(score['RMSE'],10**12,rtol=1e-14,atol=0)

    def test_nonfinite_scores_and_missing_groups_are_rejected(self):
        for y,pred,groups in [([1,np.nan,3],[1,2,3],['a']*3),
                              ([1,2,3],[1,np.inf,3],['a']*3),
                              ([1,2,3],[1,2,3],np.array(['a',None,'a'],dtype=object))]:
            with self.assertRaises(ValueError): gc.metrics(y,pred,groups)
        with self.assertRaises(ValueError):
            gc.prepare(self.x,self.y,np.full(len(self.y),np.nan))

    def test_within_only_requires_positive_regularization(self):
        p = gc.prepare(self.x,self.y,self.groups)
        for alpha in [0.,-1.,np.nan]:
            with self.assertRaises(ValueError): gc.fit(p,alpha,3.,'within_only')

    def test_outputs_cannot_overwrite_packaged_inputs(self):
        for path in [reproduce.ROOT,reproduce.ROOT/'evidence',reproduce.ROOT/'models/new_run']:
            with self.assertRaises(ValueError): reproduce.output_directory(path)
        folder = reproduce.ROOT/('.gc-ridge-test-'+uuid4().hex)
        folder.mkdir()
        target = folder/'separate_outputs'
        try:
            self.assertEqual(reproduce.output_directory(target),target.resolve())
        finally:
            self.assertEqual(folder.resolve().parent,reproduce.ROOT.resolve())
            if target.exists(): target.rmdir()
            folder.rmdir()


if __name__=='__main__':
    unittest.main()
