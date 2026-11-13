"""Conditional law, objective, sampling, and history discrepancy regressions."""
import os
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from joint_production.conditional_rates import ConditionalRateForecaster, INTERPRETATION, _u_scores
from joint_production.history_features import last_interval_rate_features
from joint_production.metrics import fit_increment_scales
from observation_reference import sample_gaussian_reports
from joint_production.torch_observation import sample_accumulated_reports


DAYS = np.array([0, 30, 60, 90, 180, 360, 720, 1080], dtype=float)
ORIGINS = [60, 90, 180, 360, 720]


def synthetic_reports(n=24):
    rng = np.random.default_rng(29)
    amplitude = rng.lognormal(0, .35, (n, 1, 3)) * [150, 700, 120]
    values = amplitude * (1-np.exp(-DAYS[None, :, None]/300))
    values += rng.normal(0, .1, values.shape)
    values[:, 0] = 0
    return values


class TestConditionalObjective(unittest.TestCase):
    def test_dense_u_score_formula_includes_hidden_origin_and_past(self):
        days = torch.tensor([0., 30, 60, 90, 180], dtype=torch.float64)
        target = torch.arange(15, dtype=torch.float64).reshape(1, 5, 3)
        target[:, 0] = 0
        observed = torch.tensor([[[1,1,1],[1,0,1],[1,0,1],[0,0,0],[0,0,0]]], dtype=torch.bool)
        samples = target[:, None].repeat(1, 4, 1, 1)
        samples += torch.tensor([0., 1, -2, 3])[None, :, None, None]
        samples[:, :, 2, 1] += torch.tensor([1., 2, 3, 4])
        samples = torch.where(observed[:, None], target[:, None], samples)
        scale = torch.tensor([[2., 4, 3], [5., 6, 8]], dtype=torch.float64)
        marginal, energy, dimension = _u_scores(target, samples, observed, days, 2, scale, torch.ones_like(scale))
        increments = ((samples[0, :, 3:] - samples[0, :, 2, None])/scale).numpy()
        actual = ((target[0, 3:] - target[0, 2])/scale).numpy()
        reference_marginal = np.abs(increments-actual).mean(axis=0)
        reference_marginal -= sum(np.abs(increments[i]-increments[j]) for i in range(4) for j in range(i+1,4))/12
        self.assertAlmostEqual(float(marginal), reference_marginal.mean(), places=12)
        scored = (~observed[0] & (days > 0)[:, None]).numpy()
        x = samples.numpy()[0][:, scored]
        y = target.numpy()[0][scored]
        expected_energy = np.linalg.norm(x-y,axis=1).mean() - sum(np.linalg.norm(x[i]-x[j]) for i in range(4) for j in range(i+1,4))/12
        self.assertEqual(int(dimension), 8)  # Hidden day30 and origin gas plus six future endpoints.
        self.assertAlmostEqual(float(energy), expected_energy/np.sqrt(8), places=12)

    def test_coincident_samples_have_finite_gradients(self):
        days = torch.tensor([0., 30, 60])
        target = torch.zeros((2, 3, 3))
        samples = torch.zeros((2, 4, 3, 3), requires_grad=True)
        observed = torch.ones_like(target, dtype=torch.bool)
        observed[:, 2] = False
        m, e, _ = _u_scores(target, samples, observed, days, 1, torch.ones((1,3)), torch.ones((1,3)))
        (m+e).sum().backward()
        self.assertTrue(torch.isfinite(samples.grad).all())
        torch.testing.assert_close(m+e, torch.zeros(2))

    def test_uniform_origin_weights_reproduce_frozen_macro_cell_score_with_missing_targets(self):
        reports = synthetic_reports(5)
        reports[0, 4, 1] = np.nan
        counts = np.array([5,4,3,2,1])
        all_context_scores, all_cell_scores = [], []
        for origin, count in zip(ORIGINS, counts):
            index = np.flatnonzero(DAYS == origin)[0]
            target = torch.tensor(reports, dtype=torch.float64)
            observed = torch.isfinite(target) & torch.tensor(DAYS <= origin)[None,:,None]
            samples = torch.tensor(np.nan_to_num(reports)[:,None].repeat(3,axis=1),dtype=torch.float64)
            for k, end in enumerate(np.flatnonzero(DAYS > origin)):
                samples[:,:,end] += k+1
            valid = np.isfinite(reports[:, DAYS > origin]) & np.isfinite(reports[:,index,None])
            correction = torch.tensor(len(reports)/valid.sum(axis=0))
            m, _, _ = _u_scores(target, samples, observed, torch.tensor(DAYS), index,
                                torch.ones((count,3)), correction)
            all_context_scores.append(float(m.mean())*count/counts.mean())
            actual = reports[:, DAYS > origin]-reports[:, index,None]
            prediction = samples.numpy()[:,:,DAYS > origin]-samples.numpy()[:,:,index,None]
            per_well = np.abs(prediction-actual[:,None]).mean(axis=1)
            all_cell_scores.extend(np.nanmean(per_well,axis=0).ravel())
        self.assertAlmostEqual(np.mean(all_context_scores), np.mean(all_cell_scores), places=12)


class TestConditionalRateForecaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = synthetic_reports()
        cls.models = {}
        for family in ("basis", "dca"):
            cls.models[family] = ConditionalRateForecaster(family, epochs=2, hidden_width=16,
                batch_size=12, train_samples=4, log_every=0, seed=17).fit(cls.reports,DAYS,origins=ORIGINS)

    def test_family_controls_share_context_masks_and_exact_training_scales(self):
        a,b = self.models.values()
        self.assertEqual(a.fit_diagnostics_["context_mask_sha256_by_epoch"], b.fit_diagnostics_["context_mask_sha256_by_epoch"])
        self.assertEqual(a.fit_diagnostics_["total_training_contexts"],24*5*2)
        self.assertEqual(a.fit_diagnostics_["optimizer_updates"],2*5*2)
        increments = np.stack([self.reports[:,DAYS==t, :][:,0]-self.reports[:,DAYS==s,:][:,0]
                               for s in ORIGINS for t in DAYS if t>s],axis=1)
        expected = fit_increment_scales(increments)
        np.testing.assert_array_equal(a.increment_scales.cpu(),expected)
        np.testing.assert_array_equal(b.increment_scales.cpu(),expected)

    def test_cutoff_leakage_zero_masks_and_conditional_interpretation(self):
        for model in self.models.values():
            history = self.reports[:2].copy()
            history[:,DAYS>90] = np.nan
            changed = history.copy()
            changed[:,DAYS>90] = np.inf
            a=model.predict_distribution(history,90,8,np.random.default_rng(13))
            b=model.predict_distribution(changed,90,8,np.random.default_rng(13))
            np.testing.assert_array_equal(a["report_cumulative"],b["report_cumulative"])
            np.testing.assert_array_equal(a["latent_parameters"],b["latent_parameters"])
            self.assertEqual(a["interpretation"],INTERPRETATION)
            self.assertNotIn("posterior_diagnostics",a)
            np.testing.assert_array_equal(a["report_cumulative"][:,:,:4],np.repeat(history[:,None,:4],8,axis=1))
            zero=history.copy();zero[:,1,0]=0
            missing=zero.copy();missing[:,1,0]=np.nan
            self.assertFalse(torch.equal(model.conditional_parameters(zero,90)[0],model.conditional_parameters(missing,90)[0]))

    def test_missing_phase_origin_and_paths_are_joint_and_physically_admissible(self):
        for model in self.models.values():
            history=self.reports[:2].copy()
            history[:,:,1]=np.nan
            result=model.predict_distribution(history,90,500,np.random.default_rng(42))
            report=result["report_cumulative"]
            self.assertTrue(np.isfinite(report).all())
            self.assertTrue(np.all(np.diff(result["latent_cumulative"],axis=2)>=0))
            self.assertTrue(np.all(result["latent_rate"]>=0))
            self.assertTrue(np.all(result["latent_total_volume"]>=result["latent_cumulative"][:,:,-1]))
            np.testing.assert_array_equal(report[:,:,0],0)
            self.assertGreater(report[0,:,3,1].std(),0)
            self.assertGreater(np.corrcoef(report[0,:,3,1],report[0,:,4,1])[0,1],.1)
            np.testing.assert_allclose(report[:,:,7]-report[:,:,3],(report[:,:,7]-report[:,:,5])+(report[:,:,5]-report[:,:,3]),rtol=1e-12,atol=1e-9)
            query=model.evaluate_trajectories(result["latent_parameters"],[0,90,135,180,1080])
            np.testing.assert_allclose(query["latent_cumulative"][:,:,[0,1,3,4]],result["latent_cumulative"][:,:,[0,3,4,7]],rtol=1e-12)

    def test_checkpoint_reproduces_draws_and_metadata(self):
        for family,model in self.models.items():
            with TemporaryDirectory() as directory:
                path=Path(directory)/f"{family}.pt"
                model.save(path)
                restored=ConditionalRateForecaster.load(path)
            self.assertEqual(model.fit_diagnostics_,restored.fit_diagnostics_)
            original=model.predict_distribution(self.reports[:2],60,12,np.random.default_rng(6))
            copied=restored.predict_distribution(self.reports[:2],60,12,np.random.default_rng(6))
            for key in ("latent_parameters","latent_cumulative","latent_rate","latent_total_volume","report_cumulative"):
                np.testing.assert_array_equal(original[key],copied[key])

    def test_training_and_prediction_bridge_agree_with_independent_numpy_reference(self):
        model=self.models["basis"]
        history=self.reports[:1].copy();history[:,DAYS>90]=np.nan;history[:,1:4,1]=np.nan
        latent=np.broadcast_to(model.rate_model.initial_latent(dtype=torch.float64).numpy(),(1,6000,12))
        q=model.evaluate_trajectories(latent,DAYS)["latent_cumulative"]
        scale=model.volume_scale.cpu().numpy()
        a=model.noise_abs.detach().double();r=model.noise_rel.detach().double()
        with torch.no_grad():
            actual=sample_accumulated_reports(torch.tensor(q/scale),torch.tensor(history/scale),torch.tensor(DAYS),
                a,r,cutoff=90,generator=torch.Generator().manual_seed(61)).numpy()*scale
        expected=sample_gaussian_reports(q,history,DAYS,a.numpy()*scale,r.numpy(),cutoff=90,
                                         rng=np.random.default_rng(62),law="accumulated")
        select=np.array([4,7,10,13,16,19,22])
        x=actual.reshape(6000,-1)[:,select]/np.tile(scale,8)[select]
        y=expected.reshape(6000,-1)[:,select]/np.tile(scale,8)[select]
        np.testing.assert_allclose(x.mean(0),y.mean(0),atol=.006)
        np.testing.assert_allclose(np.cov(x.T),np.cov(y.T),atol=.001,rtol=.12)

    def test_gradients_reach_location_cholesky_and_both_discrepancy_parameters(self):
        model=self.models["basis"]
        y=torch.tensor(self.reports[:4]/model.volume_scale.cpu().numpy(),dtype=torch.float32)
        days=torch.tensor(DAYS,dtype=torch.float32)
        observed=torch.isfinite(y)&(days<=90)[None,:,None]
        native=torch.tensor(self.reports[:4],dtype=torch.float64)
        mean,lower=model.network(model._encoder_context(y,observed,days,90,native_reports=native))
        mean.retain_grad();lower.retain_grad()
        epsilon=torch.randn((4,8,12),generator=torch.Generator().manual_seed(123))
        q=model.rate_model.cumulative(mean[:,None]+torch.einsum("nij,nsj->nsi",lower,epsilon),days)
        history=torch.where(observed,y,torch.full_like(y,float("nan")))
        report=sample_accumulated_reports(q,history,days,model.noise_abs,model.noise_rel,cutoff=90,
                                          generator=torch.Generator().manual_seed(124))
        m,e,_=_u_scores(y,report,observed,days,3,torch.ones((4,3)),torch.ones((4,3)))
        model.zero_grad();(m+e).mean().backward()
        for value in (mean.grad,lower.grad,model.noise_abs_raw.grad,model.noise_rel_raw.grad):
            self.assertTrue(torch.isfinite(value).all())
            self.assertGreater(float(value.abs().sum()),0)
        self.assertGreater(float(torch.tril(lower.grad,diagonal=-1).abs().sum()),0)

    def test_seed_reproducibility_and_no_future_availability_feature(self):
        kwargs=dict(epochs=1,hidden_width=8,batch_size=12,train_samples=4,log_every=0,seed=77)
        a=ConditionalRateForecaster(**kwargs).fit(self.reports,DAYS,origins=[60])
        b=ConditionalRateForecaster(**kwargs).fit(self.reports,DAYS,origins=[60])
        self.assertEqual(a.fit_diagnostics_,b.fit_diagnostics_)
        for key,value in a.state_dict().items():
            torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)

    def test_finite_training_value_outside_float32_support_is_not_reclassified_missing(self):
        reports=self.reports.copy()
        reports[0,1,0]=1e100
        with self.assertRaisesRegex(ValueError,"float32"):
            ConditionalRateForecaster(epochs=1,hidden_width=8,log_every=0).fit(reports,DAYS,origins=[60])

    def test_native_report_overflow_is_rejected_after_unit_conversion(self):
        model=self.models["basis"]
        def extreme_bridge(cumulative,*args,**kwargs):
            return torch.full_like(cumulative,1e308)
        with patch("joint_production.conditional_rates.sample_accumulated_reports",side_effect=extreme_bridge):
            with self.assertRaisesRegex(FloatingPointError,"Native report conversion"):
                model.predict_distribution(self.reports[:1],60,2,np.random.default_rng(20))


class TestConditionalRateHistoryFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = synthetic_reports(8)
        calls = []

        def record_features(*args, **kwargs):
            calls.append((args[0].clone(), args[1].clone(), args[3]))
            return last_interval_rate_features(*args, **kwargs)

        with patch("joint_production.conditional_rates.last_interval_rate_features", side_effect=record_features):
            cls.model = ConditionalRateForecaster(epochs=1, hidden_width=8, batch_size=4,
                train_samples=3, log_every=0, seed=19, history_dropout=.4,
                phase_dropout=.3).fit(cls.reports, DAYS, origins=[60, 90])
        cls.feature_calls = calls

    def test_training_features_receive_native_values_and_actual_dropout_masks(self):
        self.assertTrue(self.feature_calls)
        dropout_seen = False
        for native, observed, cutoff in self.feature_calls:
            self.assertEqual(native.dtype, torch.float64)
            self.assertFalse(observed[:, DAYS > cutoff].any())
            self.assertTrue(observed[:, 0].all())
            dropout_seen |= bool((~observed[:, (DAYS > 0) & (DAYS <= cutoff)]).any())
            for row in native.numpy():
                self.assertTrue(any(np.array_equal(row, source) for source in self.reports))
        self.assertTrue(dropout_seen)

    def test_encoder_preserves_native_difference_and_recomputes_after_dropout(self):
        model = ConditionalRateForecaster(hidden_width=8)
        model._initialize(DAYS, np.full(3, 1e12), np.ones((1, 3)))
        native = torch.zeros((1, len(DAYS), 3), dtype=torch.float64)
        native[:, 1], native[:, 2] = 1e12, 1e12+1
        observed = torch.zeros_like(native, dtype=torch.bool)
        observed[:, :3] = True
        normalized = (native/model.volume_scale).float()
        self.assertTrue(torch.equal(normalized[:, 1], normalized[:, 2]))
        context = model._encoder_context(normalized, observed, model.report_days.float(), 60, native_reports=native)
        expected = torch.asinh(torch.tensor(360/30/1e12, dtype=torch.float64)).float()
        torch.testing.assert_close(context[:, -6:-3], expected.expand(1, 3), rtol=0, atol=0)
        self.assertTrue((context[:, -6:-3] > 0).all())
        observed[:, 2] = False
        changed = native.clone()
        changed[:, 2:] = float("inf")
        after = model._encoder_context(normalized, observed, model.report_days.float(), 60, native_reports=changed)
        torch.testing.assert_close(after[:, -6:-3], torch.asinh(torch.tensor(12.)).expand(1, 3))

    def test_checkpoint_and_future_value_invariance(self):
        model = self.model
        history = self.reports[:2].copy()
        changed = history.copy()
        changed[:, DAYS > 60] = np.inf
        first = model.predict_samples(history, 60, 3, np.random.default_rng(23))
        second = model.predict_samples(changed, 60, 3, np.random.default_rng(23))
        np.testing.assert_array_equal(first, second)
        with TemporaryDirectory() as directory:
            path = Path(directory)/"features.pt"
            model.save(path)
            restored = ConditionalRateForecaster.load(path)
        self.assertEqual(restored.fit_diagnostics_, model.fit_diagnostics_)
        copied = restored.predict_samples(history, 60, 3, np.random.default_rng(23))
        np.testing.assert_array_equal(first, copied)

    def test_unsupported_options_rejected(self):
        for options in (dict(history_rate_features="none"), dict(covariate_preprocessing="internal"),
                        dict(basis_dictionary="slow"), dict(rate_family="arps")):
            with self.assertRaises(ValueError):
                ConditionalRateForecaster(**options)


if __name__=="__main__":
    unittest.main()
