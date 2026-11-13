"""Direct conditional mixing laws over continuous joint rate paths.

This is pi_theta(z | cutoff history, optional available attributes) times a Gaussian conditional
report bridge. The default accumulates discrepancies; an optional fixed positive
report half-life gives covariance exp(-log(2)*abs(t-s)/h)*V(min(t,s)).
It is NOT a Bayesian latent posterior and does not assert one
unconditional population/measurement law or sequential consistency across
cutoffs. History must not be used to reweight these conditional latent draws.
Positive rate families supply accounting/integration guarantees; their tails
are operating assumptions, and report scores do not identify latent truth.

One shared MLP emits a full Gaussian over 12 fast-basis or 9 DCA log parameters,
or two conditional Gaussian components sharing the same bounded covariance.
The default uses iid predictive draws; the mixture uses independent draws per
component with analytic probability weights. Both use the prediction bridge.
The default proper-score objective is equal-cell/equal-phase scaled increment
CRPS plus equal-origin joint energy on all measured, context-unobserved
positive-day reports, normalized by sqrt(the scored dimension). Both terms
have coefficient one. Uniform-origin context batches receive marginal weight
m_s / mean(m_s). Training-count correction preserves cell means if some raw
training endpoints are missing. Repeated origins are conditional tasks, not
replicated independent observations. Artificial masks are constructed studies.

The bridge pins known reports for any latent path, so historical latent/report
discrepancies are recorded explicitly as descriptive diagnostics. They are not
an added loss, an identified meter-noise estimate, or posterior adequacy tests.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .analytic_mean_score import fast_basis_increment_mean_mse
from .covariance_parameterization import initial_diagonal_anchor, normalized_row_cholesky, normalized_row_metadata
from .dca_rates import DCARate
from .earlier_history_features import (
    EARLIER_END_DAYS, FEATURE_COUNT as EARLIER_FEATURE_COUNT,
    REPORT_DAYS as EARLIER_REPORT_DAYS, earlier_interval_rate_features,
)
from .fading_observation import fading_transition, sample_fading_reports
from .gaussian_increment_score import accumulated_increment_u_crps
from .history_features import last_interval_rate_features
from .metrics import fit_increment_scales
from .rates import PositiveBasisRate
from .sampled_quantile_score import empirical_wis, sampled_u_crps
from .torch_observation import sample_accumulated_reports


INTERPRETATION = "conditional_mixing_law_not_bayesian_posterior"
MIXTURE_INITIAL_DELTA = 0.05
REPORT_INTEGRATED_SCORE_MEANING = (
    "Report-noise conditional expectation of marginal iid U-CRPS given the same latent base draws; "
    "unchanged sampled joint energy and predictive law. No general guarantee of reduced total gradient variance."
)
SAMPLED_OBJECTIVE_MEANINGS = {
    "crps": "Marginal iid U-CRPS from all 256 joint report draws, using the sorted-gap identity; self pairs excluded.",
    "empirical_wis": (
        "Marginal empirical 80%/95% WIS from all 256 joint report draws; linear quantiles "
        "[0.025,0.1,0.5,0.9,0.975], five pinball losses divided by 2.5. "
        "Finite-bank expected WIS differs from population-quantile WIS; no unbiasedness claim."
    ),
}
ENERGY_SAMPLE_SELECTION = "first_16_of_same_256_iid_joint_report_trajectories"
REPORT_SCALE_INTERPRETATION = (
    "Context-fixed conditional report kernel; not identified meter noise, a history-independent "
    "measurement likelihood, or a cross-cutoff Bayesian model."
)
ANALYTIC_MEAN_SCORE_MEANING = (
    "Squared error of the exact conditional reported-increment mean, not squared error of a "
    "sample mean or expected squared sampled error. A proper score for the conditional first "
    "moment under finite second moments; adds no latent identification or Bayesian interpretation. "
    "The positive weighted sum changes the projection within a restricted predictive family."
)
EARLIER_HISTORY_FEATURE_ORDER = "end_day_then_oil_gas_water_then_rate_duration_availability"
EARLIER_HISTORY_FEATURE_MEANING = (
    "Earlier consecutive surviving report intervals, excluding each phase's latest surviving pair; "
    "native float64 differences after cutoff and masking, retaining signed and zero reports. "
    "Deterministic coordinates of existing information, not independent likelihood terms or a Bayesian posterior."
)
EARLIER_HISTORY_FEATURE_LAYOUT = tuple(
    f"{day}:{phase}:{feature}" for day in EARLIER_END_DAYS
    for phase in ("oil", "gas", "water")
    for feature in ("asinh_signed_rate", "duration_360", "availability")
)


def _u_scores(target_reports, report_samples, context_observed, days, origin_index,
              increment_scales, target_weights):
    """Per-context iid U-CRPS and U-energy; input volumes are phase-normalized.

    increment_scales are native cell scales divided by the same phase scales.
    target_weights=N_train/n_observed_cell_phase are frozen during fitting.
    Every future interval subtracts the SAME sampled origin endpoint.
    """
    count = report_samples.shape[1]
    if count < 2:
        raise ValueError("Unbiased proper-score training requires at least two iid draws")
    future = days > days[origin_index]
    target_mask = torch.isfinite(target_reports[:, future]) & torch.isfinite(target_reports[:, origin_index, None])
    target_increment = target_reports[:, future] - target_reports[:, origin_index, None]
    target_increment = torch.where(target_mask, target_increment, torch.zeros_like(target_increment))
    sample_increment = report_samples[:, :, future] - report_samples[:, :, origin_index, None]
    standardized = sample_increment / increment_scales
    actual = target_increment / increment_scales
    pair_left, pair_right = torch.triu_indices(count, count, offset=1, device=report_samples.device)
    marginal = (standardized - actual[:, None]).abs().mean(dim=1)
    marginal -= (standardized[:, pair_left] - standardized[:, pair_right]).abs().sum(dim=1) / (count * (count - 1))
    marginal = (marginal * target_mask * target_weights).mean(dim=(-2, -1))
    unobserved = torch.isfinite(target_reports) & ~context_observed & (days > 0)[None, :, None]
    safe_target = torch.where(unobserved, target_reports, torch.zeros_like(target_reports))
    residual = torch.where(unobserved[:, None], report_samples - safe_target[:, None],
                           torch.zeros_like(report_samples)).flatten(start_dim=2)
    # vector_norm has a finite zero subgradient; do not sqrt a squared zero
    # distance and then mask an undefined derivative afterwards.
    energy = torch.linalg.vector_norm(residual, dim=-1).mean(dim=1)
    pairs = residual[:, pair_left] - residual[:, pair_right]
    energy -= torch.linalg.vector_norm(pairs, dim=-1).sum(dim=1) / (count * (count - 1))
    dimension = unobserved.sum(dim=(-2, -1))
    energy /= dimension.clamp_min(1).to(report_samples.dtype).sqrt()
    return marginal, energy, dimension


def _sampled_bank_scores(target_reports, report_samples, context_observed, days, origin_index,
                         increment_scales, target_weights, *, marginal_objective):
    """Fixed 256-draw marginal score and first-16 joint U-energy.

    Target validity is established on endpoints before subtraction/scaling.
    Any active arithmetic overflow raises before the generic finite-mask score
    can mistake it for missing data. Signed targets and frozen cell/phase
    weights are preserved. A hidden origin is subtracted from the SAME report
    draw; context dropout never changes the raw target mask. This helper draws
    no randomness and leaves the original ``_u_scores`` implementation intact.
    """
    if marginal_objective not in SAMPLED_OBJECTIVE_MEANINGS:
        raise ValueError("Unknown fixed-bank marginal objective")
    if (report_samples.ndim != 4 or report_samples.shape[1] != 256
            or report_samples.shape[-1] != 3
            or target_reports.shape != report_samples.shape[:1] + report_samples.shape[2:]):
        raise ValueError("Fixed-bank scores require targets [N,T,3] and samples [N,256,T,3]")
    if (isinstance(origin_index, (bool, np.bool_)) or not isinstance(origin_index, (int, np.integer))
            or not 0 <= origin_index < len(days) - 1
            or days.shape != (target_reports.shape[1],)
            or context_observed.shape != target_reports.shape or context_observed.dtype != torch.bool):
        raise ValueError("Fixed-bank score origin, days, or history-mask axes disagree")
    future = days > days[origin_index]
    if (increment_scales.shape != (int(future.sum()), 3) or target_weights.shape != increment_scales.shape
            or not bool(torch.isfinite(increment_scales).all()) or bool((increment_scales <= 0).any())
            or not bool(torch.isfinite(target_weights).all()) or bool((target_weights < 0).any())):
        raise ValueError("Fixed-bank scores require finite positive scales and nonnegative target weights")
    if bool(torch.isinf(target_reports).any()) or not bool(torch.isfinite(report_samples).all()):
        raise FloatingPointError("Training endpoints or joint samples exceeded floating-point support")
    finite = torch.isfinite(target_reports)
    target_mask = finite[:, future] & finite[:, origin_index, None]
    safe_target = torch.where(finite, target_reports, torch.zeros_like(target_reports))
    target_increment = safe_target[:, future] - safe_target[:, origin_index, None]
    target_increment = torch.where(target_mask, target_increment, torch.zeros_like(target_increment))
    sample_increment = report_samples[:, :, future] - report_samples[:, :, origin_index, None]
    standardized_target = target_increment / increment_scales
    standardized_samples = sample_increment / increment_scales
    if (not bool(torch.isfinite(target_increment).all())
            or not bool(torch.isfinite(standardized_target).all())
            or not bool(torch.isfinite(sample_increment).all())
            or not bool(torch.isfinite(standardized_samples).all())):
        raise FloatingPointError("Training increment subtraction or normalization exceeded floating-point support")
    score = sampled_u_crps if marginal_objective == "crps" else empirical_wis
    cells, valid = score(standardized_target, standardized_samples, target_mask=target_mask)
    if not torch.equal(valid, target_mask):
        raise FloatingPointError("Scoring changed the raw endpoint target mask")
    marginal = (cells * target_weights).mean(dim=(-2, -1))
    _, energy, dimension = _u_scores(target_reports, report_samples[:, :16], context_observed,
                                    days, origin_index, increment_scales, target_weights)
    if not bool(torch.isfinite(marginal).all()) or not bool(torch.isfinite(energy).all()):
        raise FloatingPointError("Fixed-bank training scores exceeded floating-point support")
    return marginal, energy, dimension


def _stratified_scores(target_reports, report_samples, component_weights,
                       context_observed, days, origin_index, increment_scales,
                       target_weights):
    """Unbiased two-component CRPS/energy with independent draws per stratum.

    Samples have shape [context, component=2, draw, day, phase]. The weights
    are analytic probabilities, not random component assignments. Within each
    stratum the unordered U-statistic excludes self pairs; across strata all
    pairs are independent. Both latent and bridge innovations must be drawn
    independently across all samples, including when component laws are tied.
    """
    if report_samples.ndim != 5 or report_samples.shape[1] != 2:
        raise ValueError("Stratified scores require [context,2,draw,day,phase] samples")
    count = report_samples.shape[2]
    if count < 2 or component_weights.shape != report_samples.shape[:2]:
        raise ValueError("Require at least two independent draws per component and [context,2] weights")
    if (not bool(torch.isfinite(component_weights).all()) or bool((component_weights < 0).any())
            or not torch.allclose(component_weights.sum(dim=1), torch.ones_like(component_weights[:, 0]))):
        raise ValueError("Component weights must be finite nonnegative probabilities summing to one")
    pair_left, pair_right = torch.triu_indices(count, count, offset=1, device=report_samples.device)

    def weighted_score(target_distance, within_distance, cross_distance):
        extra = (1,) * (target_distance.ndim - 3)
        weight = component_weights.reshape(*component_weights.shape, *extra)
        result = (weight * target_distance.mean(dim=2)).sum(dim=1)
        result -= (weight.square() * within_distance.sum(dim=2)).sum(dim=1) / (count * (count - 1))
        cross_weight = (component_weights[:, 0] * component_weights[:, 1]).reshape(-1, *extra)
        result -= cross_weight * cross_distance.sum(dim=(1, 2)) / count**2
        return result

    future = days > days[origin_index]
    target_mask = torch.isfinite(target_reports[:, future]) & torch.isfinite(target_reports[:, origin_index, None])
    target_increment = target_reports[:, future] - target_reports[:, origin_index, None]
    target_increment = torch.where(target_mask, target_increment, torch.zeros_like(target_increment))
    sample_increment = report_samples[:, :, :, future] - report_samples[:, :, :, origin_index, None]
    standardized = sample_increment / increment_scales
    actual = target_increment / increment_scales
    marginal = weighted_score((standardized - actual[:, None, None]).abs(),
        (standardized[:, :, pair_left] - standardized[:, :, pair_right]).abs(),
        (standardized[:, 0, :, None] - standardized[:, 1, None]).abs())
    marginal = (marginal * target_mask * target_weights).mean(dim=(-2, -1))

    unobserved = torch.isfinite(target_reports) & ~context_observed & (days > 0)[None, :, None]
    safe_target = torch.where(unobserved, target_reports, torch.zeros_like(target_reports))
    residual = torch.where(unobserved[:, None, None], report_samples - safe_target[:, None, None],
                           torch.zeros_like(report_samples)).flatten(start_dim=3)
    energy = weighted_score(torch.linalg.vector_norm(residual, dim=-1),
        torch.linalg.vector_norm(residual[:, :, pair_left] - residual[:, :, pair_right], dim=-1),
        torch.linalg.vector_norm(residual[:, 0, :, None] - residual[:, 1, None], dim=-1))
    dimension = unobserved.sum(dim=(-2, -1))
    energy /= dimension.clamp_min(1).to(report_samples.dtype).sqrt()
    return marginal, energy, dimension


class _ContextNetwork(nn.Module):
    def __init__(self, input_size, width, initial, min_std, max_std, initial_std, max_offdiag,
                 covariance_mode="bounded_cholesky"):
        super().__init__()
        self.covariance_mode = covariance_mode
        if covariance_mode == "marginal_correlation":
            self.register_buffer("covariance_diagonal_anchor_logit", torch.empty(0, dtype=torch.float64))
        self.dimension = len(initial)
        self.min_std, self.max_std, self.max_offdiag = min_std, max_std, max_offdiag
        self.hidden = nn.Sequential(nn.Linear(input_size, width), nn.SiLU(),
                                    nn.Linear(width, width), nn.SiLU())
        self.mean = nn.Linear(width, self.dimension)
        indices = torch.tril_indices(self.dimension, self.dimension)
        self.register_buffer("lower_rows", indices[0])
        self.register_buffer("lower_columns", indices[1])
        self.cholesky = nn.Linear(width, indices.shape[1])
        nn.init.zeros_(self.mean.weight)
        with torch.no_grad():
            self.mean.bias.copy_(initial)
        nn.init.zeros_(self.cholesky.weight)
        nn.init.zeros_(self.cholesky.bias)
        fraction = (initial_std - min_std) / (max_std - min_std)
        with torch.no_grad():
            self.cholesky.bias[indices[0] == indices[1]] = math.log(fraction / (1 - fraction))

    def forward(self, context):
        hidden = self.hidden(context)
        return self._parameters_from_hidden(hidden)

    def _parameters_from_hidden(self, hidden, *, mean_head=None, cholesky_head=None):
        raw = self.cholesky(hidden) if cholesky_head is None else cholesky_head(hidden)
        if self.covariance_mode == "marginal_correlation":
            lower = normalized_row_cholesky(raw, diagonal_anchor_logit=self.covariance_diagonal_anchor_logit)
            return self.mean(hidden) if mean_head is None else mean_head(hidden), lower
        diagonal = self.lower_rows == self.lower_columns
        entries = self.max_offdiag * torch.tanh(raw)
        standard_deviations = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(raw)
        entries = torch.where(diagonal[None], standard_deviations, entries)
        lower = raw.new_zeros((len(raw), self.dimension, self.dimension))
        lower[:, self.lower_rows, self.lower_columns] = entries
        return self.mean(hidden) if mean_head is None else mean_head(hidden), lower


class _HistorySpecializedContextNetwork(_ContextNetwork):
    """One joint output head for each post-mask history-availability regime.

    The trunk and context are shared unchanged. Cloning consumes no random
    numbers. Only a head receiving rows is evaluated, preserving None gradients
    and Adam state for a wholly unused head.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_history_mean = deepcopy(self.mean)
        self.no_history_cholesky = deepcopy(self.cholesky)

    def forward(self, context, no_history):
        if (no_history.dtype != torch.bool or no_history.shape != (len(context),)
                or no_history.device != context.device):
            raise ValueError("no_history must be one boolean per context on the same device")
        hidden = self.hidden(context)
        mean = hidden.new_zeros((len(hidden), self.dimension))
        lower = hidden.new_zeros((len(hidden), self.dimension, self.dimension))
        for mask, mean_head, cholesky_head in (
                (~no_history, self.mean, self.cholesky),
                (no_history, self.no_history_mean, self.no_history_cholesky)):
            rows = torch.nonzero(mask, as_tuple=False).flatten()
            if rows.numel():
                selected_mean, selected_lower = self._parameters_from_hidden(
                    hidden[rows], mean_head=mean_head, cholesky_head=cholesky_head)
                mean = mean.index_copy(0, rows, selected_mean)
                lower = lower.index_copy(0, rows, selected_lower)
        return mean, lower


class _MixtureContextNetwork(_ContextNetwork):
    """Two joint latent locations and one shared bounded Cholesky factor."""

    def __init__(self, *args, head_seed, **kwargs):
        super().__init__(*args, **kwargs)
        # The extra heads cannot alter common network initialization or any
        # training innovation stream. The initial split is fixed and treats
        # every latent coordinate symmetrically in magnitude.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(head_seed)
            self.delta = nn.Linear(self.mean.in_features, self.dimension)
            self.logits = nn.Linear(self.mean.in_features, 2)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)
        with torch.no_grad():
            signs = torch.where(torch.arange(self.dimension) % 2 == 0, 1.0, -1.0)
            self.delta.bias.copy_(MIXTURE_INITIAL_DELTA * signs)

    def forward(self, context):
        hidden = self.hidden(context)
        mean, lower = self._parameters_from_hidden(hidden)
        delta = self.delta(hidden)
        means = torch.stack((mean - delta, mean + delta), dim=1)
        # Keep analytic scoring probabilities and inference thresholds in the
        # same float64 precision; Gaussian locations/factors remain float32
        # during training, as in the original model.
        return means, lower, torch.softmax(self.logits(hidden).double(), dim=-1)


class _ContextReportScale(nn.Module):
    """Bias-free linear map initialized without any random operation."""

    def __init__(self, input_size, reference):
        super().__init__()
        self.in_features, self.out_features = input_size, 3
        self.register_parameter("bias", None)
        self.weight = nn.Parameter(reference.new_zeros((3, input_size)))

    def forward(self, context):
        return torch.nn.functional.linear(context, self.weight)


class ConditionalRateForecaster(nn.Module):
    """History-conditioned rate mixing law with an exact Gaussian report bridge.

    ``report_half_life_days=None`` preserves the accumulated kernel. A finite
    positive half-life fixes exponential discrepancy memory without adding
    trainable parameters or changing the marginal variance clock.

    ``history_head_mode='no_history'`` clones the joint mean/Cholesky head
    for wells with no observed positive-time report after cutoff and masking.
    An observed zero counts as history. The shared trunk and original context
    remain unchanged; empty histories at different cutoffs need not share a law.

    For the basis-only ``coefficient_parameterization='log_expectation'``,
    the unchanged base head plus half the frozen initial marginal variance
    predicts log expected positive coefficients. The centered correction
    ``mu=base-.5*(diag(LL^T)-v_init)`` preserves the complete initial law.
    It preserves the Gaussian family at each context, without claiming an
    identical neural function class or a covariance-independent report law.

    ``tree_static_mode='fixed_empty_history'`` requires a separate ``tree_prior``
    matrix of twelve positive native coefficient means, supplied out of fold
    for TRAIN and by a full-TRAIN teacher at prediction. Twelve standardized
    log-prior features augment the context. When no positive-time report is
    available after masking, the complete joint Gaussian has exactly those
    coefficient expectations in real arithmetic; observed zeros count as data.
    Histories containing reports retain the selected learned mean rule. This
    is a bundled conditional predictor, not a Bayesian tree-prior update.

    ``marginal_score_estimator='gaussian_report_integrated'`` analytically
    averages only the marginal U-CRPS over accumulated Gaussian report noise.
    It keeps the same latent draws, every report draw, sampled joint energy,
    score weights and prediction law. This estimator requires one Gaussian
    latent component and the accumulated report kernel; the default ``sampled``
    retains the original estimator and saved configuration. Reduced marginal
    noise alone does not guarantee reduced variance of the total gradient.

    ``report_scale_mode='contextual_multiplier'`` multiplies both phase noise
    scales by exp(log(4)*tanh(W context)), fixed across the whole phase clock.
    Its zero-initialized bias-free head preserves the initial global law. This
    is a context-specific conditional report kernel, not identified meter noise
    or a single history-independent measurement likelihood.

    The opt-in ``energy_samples=16`` uses 256 iid joint trajectories for the
    marginal score and their first 16 for the unchanged joint U-energy.
    ``marginal_objective='crps'`` uses sorted U-CRPS; ``'empirical_wis'`` uses
    the five linear empirical quantiles defining 80%/95% WIS. The latter is a
    finite-bank objective, not an unbiased population-quantile score estimator.
    Both keep the same conditional rate/report law and random-draw schedule.
    The default ``energy_samples=None`` retains the original training path.

    ``mean_score_mode='analytic_mse'`` adds coefficient-one squared error of
    the exact basis conditional increment mean to unchanged CRPS16 and energy16.
    Its lognormal mean uses every entry in each Cholesky row. The same frozen
    cell scales, target counts and origin weights apply, independently of
    artificial history masks. No parameters or predictive draws are added.

    ``earlier_history_features='actual'`` adds 45 deterministic coordinates of
    consecutive surviving report intervals before each phase's latest pair.
    Its ``zero`` control adds the same number of zero inputs. Both append zero
    columns to the original first layer without consuming initialization RNG.
    This fixed eight-endpoint feature experiment retains sampled scores, the
    accumulated report law and the shared single-Gaussian log-location head.
    The original endpoint context and latest-interval summary remain intact.

    One epoch visits every training well at every specified origin once. Batches
    share an origin, but their order is shuffled. Context/mask, latent, report,
    and network-initialization RNG streams are separate, so basis versus DCA
    dimensions cannot change their training information schedule. Mixture
    training preserves those schedules and uses no random component labels;
    prediction assigns one iid component to each complete joint trajectory.

    Optional ``covariates`` have shape [well, feature]. The default ``internal``
    preprocessing fits training-only median/std statistics and appends presence
    masks. ``external`` instead consumes a finite matrix already encoded by the
    caller, without further scaling or masks. Its preprocessing and availability
    must be established on training data by the caller. Include origin zero in
    ``fit(origins=...)`` for attribute-only complete trajectories; the only
    observed report then is the deterministic accounting zero.
    """

    def __init__(self, rate_family="basis", *, basis_dictionary="fast", seed=42,
                 epochs=100, hidden_width=128, batch_size=256, train_samples=16,
                 learning_rate=1e-3, history_dropout=0.1, phase_dropout=0.1,
                 initial_latent_std=0.25, min_latent_std=0.025, max_latent_std=1.0,
                 max_offdiag=0.2, noise_floor=1e-4, initial_noise_abs=0.02,
                 initial_noise_rel=0.08, noise_regularization=0.0,
                 gradient_clip=10.0, log_every=25, device="cpu",
                 history_rate_features="none", mixture_mode="single",
                 covariate_preprocessing="internal", report_half_life_days=None,
                 history_head_mode="shared", coefficient_parameterization="log_location",
                 tree_static_mode="none", marginal_score_estimator="sampled",
                 earlier_history_features="none", marginal_objective="crps",
                 energy_samples=None, report_scale_mode="global", mean_score_mode="none",
                 covariance_mode="bounded_cholesky"):
        super().__init__()
        if rate_family not in {"basis", "dca"} or basis_dictionary != "fast":
            raise ValueError("This pilot supports fast basis or matched DCA only")
        if min(epochs, hidden_width, batch_size) < 1 or train_samples < 2:
            raise ValueError("Positive epoch/width/batch settings and at least two iid samples are required")
        if not 0 < min_latent_std < initial_latent_std < max_latent_std or max_offdiag <= 0:
            raise ValueError("Require 0 < min_std < initial_std < max_std and positive off-diagonal bound")
        if (learning_rate <= 0 or gradient_clip <= 0 or log_every < 0 or noise_floor <= 0
                or min(initial_noise_abs, initial_noise_rel) <= noise_floor or noise_regularization < 0):
            raise ValueError("Invalid optimizer or noise parameters")
        if not 0 <= history_dropout < 1 or not 0 <= phase_dropout < 1:
            raise ValueError("Masking probabilities must be in [0,1)")
        if history_rate_features not in {"none", "zero", "last_interval"}:
            raise ValueError("history_rate_features must be none, zero, or last_interval")
        if mixture_mode not in {"single", "tied", "two"}:
            raise ValueError("mixture_mode must be single, tied, or two")
        if mixture_mode != "single" and (train_samples < 4 or train_samples % 2):
            raise ValueError("Mixture training requires an even total sample count >= 4")
        if marginal_score_estimator not in ("sampled", "gaussian_report_integrated"):
            raise ValueError("marginal_score_estimator must be sampled or gaussian_report_integrated")
        if marginal_score_estimator != "sampled" and (mixture_mode != "single" or report_half_life_days is not None):
            raise ValueError("Integrated marginal scoring requires mixture_mode='single' and accumulated reports")
        if marginal_objective not in ("crps", "empirical_wis"):
            raise ValueError("marginal_objective must be crps or empirical_wis")
        if energy_samples is not None or marginal_objective != "crps":
            if (isinstance(energy_samples, (bool, np.bool_))
                    or not isinstance(energy_samples, (int, np.integer)) or energy_samples != 16
                    or isinstance(train_samples, (bool, np.bool_))
                    or not isinstance(train_samples, (int, np.integer)) or train_samples != 256):
                raise ValueError("Fixed-bank marginal objectives require train_samples=256 and energy_samples=16")
            if (mixture_mode != "single" or history_head_mode != "shared"
                    or coefficient_parameterization != "log_location" or tree_static_mode != "none"
                    or earlier_history_features != "none" or report_half_life_days is not None
                    or history_rate_features != "last_interval" or marginal_score_estimator != "sampled"):
                raise ValueError("Fixed-bank objectives require single, shared, log_location, no tree/earlier features, last_interval, sampled scores and accumulated reports")
        if earlier_history_features not in ("none", "zero", "actual"):
            raise ValueError("earlier_history_features must be none, zero, or actual")
        if earlier_history_features != "none" and (mixture_mode != "single" or history_head_mode != "shared"
                or coefficient_parameterization != "log_location" or tree_static_mode != "none"
                or history_rate_features != "last_interval" or report_half_life_days is not None
                or marginal_score_estimator != "sampled"):
            raise ValueError("Earlier history features require single, shared, log_location, no tree, last_interval, sampled scores and accumulated reports")
        if covariate_preprocessing not in {"internal", "external"}:
            raise ValueError("covariate_preprocessing must be internal or external")
        if history_head_mode not in {"shared", "no_history"}:
            raise ValueError("history_head_mode must be shared or no_history")
        if history_head_mode != "shared" and mixture_mode != "single":
            raise ValueError("History head specialization currently requires mixture_mode='single'")
        if coefficient_parameterization not in {"log_location", "log_expectation"}:
            raise ValueError("coefficient_parameterization must be log_location or log_expectation")
        if coefficient_parameterization == "log_expectation" and (rate_family != "basis"
                or mixture_mode != "single" or history_head_mode != "shared"):
            raise ValueError("log_expectation requires basis, single mixture mode and shared history head")
        if tree_static_mode not in {"none", "fixed_empty_history"}:
            raise ValueError("tree_static_mode must be none or fixed_empty_history")
        if tree_static_mode != "none" and (rate_family != "basis" or mixture_mode != "single"
                or history_head_mode != "shared" or report_half_life_days is not None):
            raise ValueError("Tree static conditioning requires basis, single, shared and accumulated reports")
        if report_half_life_days is not None:
            if (isinstance(report_half_life_days, (bool, np.bool_))
                    or not isinstance(report_half_life_days, (int, float, np.integer, np.floating))
                    or not math.isfinite(report_half_life_days) or report_half_life_days <= 0):
                raise ValueError("report_half_life_days must be None or finite and positive")
            if mixture_mode != "single":
                raise ValueError("Finite report_half_life_days currently requires mixture_mode='single'")
        if report_scale_mode not in {"global", "contextual_multiplier"}:
            raise ValueError("report_scale_mode must be global or contextual_multiplier")
        if report_scale_mode != "global" and (rate_family != "basis" or mixture_mode != "single"
                or history_head_mode != "shared" or coefficient_parameterization != "log_location"
                or tree_static_mode != "none" or earlier_history_features != "none"
                or report_half_life_days is not None or history_rate_features != "last_interval"
                or covariate_preprocessing != "external" or marginal_score_estimator != "sampled"
                or marginal_objective != "crps" or energy_samples is not None or train_samples != 16):
            raise ValueError("Contextual report scales require fast basis, shared single log_location, external last_interval, sampled CRPS16 and accumulated reports without other experimental modes")
        if mean_score_mode not in {"none", "analytic_mse"}:
            raise ValueError("mean_score_mode must be none or analytic_mse")
        if mean_score_mode != "none" and (rate_family != "basis" or mixture_mode != "single"
                or history_head_mode != "shared" or coefficient_parameterization != "log_location"
                or tree_static_mode != "none" or earlier_history_features != "none"
                or report_half_life_days is not None or history_rate_features != "last_interval"
                or covariate_preprocessing != "external" or marginal_score_estimator != "sampled"
                or marginal_objective != "crps" or energy_samples is not None or train_samples != 16
                or report_scale_mode != "global"):
            raise ValueError("Analytic mean scoring requires fast basis, shared single log_location, external last_interval, global noise, sampled CRPS16 and accumulated reports without other experimental modes")
        if covariance_mode not in {"bounded_cholesky", "marginal_correlation"}:
            raise ValueError("covariance_mode must be bounded_cholesky or marginal_correlation")
        if covariance_mode != "bounded_cholesky" and (rate_family != "basis" or mixture_mode != "single"
                or history_head_mode != "shared" or coefficient_parameterization != "log_location"
                or tree_static_mode != "none" or earlier_history_features != "none"
                or report_half_life_days is not None or history_rate_features != "last_interval"
                or covariate_preprocessing != "external" or marginal_score_estimator != "sampled"
                or marginal_objective != "crps" or energy_samples is not None or train_samples != 16
                or report_scale_mode != "global" or mean_score_mode != "none"
                or (initial_latent_std, min_latent_std, max_latent_std, max_offdiag) != (.25, .025, 1., .2)):
            raise ValueError("Marginal correlation covariance requires the retained fast basis, single/shared log_location, external last_interval, global accumulated noise, CRPS16+energy16, original covariance initialization and no other experimental modes")
        self.covariance_mode = covariance_mode
        self.mean_score_mode = mean_score_mode
        self.report_scale_mode = report_scale_mode
        self.mixture_mode = mixture_mode
        self.history_head_mode = history_head_mode
        self.coefficient_parameterization = coefficient_parameterization
        self.tree_static_mode = tree_static_mode
        self.marginal_score_estimator = marginal_score_estimator
        self.marginal_objective = marginal_objective
        self.energy_samples = None if energy_samples is None else int(energy_samples)
        self.earlier_history_features = earlier_history_features
        self.report_half_life_days = None if report_half_life_days is None else float(report_half_life_days)
        self.covariate_preprocessing = covariate_preprocessing
        self.n_covariates_ = self.covariate_feature_count_ = 0
        self.covariate_center_ = np.empty(0, dtype=np.float64)
        self.covariate_scale_ = np.empty(0, dtype=np.float64)
        self.configuration = dict(rate_family=rate_family, basis_dictionary=basis_dictionary, seed=int(seed),
            epochs=int(epochs), hidden_width=int(hidden_width), batch_size=int(batch_size), train_samples=int(train_samples),
            learning_rate=float(learning_rate), history_dropout=float(history_dropout), phase_dropout=float(phase_dropout),
            initial_latent_std=float(initial_latent_std), min_latent_std=float(min_latent_std), max_latent_std=float(max_latent_std),
            max_offdiag=float(max_offdiag), noise_floor=float(noise_floor), initial_noise_abs=float(initial_noise_abs),
            initial_noise_rel=float(initial_noise_rel), noise_regularization=float(noise_regularization),
            gradient_clip=float(gradient_clip), log_every=int(log_every), device=str(device),
            history_rate_features=history_rate_features)
        # Omitting the default preserves existing schema-1 configuration and
        # state dictionaries as well as the original iid training behavior.
        if mixture_mode != "single":
            self.configuration["mixture_mode"] = mixture_mode
        if covariate_preprocessing != "internal":
            self.configuration["covariate_preprocessing"] = covariate_preprocessing
        if self.report_half_life_days is not None:
            self.configuration["report_half_life_days"] = self.report_half_life_days
        if history_head_mode != "shared":
            self.configuration["history_head_mode"] = history_head_mode
        if coefficient_parameterization != "log_location":
            self.configuration["coefficient_parameterization"] = coefficient_parameterization
        if tree_static_mode != "none":
            self.configuration["tree_static_mode"] = tree_static_mode
        if marginal_score_estimator != "sampled":
            self.configuration["marginal_score_estimator"] = marginal_score_estimator
        if self.energy_samples is not None:
            self.configuration.update(marginal_objective=self.marginal_objective, energy_samples=self.energy_samples)
        if earlier_history_features != "none":
            self.configuration["earlier_history_features"] = earlier_history_features
        if report_scale_mode != "global":
            self.configuration["report_scale_mode"] = report_scale_mode
        if mean_score_mode != "none":
            self.configuration["mean_score_mode"] = mean_score_mode
        if covariance_mode != "bounded_cholesky":
            self.configuration["covariance_mode"] = covariance_mode
        for key, value in self.configuration.items():
            if key != "device":
                setattr(self, key, value)
        self.rate_model = PositiveBasisRate("fast") if rate_family == "basis" else DCARate()
        self.latent_size = self.rate_model.latent_size
        self.network = None
        self.register_buffer("report_days", torch.empty(0, dtype=torch.float64, device=device))
        self.register_buffer("volume_scale", torch.empty(0, dtype=torch.float64, device=device))
        self.register_buffer("increment_scales", torch.empty(0, dtype=torch.float64, device=device))
        if coefficient_parameterization == "log_expectation":
            self.register_buffer("coefficient_initial_variance", torch.empty(0, dtype=torch.float32, device=device))
        if tree_static_mode != "none":
            self.register_buffer("tree_log_prior_center", torch.zeros(12, dtype=torch.float64, device=device))
            self.register_buffer("tree_log_prior_scale", torch.ones(12, dtype=torch.float64, device=device))
        inverse = lambda value: math.log(math.expm1(value - noise_floor))
        self.noise_abs_raw = nn.Parameter(torch.full((3,), inverse(initial_noise_abs), dtype=torch.float32, device=device))
        self.noise_rel_raw = nn.Parameter(torch.full((3,), inverse(initial_noise_rel), dtype=torch.float32, device=device))

    @property
    def device(self):
        return self.noise_abs_raw.device

    @property
    def noise_abs(self):
        return torch.nn.functional.softplus(self.noise_abs_raw) + self.noise_floor

    @property
    def noise_rel(self):
        return torch.nn.functional.softplus(self.noise_rel_raw) + self.noise_floor

    @property
    def observation_law(self):
        return ("conditional_gaussian_accumulated_bridge" if self.report_half_life_days is None
                else "conditional_gaussian_fading_bridge")

    def _report_scale_metadata(self):
        if self.report_scale_mode == "global":
            return {}
        return {"report_scale_mode": self.report_scale_mode,
                "report_scale_multiplier_bounds": [0.25, 4.0],
                "report_scale_multiplier_formula": "exp(log(4)*tanh(context @ weight.T))",
                "report_scale_clock": "one_context_fixed_phase_multiplier_for_entire_accumulated_clock",
                "report_scale_interpretation": REPORT_SCALE_INTERPRETATION,
                "report_scale_head_initialization": "exact_zero_weight_without_rng",
                "report_scale_arithmetic": "float32_head_and_products_using_recorded_float32_fitted_bases_at_prediction_then_float64_promotion",
                "noise_parameter_scope": "fit_noise_abs_scaled_and_noise_rel_are_global_base_parameters"}

    def _covariance_metadata(self, *, diagonal_anchor_logit=None):
        if self.covariance_mode == "bounded_cholesky":
            return {}
        if diagonal_anchor_logit is None:
            diagonal_anchor_logit = self.network.covariance_diagonal_anchor_logit
        return {"covariance_mode": self.covariance_mode, **normalized_row_metadata(diagonal_anchor_logit),
                "covariance_moment_scope": "finite_conditional_lognormal_moments_at_finite_context; no_population_moment_guarantee",
                "covariance_initial_law": "L=recorded_legacy_initial_std_float32*I; nominal_L=.25*I; original_raw_parameters_and_local_factor_Jacobian_at_float32_precision"}

    def _mean_score_metadata(self):
        if self.mean_score_mode == "none":
            return {}
        return {"mean_score_mode": self.mean_score_mode, "mean_score_coefficient": 1.0,
                "mean_score_meaning": ANALYTIC_MEAN_SCORE_MEANING,
                "mean_score_arithmetic": "float64_analytic_moments_from_existing_float32_context_and_normalized_training_inputs"}

    def _report_scale_head_sha256(self):
        return hashlib.sha256(self.report_scale_head.weight.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    def _report_scale_base_raw_sha256(self):
        return hashlib.sha256(b"".join(value.detach().cpu().contiguous().numpy().tobytes()
            for value in (self.noise_abs_raw, self.noise_rel_raw))).hexdigest()

    def effective_noise(self, context, *, fitted_base=False):
        """Return a, r, g for the same post-mask encoder context used by the latent law.

        Global mode returns the original tensors and no multiplier. In the
        opt-in mode, each well/phase uses one g for its entire variance clock.
        The caller must construct context only from cutoff-available inputs.
        Prediction preserves the recorded fitted float32 base values across
        devices; training differentiates through the original raw parameters.
        """
        if self.report_scale_mode == "global":
            return self.noise_abs, self.noise_rel, None
        if (context.ndim != 2 or context.shape[1] != self.report_scale_head.in_features
                or not bool(torch.isfinite(context).all())):
            raise ValueError("Contextual report scale requires a finite aligned encoder context")
        multiplier = torch.exp(math.log(4.0) * torch.tanh(self.report_scale_head(context)))
        base_a, base_r = self.noise_abs, self.noise_rel
        if fitted_base and "noise_abs_scaled" in self.fit_diagnostics_:
            if self._report_scale_base_raw_sha256() != self.fit_diagnostics_["report_scale_base_raw_sha256"]:
                raise ValueError("Contextual base noise parameters changed since fitting; recorded prediction bases are stale")
            base_a = context.new_tensor(self.fit_diagnostics_["noise_abs_scaled"])
            base_r = context.new_tensor(self.fit_diagnostics_["noise_rel"])
        a, r = base_a * multiplier, base_r * multiplier
        if not all(bool(torch.isfinite(value).all()) for value in (a, r, multiplier)):
            raise FloatingPointError("Nonfinite contextual report noise parameters")
        return a, r, multiplier

    def _training_score_metadata(self):
        """Describe only the opt-in objective; legacy metadata remains absent."""
        if self.energy_samples is None:
            return {}
        return {"marginal_objective": self.marginal_objective,
                "energy_samples": self.energy_samples,
                "marginal_objective_meaning": SAMPLED_OBJECTIVE_MEANINGS[self.marginal_objective],
                "energy_sample_selection": ENERGY_SAMPLE_SELECTION}

    def _sample_reports(self, cumulative, history, days, noise_abs, noise_rel, **kwargs):
        """Use the identical conditional report kernel in optimization and sampling."""
        if self.report_half_life_days is None:
            return sample_accumulated_reports(cumulative, history, days, noise_abs, noise_rel, **kwargs)
        return sample_fading_reports(cumulative, history, days, noise_abs, noise_rel,
                                     half_life_days=self.report_half_life_days, **kwargs)

    def _initialize(self, days, phase_scales, increment_scales):
        self.report_days = torch.as_tensor(days, dtype=torch.float64, device=self.device)
        if self.earlier_history_features != "none" and not torch.equal(
                self.report_days, self.report_days.new_tensor(EARLIER_REPORT_DAYS)):
            raise ValueError("Earlier history features require the frozen eight-endpoint report grid")
        self.volume_scale = torch.as_tensor(phase_scales, dtype=torch.float64, device=self.device)
        self.increment_scales = torch.as_tensor(increment_scales, dtype=torch.float64, device=self.device)
        cuda_devices = [self.device.index or torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.seed)
            initial = self.rate_model.initial_latent(dtype=torch.float32)
            extra_features = 0 if self.history_rate_features == "none" else 6
            network_class = (_HistorySpecializedContextNetwork if self.history_head_mode == "no_history"
                             else _MixtureContextNetwork if self.mixture_mode == "two" else _ContextNetwork)
            extra = {"head_seed": self.seed + 4001} if self.mixture_mode == "two" else {}
            if self.covariance_mode != "bounded_cholesky":
                extra["covariance_mode"] = self.covariance_mode
            self.network = network_class(2 * len(days) * 3 + 2 * len(days) + 1 + extra_features
                                          + self.covariate_feature_count_, self.hidden_width,
                                          initial, self.min_latent_std, self.max_latent_std,
                                          self.initial_latent_std, self.max_offdiag, **extra).to(self.device)
        if self.covariance_mode != "bounded_cholesky":
            # The old float32 sigmoid/multiply/add differs by one ULP on the
            # tested CPU/CUDA runtimes. Match the final training device's old
            # initial law, then retain these anchor bytes across device moves.
            self.network.covariance_diagonal_anchor_logit = initial_diagonal_anchor(self.device)
        if self.tree_static_mode != "none":
            # Build exactly the old network first. Extending the existing
            # parameter by zero columns consumes no initialization RNG.
            first = self.network.hidden[0]
            first.weight = nn.Parameter(torch.cat((first.weight.detach(), first.weight.new_zeros((first.out_features, 12))), dim=1))
            first.in_features += 12
        if self.earlier_history_features != "none":
            first = self.network.hidden[0]
            first.weight = nn.Parameter(torch.cat((first.weight.detach(),
                first.weight.new_zeros((first.out_features, EARLIER_FEATURE_COUNT))), dim=1))
            first.in_features += EARLIER_FEATURE_COUNT
        if self.report_scale_mode != "global":
            self.report_scale_head = _ContextReportScale(self.network.hidden[0].in_features, self.noise_abs_raw)
        if self.coefficient_parameterization == "log_expectation":
            # Decode AFTER moving the initialized network to its final device.
            # No random operation or extra trainable parameter is introduced.
            with torch.no_grad():
                context = self.noise_abs_raw.new_zeros((1, self.network.hidden[0].in_features))
                _, initial_lower = self.network(context)
                self.coefficient_initial_variance = initial_lower.square().sum(dim=-1)[0].clone()
        digest, common_digest, head_digest, clone_digest = (hashlib.sha256() for _ in range(4))
        for name, parameter in self.named_parameters():
            values = parameter.detach().cpu().contiguous()
            def update(hasher, canonical_name, data=values):
                hasher.update(json.dumps([canonical_name, str(data.dtype), list(data.shape)], separators=(",", ":")).encode())
                hasher.update(data.contiguous().numpy().tobytes())
            update(digest, name)
            if name == "report_scale_head.weight":
                continue
            if name.startswith(("network.no_history_mean.", "network.no_history_cholesky.")):
                update(clone_digest, name.replace("network.no_history_", "network.", 1))
            else:
                common_values = values[:, :-12] if self.tree_static_mode != "none" and name == "network.hidden.0.weight" else values
                if self.earlier_history_features != "none" and name == "network.hidden.0.weight":
                    common_values = values[:, :-EARLIER_FEATURE_COUNT]
                update(common_digest, name, common_values)
                if name.startswith(("network.mean.", "network.cholesky.")):
                    update(head_digest, name)
        self.initial_parameter_sha256_ = digest.hexdigest()
        self.common_initial_parameter_sha256_ = common_digest.hexdigest()
        self.initial_output_head_sha256_ = head_digest.hexdigest()
        self.no_history_head_initial_sha256_ = clone_digest.hexdigest() if self.history_head_mode == "no_history" else None

    @staticmethod
    def _no_history_mask(observed, days):
        """Observed positive-time zero is data; the accounting origin is not."""
        return ~(observed & (days > 0)[None, :, None]).any(dim=(-2, -1))

    def _network_parameters(self, context, observed, days, *, tree_log_prior=None):
        if self.history_head_mode == "no_history":
            return self.network(context, self._no_history_mask(observed, days))
        if self.coefficient_parameterization == "log_location" and self.tree_static_mode == "none":
            return self.network(context)
        base, lower = self.network(context)
        # Include every lower-triangular entry in each marginal variance.
        # Do not detach: dispersion gradients must include the mean correction.
        variance = lower.square().sum(dim=-1)
        mean = base if self.coefficient_parameterization == "log_location" else base - .5*(variance-self.coefficient_initial_variance)
        if self.tree_static_mode != "none":
            if tree_log_prior is None or tree_log_prior.shape != mean.shape:
                raise ValueError("Tree conditioning requires aligned phase-normalized coefficient logs")
            empty = self._no_history_mask(observed, days)
            tree_mean = tree_log_prior - .5*variance
            # Entirely unused base heads retain None gradients and Adam state.
            if bool(empty.all()):
                return tree_mean, lower
            mean = torch.where(empty[:, None], tree_mean, mean)
        return mean, lower

    def _tree_prior_inputs(self, tree_prior, n, phase_scales, *, fit):
        """Separate native positive means; fitting input must be TRAIN OOF only.

        Phase-major coordinates are oil[4], gas[4], water[4]. The model cannot
        infer teacher provenance from a matrix; the runner binds OOF identities.
        No query record or future report participates in this preprocessing.
        """
        if self.tree_static_mode == "none":
            if tree_prior is not None:
                raise ValueError("tree_prior requires tree_static_mode='fixed_empty_history'")
            return None, None
        values = np.asarray(tree_prior, dtype=np.float64)
        phase_scales = np.asarray(phase_scales, dtype=np.float64)
        if (values.shape != (n, 12) or not np.isfinite(values).all() or np.any(values <= 0)
                or phase_scales.shape != (3,) or not np.isfinite(phase_scales).all() or np.any(phase_scales <= 0)):
            raise ValueError("Tree prior must contain finite positive native coefficients [well,12]")
        logs = np.log(values)-np.repeat(np.log(phase_scales), 4)
        if fit:
            center = np.median(logs, axis=0)
            scale = np.maximum(np.std(logs, axis=0, ddof=0), 1e-8)
            self.tree_log_prior_center = torch.as_tensor(center, dtype=torch.float64, device=self.device)
            self.tree_log_prior_scale = torch.as_tensor(scale, dtype=torch.float64, device=self.device)
            self.tree_prior_training_sha256_ = hashlib.sha256(values.astype("<f8").tobytes()).hexdigest()
        center, scale = self.tree_log_prior_center.cpu().numpy(), self.tree_log_prior_scale.cpu().numpy()
        with np.errstate(over="ignore", invalid="ignore"):
            features = ((logs-center)/scale).astype(np.float32)
            logs = logs.astype(np.float32)
        if not np.isfinite(features).all() or not np.isfinite(logs).all():
            raise ValueError("Tree prior features exceed float32 support")
        return logs, features

    @staticmethod
    def _context(reports, observed, days, origin):
        values = torch.asinh(torch.where(observed, reports, torch.zeros_like(reports)))
        grid = (days / days[-1]).expand(len(reports), -1)
        normalized_origin = torch.full((len(reports), 1), float(origin) / float(days[-1]),
                                       dtype=reports.dtype, device=reports.device)
        return torch.cat((values.flatten(start_dim=1), observed.to(reports.dtype).flatten(start_dim=1),
                          normalized_origin, grid, grid-normalized_origin), dim=-1)

    def _encoder_context(self, reports, observed, days, origin, *, native_reports=None, covariates=None):
        """Append available static attributes and optional interval-rate features.

        native_reports must retain the original float64 volumes. The observation
        mask is the actual post-dropout mask; no feature is recovered from a
        transformed/float32 endpoint or a dropped report. Zero control performs
        the same validation as the feature mode and consumes no additional RNG.
        """
        context = self._context(reports, observed, days, origin)
        if covariates is not None:
            context = torch.cat((context, covariates), dim=-1)
        if self.history_rate_features == "none":
            return context
        if native_reports is None:
            raise ValueError("History rate features require original native float64 reports")
        features = last_interval_rate_features(native_reports, observed, self.report_days,
            origin, self.volume_scale, dtype=context.dtype,
            zero_control=self.history_rate_features == "zero")
        context = torch.cat((context, features), dim=-1)
        if self.earlier_history_features != "none":
            earlier = earlier_interval_rate_features(native_reports, observed, self.report_days,
                origin, self.volume_scale, dtype=context.dtype,
                zero_control=self.earlier_history_features == "zero")
            context = torch.cat((context, earlier), dim=-1)
        return context

    def _earlier_history_metadata(self):
        if self.earlier_history_features == "none":
            return {}
        return {"earlier_history_features": self.earlier_history_features,
            "earlier_history_feature_count": EARLIER_FEATURE_COUNT,
            "earlier_history_feature_end_days": list(EARLIER_END_DAYS),
            "earlier_history_feature_order": EARLIER_HISTORY_FEATURE_ORDER,
            "earlier_history_feature_layout": list(EARLIER_HISTORY_FEATURE_LAYOUT),
            "earlier_history_feature_meaning": EARLIER_HISTORY_FEATURE_MEANING}

    def _covariates(self, covariates, n, *, fit):
        if covariates is None:
            if fit:
                self.n_covariates_ = self.covariate_feature_count_ = 0
                self.covariate_center_ = np.empty(0, dtype=np.float64)
                self.covariate_scale_ = np.empty(0, dtype=np.float64)
            elif self.n_covariates_:
                raise ValueError("This fitted model requires cutoff-available covariates")
            return np.empty((n, 0), dtype=np.float32)
        values = np.asarray(covariates, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != n or np.isinf(values).any():
            raise ValueError("covariates must have shape [well, feature], finite or NaN")
        valid = np.isfinite(values)
        if self.covariate_preprocessing == "external" and not valid.all():
            raise ValueError("Externally preprocessed covariates must all be finite")
        if fit:
            self.n_covariates_ = values.shape[1]
            self.covariate_feature_count_ = self.n_covariates_ * (2 if self.covariate_preprocessing == "internal" else 1)
            self.covariate_center_ = self.covariate_scale_ = np.empty(0, dtype=np.float64)
            if self.covariate_preprocessing == "internal":
                self.covariate_center_ = np.array([
                    np.median(values[valid[:, j], j]) if valid[:, j].any() else 0.0
                    for j in range(self.n_covariates_)])
                with np.errstate(over="ignore", invalid="ignore"):
                    self.covariate_scale_ = np.array([
                        max(float(np.std(values[valid[:, j], j])), 1e-8) if valid[:, j].any() else 1.0
                        for j in range(self.n_covariates_)])
                if not np.isfinite(self.covariate_center_).all() or not np.isfinite(self.covariate_scale_).all():
                    raise ValueError("Training covariate statistics exceeded finite floating-point support")
        if values.shape[1] != self.n_covariates_:
            raise ValueError("Covariate dimension differs from training")
        with np.errstate(over="ignore", invalid="ignore"):
            if self.covariate_preprocessing == "internal":
                standardized = np.where(valid, (values-self.covariate_center_)/self.covariate_scale_, 0.0)
                values = np.concatenate((standardized, valid.astype(float)), axis=1)
            features = values.astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Covariate features cannot be represented in float32")
        return features

    def _mixture_network_parameters(self, context):
        if self.mixture_mode == "two":
            return self.network(context)
        mean, lower = self.network(context)
        return mean[:, None].expand(-1, 2, -1), lower, mean.new_full((len(mean), 2), 0.5, dtype=torch.float64)

    def fit(self, train_reports, report_days, *, origins=None, covariates=None, tree_prior=None):
        reports = np.asarray(train_reports, dtype=np.float64)
        days = np.asarray(report_days, dtype=np.float64)
        if (reports.ndim != 3 or reports.shape[-1] != 3 or not all(reports.shape)
                or days.shape != (reports.shape[1],) or len(days) < 3
                or not np.isfinite(days).all() or days[0] != 0 or (np.diff(days) <= 0).any()
                or np.isinf(reports).any()):
            raise ValueError("Require finite increasing canonical days starting at zero and reports [well,time,3], NaN allowed")
        if np.any(np.isfinite(reports[:, 0]) & (reports[:, 0] != 0)):
            raise ValueError("Day zero is the deterministic zero accounting origin")
        if self.earlier_history_features != "none" and not np.array_equal(days, EARLIER_REPORT_DAYS):
            raise ValueError("Earlier history features require the frozen eight-endpoint report grid")
        reports = reports.copy()
        reports[:, 0] = 0.0
        self.origins_ = np.asarray(days[1:-1] if origins is None else origins, dtype=float)
        if (self.origins_.ndim != 1 or not len(self.origins_) or not np.isin(self.origins_, days[:-1]).all()
                or len(np.unique(self.origins_)) != len(self.origins_)):
            raise ValueError("Origins must be distinct report days with a future endpoint")
        self.cells_ = [(float(s), float(t)) for s in self.origins_ for t in days if t > s]
        indices = {float(day): index for index, day in enumerate(days)}
        with np.errstate(over="raise", invalid="ignore"):
            increments = np.stack([reports[:, indices[t]]-reports[:, indices[s]] for s, t in self.cells_], axis=1)
        scales, scale_details = fit_increment_scales(increments, return_details=True)
        counts = np.isfinite(increments).sum(axis=0)
        if (counts == 0).any():
            raise ValueError("Every frozen cell/phase requires at least one measured training increment")
        phase_scales, phase_sources = [], []
        for phase in range(3):
            positive = reports[:, -1, phase]
            positive = positive[np.isfinite(positive) & (positive > 0)]
            source = "positive_terminal_train_report_median"
            if not len(positive):
                positive = reports[:, :, phase]
                positive = positive[np.isfinite(positive) & (positive > 0)]
                source = "pooled_positive_train_report_median"
            phase_scales.append(float(np.median(positive)) if len(positive) else 1.0)
            phase_sources.append(source if len(positive) else "one_native_unit_fallback")
        phase_scales = np.asarray(phase_scales)
        if not np.all(np.isfinite(phase_scales) & (phase_scales > 0)):
            raise ValueError("Phase scales must be finite and positive")
        covariate_features = self._covariates(covariates, len(reports), fit=True)
        tree_logs, tree_features = self._tree_prior_inputs(tree_prior, len(reports), phase_scales, fit=True)
        self._initialize(days, phase_scales, scales)
        self.scale_provenance_ = {"fit_split": "train", "phase_scales": phase_scales.tolist(),
                                  "phase_scale_sources": phase_sources, "increment_scale_details": scale_details,
                                  "cells_origin_endpoint_days": self.cells_}
        normalized = reports / phase_scales
        if not np.isfinite(normalized[np.isfinite(reports)]).all():
            raise ValueError("Training normalization exceeded floating-point support")
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=self.device)
        y, times = tensor(normalized), tensor(days)
        covariates_tensor = tensor(covariate_features) if self.covariate_feature_count_ else None
        tree_log_tensor = None if tree_logs is None else tensor(tree_logs)
        tree_feature_tensor = None if tree_features is None else tensor(tree_features)
        native_reports = (torch.as_tensor(reports, dtype=torch.float64, device=self.device)
                          if self.history_rate_features != "none" else None)
        cell_scales, count_weights = tensor(scales / phase_scales), tensor(len(reports) / counts)
        if (bool(torch.isinf(y).any()) or not bool(torch.isfinite(cell_scales).all())
                or bool((cell_scales <= 0).any())):
            raise ValueError("Training inputs cannot be represented in float32")
        measured = torch.isfinite(y)
        context_rng = np.random.default_rng(self.seed + 101)
        mask_rng = np.random.default_rng(self.seed + 1009)
        latent_rng = torch.Generator(device=self.device).manual_seed(self.seed + 2017)
        report_rng = torch.Generator(device=self.device).manual_seed(self.seed + 3001)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        history = []
        total_updates = 0
        context_hashes = []
        per_origin_count = np.array([int((days > s).sum()) for s in self.origins_])
        mean_count = float(per_origin_count.mean())
        self.train()
        for epoch in range(self.epochs):
            batches = []
            for oi in range(len(self.origins_)):
                order = context_rng.permutation(len(reports))
                batches.extend((oi, order[start:start+self.batch_size]) for start in range(0, len(reports), self.batch_size))
            totals = np.zeros(4 if self.mean_score_mode == "none" else 5)
            clipped, largest_gradient = 0, 0.0
            digest = hashlib.sha256()
            for bi in context_rng.permutation(len(batches)):
                oi, rows = batches[bi]
                origin = self.origins_[oi]
                selected = torch.as_tensor(rows, device=self.device)
                actual = y[selected]
                observed = measured[selected] & (times <= origin)[None, :, None]
                keep_report = mask_rng.random((len(rows), len(days), 3)) >= self.history_dropout
                keep_phase = mask_rng.random((len(rows), 1, 3)) >= self.phase_dropout
                observed &= torch.as_tensor(keep_report & keep_phase, device=self.device)
                observed[:, 0] = True
                digest.update(np.asarray([oi], dtype=np.int64).tobytes())
                digest.update(rows.astype(np.int64).tobytes())
                digest.update(observed.detach().cpu().numpy().tobytes())
                context = self._encoder_context(actual, observed, times, origin,
                    native_reports=None if native_reports is None else native_reports[selected],
                    covariates=None if covariates_tensor is None else covariates_tensor[selected])
                if self.mixture_mode == "single":
                    if tree_feature_tensor is None:
                        mean, lower = self._network_parameters(context, observed, times)
                    else:
                        context = torch.cat((context, tree_feature_tensor[selected]), dim=-1)
                        mean, lower = self._network_parameters(context, observed, times, tree_log_prior=tree_log_tensor[selected])
                else:
                    means, lower, weights = self._mixture_network_parameters(context)
                epsilon = torch.randn((len(rows), self.train_samples, self.latent_size),
                                      generator=latent_rng, dtype=torch.float32, device=self.device)
                if self.mixture_mode == "single":
                    z = mean[:, None] + torch.einsum("nij,nsj->nsi", lower, epsilon)
                else:
                    stratified_mean = means.repeat_interleave(self.train_samples // 2, dim=1)
                    z = stratified_mean + torch.einsum("nij,nsj->nsi", lower, epsilon)
                cumulative = self.rate_model.cumulative(z, times)
                context_reports = torch.where(observed, actual, torch.full_like(actual, float("nan")))
                noise_abs, noise_rel, _ = self.effective_noise(context)
                draws = self._sample_reports(cumulative, context_reports, times,
                    noise_abs, noise_rel, cutoff=float(origin), generator=report_rng)
                cell = [i for i, (s, _) in enumerate(self.cells_) if s == origin]
                if self.mixture_mode == "single":
                    if self.energy_samples is None:
                        marginal, energy, dimension = _u_scores(actual, draws, observed, times, indices[float(origin)],
                                                                cell_scales[cell], count_weights[cell])
                    else:
                        marginal, energy, dimension = _sampled_bank_scores(actual, draws, observed, times,
                            indices[float(origin)], cell_scales[cell], count_weights[cell],
                            marginal_objective=self.marginal_objective)
                    if self.marginal_score_estimator == "gaussian_report_integrated":
                        marginal = accumulated_increment_u_crps(actual, cumulative, times, indices[float(origin)],
                            self.noise_abs, self.noise_rel, cell_scales[cell], count_weights[cell])
                else:
                    stratified_draws = draws.reshape(len(rows), 2, self.train_samples // 2, len(days), 3)
                    marginal, energy, dimension = _stratified_scores(actual, stratified_draws, weights, observed,
                        times, indices[float(origin)], cell_scales[cell], count_weights[cell])
                marginal = marginal * (per_origin_count[oi] / mean_count)
                penalty = 0.5 * self.noise_regularization * (
                    torch.log(self.noise_abs / self.initial_noise_abs).square().sum()
                    + torch.log(self.noise_rel / self.initial_noise_rel).square().sum())
                loss = (marginal + energy).mean() + penalty
                if self.mean_score_mode != "none":
                    mean_score = fast_basis_increment_mean_mse(actual, mean, lower, times, indices[float(origin)],
                        cell_scales[cell], count_weights[cell]) * (per_origin_count[oi] / mean_count)
                    loss = loss + mean_score.mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite conditional proper-score objective at epoch {epoch+1}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                norm = float(gradient)
                clipped += norm > self.gradient_clip
                largest_gradient = max(largest_gradient, norm)
                components = [float(loss.detach()), float(marginal.mean().detach()),
                              float(energy.mean().detach()), float(penalty.detach())]
                if self.mean_score_mode != "none":
                    components.append(float(mean_score.mean().detach()))
                totals += np.array(components) * len(rows)
                total_updates += 1
            totals /= len(reports) * len(self.origins_)
            entry = dict(epoch=epoch+1, loss=float(totals[0]), macro_increment_crps=float(totals[1]),
                         joint_energy=float(totals[2]), noise_penalty=float(totals[3]),
                         max_gradient_norm=largest_gradient, clipped_updates=int(clipped))
            if self.mean_score_mode != "none":
                entry.update(macro_increment_mean_mse=float(totals[4]), mean_score_mode=self.mean_score_mode)
            if self.covariance_mode != "bounded_cholesky":
                entry["covariance_mode"] = self.covariance_mode
            if self.energy_samples is not None:
                entry.update(marginal_objective=self.marginal_objective, energy_samples=self.energy_samples)
                if self.marginal_objective == "empirical_wis":
                    entry["macro_increment_wis"] = entry.pop("macro_increment_crps")
            history.append(entry)
            context_hashes.append(digest.hexdigest())
            if self.log_every and ((epoch+1) % self.log_every == 0 or epoch+1 == self.epochs):
                print(json.dumps({"event": "conditional_rate_epoch", "family": self.rate_family,
                                  "seed": self.seed, **entry}), flush=True)
        self.eval()
        self.fit_diagnostics_ = {
            "interpretation": INTERPRETATION, "fit_split": "train", "training_wells": len(reports),
            "origins_days": self.origins_.tolist(), "cells_origin_endpoint_days": self.cells_,
            "increment_scales_native": scales.tolist(), "increment_observation_counts": counts.tolist(),
            "phase_scales_native": phase_scales.tolist(), "phase_order": ["oil", "gas", "water"],
            "epoch_semantics": "every_training_well_at_every_origin_once",
            "contexts_per_epoch": len(reports)*len(self.origins_), "optimizer_updates": total_updates,
            "total_training_contexts": len(reports)*len(self.origins_)*self.epochs,
            "context_mask_sha256_by_epoch": context_hashes,
            "history_rate_features": self.history_rate_features,
            "initial_parameter_sha256": self.initial_parameter_sha256_,
            "rng_seeds": {"network": self.seed, "context_order": self.seed+101, "masks": self.seed+1009,
                          "latent": self.seed+2017, "report": self.seed+3001},
            "objective": "macro_scaled_increment_crps_u + mean_origin_report_energy_u/sqrt(d)",
            "score_coefficients": [1.0, 1.0], "train_samples_iid": self.train_samples,
            "uniform_origin_marginal_weights": (per_origin_count/mean_count).tolist(),
            "missing_target_weighting": "inverse frozen training observation fraction per cell/phase",
            "noise_abs_scaled": self.noise_abs.detach().cpu().tolist(),
            "noise_rel": self.noise_rel.detach().cpu().tolist(), "optimization": history,
            "configuration": self.configuration,
        }
        if self.n_covariates_ or self.covariate_preprocessing != "internal":
            self.fit_diagnostics_.update(covariate_preprocessing=self.covariate_preprocessing,
                covariate_count=self.n_covariates_, covariate_feature_count=self.covariate_feature_count_)
        if self.mean_score_mode != "none":
            self.fit_diagnostics_.update(**self._mean_score_metadata(), score_coefficients=[1.0, 1.0, 1.0],
                objective="macro_scaled_increment_crps_u + mean_origin_report_energy_u/sqrt(d) + macro_scaled_analytic_increment_mean_mse")
        self.fit_diagnostics_.update(self._covariance_metadata())
        if self.report_scale_mode != "global":
            self.fit_diagnostics_.update(**self._report_scale_metadata(),
                report_scale_head_sha256=self._report_scale_head_sha256(),
                report_scale_base_raw_sha256=self._report_scale_base_raw_sha256(),
                common_initial_parameter_sha256=self.common_initial_parameter_sha256_)
        if self.marginal_score_estimator != "sampled":
            self.fit_diagnostics_.update(marginal_score_estimator=self.marginal_score_estimator,
                marginal_score_estimator_meaning=REPORT_INTEGRATED_SCORE_MEANING)
        if self.energy_samples is not None:
            objective = ("macro_scaled_increment_crps_sorted_u_256" if self.marginal_objective == "crps"
                         else "macro_scaled_increment_empirical_wis_256")
            self.fit_diagnostics_.update(**self._training_score_metadata(),
                objective=objective + " + mean_origin_report_energy_u_first16/sqrt(d)")
        if self.earlier_history_features != "none":
            self.fit_diagnostics_.update(**self._earlier_history_metadata(),
                common_initial_parameter_sha256=self.common_initial_parameter_sha256_)
        if self.report_half_life_days is not None:
            self.fit_diagnostics_.update(observation_law=self.observation_law,
                                        report_half_life_days=self.report_half_life_days)
        if self.history_head_mode != "shared":
            self.fit_diagnostics_.update(history_head_mode=self.history_head_mode,
                common_initial_parameter_sha256=self.common_initial_parameter_sha256_,
                initial_output_head_sha256=self.initial_output_head_sha256_,
                no_history_head_initial_sha256=self.no_history_head_initial_sha256_,
                history_head_routing="no permitted observed positive-time report after cutoff and masking; observed zero counts as data",
                history_context_canonicalization=False)
        if self.coefficient_parameterization != "log_location":
            self.fit_diagnostics_.update(coefficient_parameterization=self.coefficient_parameterization,
                coefficient_initial_variance=self.coefficient_initial_variance.detach().cpu().tolist(),
                coefficient_mean_formula="E[exp(z)|context]=exp(base_head+0.5*coefficient_initial_variance)",
                coefficient_variance_reference="Actual initialized Cholesky row squared norms on training device; frozen buffer")
        if self.tree_static_mode != "none":
            self.fit_diagnostics_.update(tree_static_mode=self.tree_static_mode,
                tree_prior_feature_count=12, tree_prior_training_sha256=self.tree_prior_training_sha256_,
                tree_prior_training_hash_encoding="SHA256 of C-order little-endian float64 native [N,12] coefficient bytes; no shape prefix",
                tree_log_prior_center=self.tree_log_prior_center.cpu().tolist(),
                tree_log_prior_scale=self.tree_log_prior_scale.cpu().tolist(),
                tree_prior_fit_source="supplied out-of-fold TRAIN coefficient means; provenance bound by runner",
                tree_prior_units=["BBL"]*4+["MCF"]*4+["BBL"]*4,
                tree_empty_history_mean="E[exp(z)]=native tree coefficients / phase scales, regardless of full covariance",
                common_initial_parameter_sha256=self.common_initial_parameter_sha256_,
                tree_extra_initial_columns="12 exact-zero columns; original initialization RNG unchanged")
        if self.mixture_mode != "single":
            self.fit_diagnostics_.pop("train_samples_iid")
            self.fit_diagnostics_.update({
                "mixture_mode": self.mixture_mode,
                "mixture_components": 2,
                "mixture_covariance": "shared_bounded_full_cholesky",
                "component_scope": "one_component_for_entire_joint_latent_trajectory",
                "initial_component_delta": MIXTURE_INITIAL_DELTA if self.mixture_mode == "two" else 0.0,
                "initial_component_weights": [0.5, 0.5],
                "train_samples_total": self.train_samples,
                "train_samples_independent_per_component": self.train_samples // 2,
                "objective": "macro_scaled_increment_crps_stratified_u + mean_origin_report_energy_stratified_u/sqrt(d)",
                "training_component_sampling": "none_analytic_differentiable_probability_weights",
                "prediction_rng": "three_separate_numpy_child_streams_latent_report_component",
                "mixture_probability_dtype": "float64",
            })
            if self.mixture_mode == "two":
                self.fit_diagnostics_["rng_seeds"]["mixture_heads"] = self.seed + 4001
        self.training_diagnostics_ = self.fit_diagnostics_
        return self

    def _history(self, history, origin):
        if not hasattr(self, "fit_diagnostics_"):
            raise RuntimeError("Fit the conditional rate model before forecasting")
        reports = (np.asarray(history, dtype=float) if self.report_scale_mode == "global"
                   else np.asarray(history, dtype=object))
        days = self.report_days.cpu().numpy()
        if reports.ndim != 3 or not len(reports) or reports.shape[1:] != (len(days), 3):
            raise ValueError("History must have shape [nonempty well,report day,3]")
        if origin not in self.origins_:
            raise ValueError("Forecast origin was not included in training")
        reports = reports.copy()
        reports[:, days > origin] = np.nan  # Before any value validation or feature construction.
        reports = np.asarray(reports, dtype=float)
        if np.isinf(reports).any():
            raise ValueError("Available history must be finite or NaN, never infinite")
        if np.any(np.isfinite(reports[:, 0]) & (reports[:, 0] != 0)):
            raise ValueError("Day zero is deterministic zero")
        reports[:, 0] = 0.0
        return reports

    def conditional_parameters(self, history, origin, covariates=None, tree_prior=None):
        if self.mixture_mode != "single":
            raise ValueError("A mixture is not one Gaussian; use conditional_mixture_parameters")
        return self._conditional_parameters(history, origin, covariates, tree_prior)

    def conditional_mixture_parameters(self, history, origin, covariates=None, tree_prior=None):
        """Return joint component means, shared Cholesky, and probabilities.

        A single-Gaussian model is represented by one component here. The
        existing conditional_parameters method retains its original two-value
        Gaussian API and never silently moment-matches a mixture.
        """
        if self.mixture_mode == "single":
            mean, lower = self._conditional_parameters(history, origin, covariates, tree_prior)
            return mean[:, None], lower, mean.new_ones((len(mean), 1))
        return self._conditional_parameters(history, origin, covariates, tree_prior)

    def _conditional_parameters(self, history, origin, covariates=None, tree_prior=None, *, return_noise=False):
        reports = self._history(history, origin)
        covariate_features = self._covariates(covariates, len(reports), fit=False)
        tree_logs, tree_features = self._tree_prior_inputs(tree_prior, len(reports), self.volume_scale.cpu().numpy(), fit=False)
        normalized = reports / self.volume_scale.cpu().numpy()
        if not np.isfinite(normalized[np.isfinite(reports)]).all():
            raise ValueError("History normalization is not representable")
        values = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        if bool(torch.isinf(values).any()):
            raise ValueError("History is not representable in float32")
        times = self.report_days.to(torch.float32)
        native_reports = (torch.as_tensor(reports, dtype=torch.float64, device=self.device)
                          if self.history_rate_features != "none" else None)
        covariates_tensor = (torch.as_tensor(covariate_features, dtype=torch.float32, device=self.device)
                             if self.covariate_feature_count_ else None)
        with torch.no_grad():
            context = self._encoder_context(values, torch.isfinite(values), times, origin,
                                            native_reports=native_reports, covariates=covariates_tensor)
            if self.mixture_mode == "single":
                if tree_features is None:
                    mean, lower = self._network_parameters(context, torch.isfinite(values), times)
                else:
                    context = torch.cat((context, torch.as_tensor(tree_features, device=self.device)), dim=-1)
                    mean, lower = self._network_parameters(context, torch.isfinite(values), times,
                        tree_log_prior=torch.as_tensor(tree_logs, device=self.device))
                if not bool(torch.isfinite(mean).all() & torch.isfinite(lower).all()):
                    raise FloatingPointError("Nonfinite conditional Gaussian parameters")
                if return_noise:
                    a, r, multiplier = self.effective_noise(context, fitted_base=True)
                    return mean.double(), lower.double(), a.double(), r.double(), (None if multiplier is None else multiplier.double())
                return mean.double(), lower.double()
            means, lower, weights = self._mixture_network_parameters(context)
        if not all(bool(torch.isfinite(value).all()) for value in (means, lower, weights)):
            raise FloatingPointError("Nonfinite conditional Gaussian mixture parameters")
        return means.double(), lower.double(), weights.double()

    def evaluate_trajectories(self, latent_parameters, query_days, *, include_rate_derivatives=False):
        """Query a fixed draw; optional basis q'/q'' use native volume/day²/day³.

        Derivatives are model-implied, with right derivatives at day zero.
        Families with knots need a separate boundary convention and do not
        expose this option. The conditional law and sampled weights are fixed.
        """
        if not hasattr(self, "fit_diagnostics_"):
            raise RuntimeError("Fit the model before querying trajectories")
        if include_rate_derivatives and not isinstance(self.rate_model, PositiveBasisRate):
            raise NotImplementedError("Rate derivative queries currently require the positive basis family")
        z = (latent_parameters.to(dtype=torch.float64, device=self.device) if isinstance(latent_parameters, torch.Tensor)
             else torch.tensor(latent_parameters, dtype=torch.float64, device=self.device))
        days = torch.as_tensor(query_days, dtype=torch.float64, device=self.device)
        if (z.shape[-1] != self.latent_size or days.ndim != 1 or not bool(torch.isfinite(days).all())
                or bool((days < 0).any()) or not bool(torch.isfinite(z).all())):
            raise ValueError("Expected finite latent parameters and finite nonnegative query days")
        with torch.no_grad():
            result = {"latent_cumulative": self.rate_model.cumulative(z, days)*self.volume_scale,
                      "latent_rate": self.rate_model.rate(z, days)*self.volume_scale,
                      "latent_total_volume": self.rate_model.total_volume(z)*self.volume_scale}
            if include_rate_derivatives:
                result.update(
                    latent_rate_derivative=self.rate_model.rate_derivative(z, days, order=1)*self.volume_scale,
                    latent_rate_second_derivative=self.rate_model.rate_derivative(z, days, order=2)*self.volume_scale)
        if not all(bool(torch.isfinite(value).all()) for value in result.values()):
            raise FloatingPointError("Latent trajectory query exceeded floating-point support; no clipping")
        return {key: value.cpu().numpy() for key, value in result.items()}

    def _history_discrepancies(self, history, cumulative, *, noise_abs=None, noise_rel=None):
        """Describe latent-history agreement; never treat pinned reports as fit evidence."""
        days = self.report_days.cpu().numpy()
        scale = self.volume_scale.cpu().numpy()
        q = cumulative / scale
        y = history / scale
        if self.report_scale_mode != "global" and (noise_abs is None or noise_rel is None):
            raise ValueError("Contextual history discrepancies require the issued effective noise")
        a = (self.noise_abs if noise_abs is None else noise_abs).detach().cpu().numpy()
        r = (self.noise_rel if noise_rel is None else noise_rel).detach().cpu().numpy()
        rms, med_z, extreme, counts = (np.full((len(y), 3), np.nan), np.full((len(y), 3), np.nan),
                                      np.full((len(y), 3), np.nan), np.zeros((len(y), 3), dtype=int))
        for well in range(len(y)):
            for phase in range(3):
                a_phase = float(a[phase] if a.ndim == 1 else a[well, phase])
                r_phase = float(r[phase] if r.ndim == 1 else r[well, phase])
                observed = np.flatnonzero(np.isfinite(y[well, :, phase]) & (days > 0))
                counts[well, phase] = len(observed)
                if not len(observed):
                    continue
                errors, standardized = [], []
                previous_q = np.zeros(q.shape[1])
                previous_error = np.zeros(q.shape[1])
                previous_time = 0.0
                for j in observed:
                    current_q = q[well, :, j, phase]
                    error = y[well, j, phase]-current_q
                    errors.extend(error.tolist())
                    if self.report_half_life_days is None:
                        variance = (a_phase**2 * (days[j]-previous_time)/360
                                    + r_phase**2 * (current_q-previous_q)*(current_q+previous_q))
                        standardized.extend(((error-previous_error)/np.sqrt(variance)).tolist())
                    else:
                        # Reuse the sampler's exact transition across successive
                        # observed endpoints, including skipped historical reports.
                        rho, variance = fading_transition(
                            torch.as_tensor(previous_q, dtype=torch.float64),
                            torch.as_tensor(current_q, dtype=torch.float64),
                            torch.tensor(previous_time, dtype=torch.float64),
                            torch.tensor(float(days[j]), dtype=torch.float64),
                            torch.tensor(a_phase, dtype=torch.float64),
                            torch.tensor(r_phase, dtype=torch.float64),
                            half_life_days=self.report_half_life_days)
                        standardized.extend(((error-rho.numpy()*previous_error)
                                             / np.sqrt(variance.numpy())).tolist())
                    previous_q, previous_error, previous_time = current_q, error, days[j]
                errors, standardized = np.asarray(errors), np.asarray(standardized)
                if not np.isfinite(errors).all() or not np.isfinite(standardized).all():
                    raise FloatingPointError("Historical latent discrepancy diagnostics exceeded numerical support")
                error_scale = np.max(np.abs(errors))
                rms[well, phase] = error_scale * np.sqrt(np.mean((errors/error_scale)**2)) if error_scale else 0.0
                med_z[well, phase] = np.median(np.abs(standardized))
                extreme[well, phase] = np.mean(np.abs(standardized) > 3)
        def serial(array):
            return [[float(v) if np.isfinite(v) else None for v in row] for row in array]
        result = {"interpretation": INTERPRETATION, "history_observed_counts_by_phase": counts.tolist(),
                "historical_latent_rmse_scaled_by_phase": serial(rms),
                "historical_abs_standardized_innovation_median_by_phase": serial(med_z),
                "historical_abs_standardized_innovation_gt3_fraction_by_phase": serial(extreme),
                "diagnostic_scope": "descriptive latent/report discrepancy, not a likelihood penalty or posterior check",
                "latent_history_identification": "not established by exact repetition of observed reports",
                "cross_cutoff_bayesian_consistency": False}
        if self.report_half_life_days is not None:
            result.update(observation_law=self.observation_law, report_half_life_days=self.report_half_life_days,
                          standardized_innovation_definition="(e_j-rho_ij*e_i)/sqrt(V_j-rho_ij^2*V_i), successive observed endpoints")
        return result

    def predict_distribution(self, history, origin, n_samples, rng, covariates=None, tree_prior=None):
        if self.tree_static_mode == "none" and tree_prior is not None:
            raise ValueError("tree_prior requires explicit tree_static_mode")
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        reports = self._history(history, origin)
        mixture_metadata = {}
        noise_metadata = {}
        if self.mixture_mode == "single":
            if self.report_scale_mode == "global":
                mean, lower = self.conditional_parameters(reports, origin, covariates, tree_prior)
            else:
                mean, lower, noise_abs, noise_rel, multiplier = self._conditional_parameters(
                    reports, origin, covariates, tree_prior, return_noise=True)
                noise_metadata = {**self._report_scale_metadata(),
                    "report_scale_multiplier": multiplier.cpu().numpy(),
                    "noise_abs_scaled": noise_abs.cpu().numpy(), "noise_rel": noise_rel.cpu().numpy()}
            latent_rng = report_rng = rng
        else:
            means, lower, weights = self.conditional_mixture_parameters(reports, origin, covariates)
            stream_seeds = rng.integers(0, np.iinfo(np.int64).max, size=3, dtype=np.int64)
            latent_rng, report_rng, component_rng = [np.random.default_rng(int(seed)) for seed in stream_seeds]
            uniforms = component_rng.random((len(reports), n_samples))
            components = uniforms >= weights[:, 0, None].cpu().numpy()
            component_tensor = torch.as_tensor(components, dtype=torch.long, device=self.device)
            selected_mean = means.gather(1, component_tensor[:, :, None].expand(-1, -1, self.latent_size))
            mixture_metadata = {"mixture_mode": self.mixture_mode,
                                "mixture_component": components.astype(np.int64),
                                "mixture_component_probabilities": weights.cpu().numpy(),
                                "mixture_component_means": means.cpu().numpy(),
                                "component_scope": "one_component_for_entire_joint_latent_trajectory",
                                "prediction_rng_seeds": dict(zip(("latent", "report", "component"), map(int, stream_seeds)))}
        epsilon = torch.as_tensor(latent_rng.standard_normal((len(reports), n_samples, self.latent_size)),
                                  dtype=torch.float64, device=self.device)
        location = mean[:, None] if self.mixture_mode == "single" else selected_mean
        z = location + torch.einsum("nij,nsj->nsi", lower, epsilon)
        paths = self.evaluate_trajectories(z, self.report_days)
        q = torch.as_tensor(paths["latent_cumulative"], dtype=torch.float64, device=self.device) / self.volume_scale
        normalized = torch.as_tensor(reports, dtype=torch.float64, device=self.device) / self.volume_scale
        normals = torch.as_tensor(report_rng.standard_normal(q.shape), dtype=torch.float64, device=self.device)
        if self.report_scale_mode == "global":
            noise_abs, noise_rel = self.noise_abs.double(), self.noise_rel.double()
        with torch.no_grad():
            predicted = self._sample_reports(q, normalized, self.report_days, noise_abs,
                noise_rel, cutoff=float(origin), standard_normals=normals) * self.volume_scale
        report = predicted.cpu().numpy()
        # Restore original native float64 observations, avoiding scaling roundoff.
        report = np.where(np.isfinite(reports)[:, None], reports[:, None], report)
        if not np.isfinite(report).all():
            raise FloatingPointError("Native report conversion exceeded floating-point support; no clipping")
        diagnostics = (self._history_discrepancies(reports, paths["latent_cumulative"])
                       if self.report_scale_mode == "global" else
                       self._history_discrepancies(reports, paths["latent_cumulative"],
                           noise_abs=noise_abs, noise_rel=noise_rel))
        diagnostics.update(self._report_scale_metadata())
        diagnostics.update(self._covariance_metadata())
        score_metadata = ({} if self.marginal_score_estimator == "sampled" else
                          {"marginal_score_estimator": self.marginal_score_estimator})
        score_metadata.update(self._training_score_metadata())
        score_metadata.update(self._mean_score_metadata())
        score_metadata.update(self._covariance_metadata())
        head_metadata = {"history_head_mode": self.history_head_mode}
        coefficient_metadata = {"coefficient_parameterization": self.coefficient_parameterization}
        tree_metadata = {"tree_static_mode": self.tree_static_mode}
        if self.tree_static_mode != "none":
            empty = ~np.any(np.isfinite(reports) & (self.report_days.cpu().numpy() > 0)[None,:,None], axis=(1,2))
            tree_metadata.update(tree_prior_native=np.asarray(tree_prior, dtype=np.float64), tree_empty_history_mask=empty)
            diagnostics.update(tree_static_mode=self.tree_static_mode, tree_empty_history_mask=empty.tolist())
        if self.coefficient_parameterization != "log_location":
            coefficient_metadata["coefficient_initial_variance"] = self.coefficient_initial_variance.detach().cpu().numpy()
            diagnostics["coefficient_parameterization"] = self.coefficient_parameterization
            diagnostics["coefficient_initial_variance"] = self.coefficient_initial_variance.detach().cpu().tolist()
        if self.history_head_mode == "no_history":
            no_history = ~np.any(np.isfinite(reports) & (self.report_days.cpu().numpy() > 0)[None, :, None], axis=(1, 2))
            head_metadata["no_history_head_mask"] = no_history
            diagnostics["history_head_mode"] = self.history_head_mode
            diagnostics["no_history_head_mask"] = no_history.tolist()
            diagnostics["history_context_canonicalization"] = False
        if self.mixture_mode != "single":
            diagnostics["mixture"] = {
                "scope": "descriptive conditional component usage, not identified physical populations",
                "component_probabilities": weights.cpu().tolist(),
                "component_entropy_nats": (-(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(dim=-1)).cpu().tolist(),
                "component_mean_separation_l2": torch.linalg.vector_norm(means[:, 1] - means[:, 0], dim=-1).cpu().tolist(),
            }
        self.last_conditional_diagnostics_ = diagnostics
        return {**paths, **mixture_metadata, **head_metadata, **coefficient_metadata, **tree_metadata, **score_metadata, **noise_metadata,
                **self._earlier_history_metadata(), "latent_parameters": z.cpu().numpy(), "report_cumulative": report,
                "conditional_diagnostics": diagnostics, "interpretation": INTERPRETATION,
                "phase_order": ["oil", "gas", "water"], "observation_law": self.observation_law,
                "report_half_life_days": self.report_half_life_days}

    def predict_samples(self, history, origin, n_samples, rng, covariates=None, tree_prior=None):
        return self.predict_distribution(history, origin, n_samples, rng, covariates, tree_prior)["report_cumulative"]

    def save(self, path):
        if not hasattr(self, "fit_diagnostics_"):
            raise RuntimeError("Fit the model before saving")
        schema_version = 3 if self.covariance_mode != "bounded_cholesky" else (1 if self.mixture_mode == "single" else 2)
        checkpoint = {"schema_version": schema_version, "configuration": self.configuration,
                    "state_dict": {key: value.detach().cpu() for key, value in self.state_dict().items()},
                    "origins": self.origins_.tolist(), "cells": self.cells_,
                    "fit_diagnostics": self.fit_diagnostics_, "scale_provenance": self.scale_provenance_}
        if self.n_covariates_ or self.covariate_preprocessing != "internal":
            checkpoint["covariate_metadata"] = {"n_covariates": self.n_covariates_,
                "feature_count": self.covariate_feature_count_, "center": self.covariate_center_.tolist(),
                "scale": self.covariate_scale_.tolist()}
        if self.marginal_score_estimator != "sampled":
            checkpoint.update(marginal_score_estimator=self.marginal_score_estimator,
                              marginal_score_estimator_meaning=REPORT_INTEGRATED_SCORE_MEANING)
        checkpoint.update(self._training_score_metadata())
        checkpoint.update(self._mean_score_metadata())
        checkpoint.update(self._covariance_metadata())
        if self.report_scale_mode != "global":
            checkpoint.update(**self._report_scale_metadata(),
                report_scale_head_sha256=self._report_scale_head_sha256(),
                report_scale_base_raw_sha256=self._report_scale_base_raw_sha256(),
                common_initial_parameter_sha256=self.common_initial_parameter_sha256_)
        if self.earlier_history_features != "none":
            checkpoint.update(**self._earlier_history_metadata(),
                              common_initial_parameter_sha256=self.common_initial_parameter_sha256_)
        if self.report_half_life_days is not None:
            checkpoint.update(observation_law=self.observation_law,
                              report_half_life_days=self.report_half_life_days)
        if self.history_head_mode != "shared":
            checkpoint.update(history_head_mode=self.history_head_mode,
                common_initial_parameter_sha256=self.common_initial_parameter_sha256_,
                initial_output_head_sha256=self.initial_output_head_sha256_,
                no_history_head_initial_sha256=self.no_history_head_initial_sha256_)
        if self.coefficient_parameterization != "log_location":
            checkpoint.update(coefficient_parameterization=self.coefficient_parameterization,
                coefficient_initial_variance=self.coefficient_initial_variance.detach().cpu().tolist())
        if self.tree_static_mode != "none":
            checkpoint.update(tree_static_mode=self.tree_static_mode, tree_prior_metadata={
                "training_sha256":self.tree_prior_training_sha256_,
                "log_center":self.tree_log_prior_center.cpu().tolist(),
                "log_scale":self.tree_log_prior_scale.cpu().tolist(), "feature_count":12,
                "coefficient_order":"oil[4],gas[4],water[4]; native BBL,MCF,BBL"})
        torch.save(checkpoint, Path(path))

    @classmethod
    def load(cls, path, *, device="cpu"):
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        if checkpoint["schema_version"] not in {1, 2, 3}:
            raise ValueError("Unsupported conditional rate checkpoint")
        covariance_mode = checkpoint["configuration"].get("covariance_mode", "bounded_cholesky")
        expected_schema = (3 if covariance_mode != "bounded_cholesky" else
                           1 if checkpoint["configuration"].get("mixture_mode", "single") == "single" else 2)
        if checkpoint["schema_version"] != expected_schema:
            raise ValueError("Conditional rate checkpoint schema and covariance/mixture mode disagree")
        model = cls(**dict(checkpoint["configuration"], device=device))
        fit_metadata = checkpoint["fit_diagnostics"]
        if any(metadata.get("covariance_mode", "bounded_cholesky") != model.covariance_mode
               for metadata in (checkpoint, fit_metadata, fit_metadata.get("configuration", {}))):
            raise ValueError("Conditional rate checkpoint covariance mode metadata disagree")
        if model.covariance_mode != "bounded_cholesky":
            saved_anchor = checkpoint.get("covariance_diagonal_anchor_logit")
            if (isinstance(saved_anchor, bool) or not isinstance(saved_anchor, (int, float))
                    or not math.isfinite(saved_anchor)):
                raise ValueError("Conditional rate checkpoint lacks a finite covariance anchor")
            if any(metadata.get(key) != value for metadata in (checkpoint, fit_metadata)
                   for key, value in model._covariance_metadata(diagonal_anchor_logit=saved_anchor).items()):
                raise ValueError("Conditional rate checkpoint covariance support/arithmetic metadata disagree")
            if (fit_metadata.get("objective") != "macro_scaled_increment_crps_u + mean_origin_report_energy_u/sqrt(d)"
                    or fit_metadata.get("score_coefficients") != [1.0, 1.0]
                    or fit_metadata.get("train_samples_iid") != 16
                    or fit_metadata.get("interpretation") != INTERPRETATION
                    or len(fit_metadata.get("optimization", [])) != model.epochs
                    or any(row.get("covariance_mode") != model.covariance_mode
                           for row in fit_metadata.get("optimization", []))):
                raise ValueError("Conditional rate checkpoint covariance training metadata disagree")
        elif any(key.startswith("covariance_") and key != "covariance_mode"
                 for metadata in (checkpoint, fit_metadata, *fit_metadata.get("optimization", [])) for key in metadata):
            raise ValueError("Default checkpoint cannot carry marginal covariance metadata")
        elif any(row.get("covariance_mode", "bounded_cholesky") != "bounded_cholesky"
                 for row in fit_metadata.get("optimization", [])):
            raise ValueError("Default checkpoint contradicts covariance optimization metadata")
        if any(metadata.get("mean_score_mode", "none") != model.mean_score_mode
               for metadata in (checkpoint, fit_metadata, fit_metadata.get("configuration", {}))):
            raise ValueError("Conditional rate checkpoint mean score mode metadata disagree")
        expected_mean_score = model._mean_score_metadata()
        if model.mean_score_mode != "none":
            if any(metadata.get(key) != value for metadata in (checkpoint, fit_metadata)
                   for key, value in expected_mean_score.items()):
                raise ValueError("Conditional rate checkpoint mean score coefficient/meaning metadata disagree")
            if (fit_metadata.get("objective") != "macro_scaled_increment_crps_u + mean_origin_report_energy_u/sqrt(d) + macro_scaled_analytic_increment_mean_mse"
                    or fit_metadata.get("score_coefficients") != [1.0, 1.0, 1.0]
                    or fit_metadata.get("train_samples_iid") != 16
                    or fit_metadata.get("interpretation") != INTERPRETATION):
                raise ValueError("Conditional rate checkpoint analytic mean objective metadata disagree")
            if any(row.get("mean_score_mode") != "analytic_mse"
                    or not isinstance(row.get("macro_increment_mean_mse"), (int, float))
                    or not math.isfinite(row["macro_increment_mean_mse"])
                    or row["macro_increment_mean_mse"] < 0 for row in fit_metadata.get("optimization", [])):
                raise ValueError("Conditional rate checkpoint analytic mean optimization metadata disagree")
        elif any(key.startswith("mean_score_") and key != "mean_score_mode"
                 for metadata in (checkpoint, fit_metadata) for key in metadata):
            raise ValueError("Default checkpoint cannot carry analytic mean score metadata")
        elif ("macro_scaled_analytic_increment_mean_mse" in fit_metadata.get("objective", "")
                or fit_metadata.get("score_coefficients") == [1.0, 1.0, 1.0]
                or any("macro_increment_mean_mse" in row or any(key.startswith("mean_score_") for key in row)
                       for row in fit_metadata.get("optimization", []))):
            raise ValueError("Default checkpoint contradicts analytic mean objective/log metadata")
        if any(metadata.get("report_scale_mode", "global") != model.report_scale_mode
               for metadata in (checkpoint, fit_metadata, fit_metadata.get("configuration", {}))):
            raise ValueError("Conditional rate checkpoint report scale mode metadata disagree")
        scale_metadata = model._report_scale_metadata()
        if model.report_scale_mode != "global":
            if any(metadata.get(key) != value for metadata in (checkpoint, fit_metadata)
                   for key, value in scale_metadata.items()):
                raise ValueError("Conditional rate checkpoint report scale meaning/bounds metadata disagree")
            for key in ("common_initial_parameter_sha256", "report_scale_head_sha256", "report_scale_base_raw_sha256"):
                digest = checkpoint.get(key)
                if (not isinstance(digest, str) or len(digest) != 64
                        or any(letter not in "0123456789abcdef" for letter in digest)
                        or fit_metadata.get(key) != digest):
                    raise ValueError("Conditional rate checkpoint report scale hash metadata disagree")
            if (fit_metadata.get("interpretation") != INTERPRETATION
                    or fit_metadata.get("train_samples_iid") != 16):
                raise ValueError("Conditional rate checkpoint contextual scale protocol metadata disagree")
        elif any(key.startswith("report_scale_") and key != "report_scale_mode"
                 for metadata in (checkpoint, fit_metadata) for key in metadata):
            raise ValueError("Global report checkpoint cannot carry contextual scale metadata")
        if any(metadata.get("marginal_objective", "crps") != model.marginal_objective
               or metadata.get("energy_samples") != model.energy_samples
               for metadata in (checkpoint, fit_metadata, fit_metadata.get("configuration", {}))):
            raise ValueError("Conditional rate checkpoint marginal objective/energy metadata disagree")
        expected_training_score = model._training_score_metadata()
        for metadata in (checkpoint, fit_metadata):
            if any(metadata.get(key) != expected_training_score.get(key)
                   for key in ("marginal_objective_meaning", "energy_sample_selection")):
                raise ValueError("Conditional rate checkpoint marginal objective meaning metadata disagree")
        if model.energy_samples is not None:
            expected_objective = ("macro_scaled_increment_crps_sorted_u_256" if model.marginal_objective == "crps"
                                  else "macro_scaled_increment_empirical_wis_256")
            if (fit_metadata.get("objective") != expected_objective + " + mean_origin_report_energy_u_first16/sqrt(d)"
                    or fit_metadata.get("train_samples_iid") != 256):
                raise ValueError("Conditional rate checkpoint fixed-bank objective/sample count disagree")
            score_key = "macro_increment_crps" if model.marginal_objective == "crps" else "macro_increment_wis"
            other_key = "macro_increment_wis" if model.marginal_objective == "crps" else "macro_increment_crps"
            if any(row.get("marginal_objective") != model.marginal_objective or row.get("energy_samples") != 16
                   or score_key not in row or other_key in row for row in fit_metadata.get("optimization", [])):
                raise ValueError("Conditional rate checkpoint optimization score labels disagree")
        if (checkpoint.get("marginal_score_estimator", "sampled") != model.marginal_score_estimator
                or fit_metadata.get("marginal_score_estimator", "sampled") != model.marginal_score_estimator
                or fit_metadata.get("configuration", {}).get("marginal_score_estimator", "sampled") != model.marginal_score_estimator):
            raise ValueError("Conditional rate checkpoint marginal score estimator metadata disagree")
        expected_score_meaning = (None if model.marginal_score_estimator == "sampled"
                                  else REPORT_INTEGRATED_SCORE_MEANING)
        if any(metadata.get("marginal_score_estimator_meaning") != expected_score_meaning
               for metadata in (checkpoint, fit_metadata)):
            raise ValueError("Conditional rate checkpoint marginal score meaning metadata disagree")
        if (checkpoint.get("earlier_history_features", "none") != model.earlier_history_features
                or fit_metadata.get("earlier_history_features", "none") != model.earlier_history_features
                or fit_metadata.get("configuration", {}).get("earlier_history_features", "none") != model.earlier_history_features):
            raise ValueError("Conditional rate checkpoint earlier history mode metadata disagree")
        earlier_metadata = model._earlier_history_metadata()
        if model.earlier_history_features != "none":
            if any(metadata.get(key) != value for metadata in (checkpoint, fit_metadata)
                   for key, value in earlier_metadata.items()):
                raise ValueError("Conditional rate checkpoint earlier history layout metadata disagree")
            digest = checkpoint.get("common_initial_parameter_sha256")
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(letter not in "0123456789abcdef" for letter in digest)
                    or fit_metadata.get("common_initial_parameter_sha256") != digest):
                raise ValueError("Conditional rate checkpoint earlier history initialization metadata disagree")
        elif any(key.startswith("earlier_history_feature_") for metadata in (checkpoint, fit_metadata) for key in metadata):
            raise ValueError("Default checkpoint cannot carry an earlier history feature layout")
        if (checkpoint.get("tree_static_mode", "none") != model.tree_static_mode
                or checkpoint["fit_diagnostics"].get("tree_static_mode", "none") != model.tree_static_mode):
            raise ValueError("Conditional rate checkpoint tree mode metadata disagree")
        if (checkpoint.get("coefficient_parameterization", "log_location") != model.coefficient_parameterization
                or checkpoint["fit_diagnostics"].get("coefficient_parameterization", "log_location") != model.coefficient_parameterization):
            raise ValueError("Conditional rate checkpoint coefficient parameterization metadata disagree")
        if (checkpoint.get("history_head_mode", "shared") != model.history_head_mode
                or checkpoint["fit_diagnostics"].get("history_head_mode", "shared") != model.history_head_mode):
            raise ValueError("Conditional rate checkpoint history head metadata disagree")
        if model.history_head_mode != "shared":
            for key in ("common_initial_parameter_sha256", "initial_output_head_sha256", "no_history_head_initial_sha256"):
                value = checkpoint.get(key)
                if (not isinstance(value, str) or len(value) != 64
                        or value != checkpoint["fit_diagnostics"].get(key)):
                    raise ValueError("Conditional rate checkpoint history head initialization metadata disagree")
            if checkpoint["initial_output_head_sha256"] != checkpoint["no_history_head_initial_sha256"]:
                raise ValueError("Conditional rate checkpoint history head was not initialized as a clone")
        if (checkpoint.get("observation_law", "conditional_gaussian_accumulated_bridge") != model.observation_law
                or checkpoint.get("report_half_life_days") != model.report_half_life_days
                or checkpoint["fit_diagnostics"].get("observation_law", "conditional_gaussian_accumulated_bridge") != model.observation_law
                or checkpoint["fit_diagnostics"].get("report_half_life_days") != model.report_half_life_days):
            raise ValueError("Conditional rate checkpoint report law and half-life metadata disagree")
        state = checkpoint["state_dict"]
        covariance_anchor = state.get("network.covariance_diagonal_anchor_logit")
        if model.covariance_mode != "bounded_cholesky":
            if (not isinstance(covariance_anchor, torch.Tensor) or covariance_anchor.dtype != torch.float64
                    or covariance_anchor.shape != () or covariance_anchor.requires_grad
                    or not bool(torch.isfinite(covariance_anchor))
                    or covariance_anchor.item() != saved_anchor):
                raise ValueError("Checkpoint covariance anchor buffer and recorded metadata disagree")
        elif covariance_anchor is not None:
            raise ValueError("Default checkpoint cannot carry a covariance anchor buffer")
        tree_metadata = checkpoint.get("tree_prior_metadata")
        if model.tree_static_mode != "none":
            if (not isinstance(tree_metadata, dict) or tree_metadata.get("feature_count") != 12
                    or tree_metadata.get("coefficient_order") != "oil[4],gas[4],water[4]; native BBL,MCF,BBL"
                    or checkpoint["fit_diagnostics"].get("tree_prior_feature_count") != 12):
                raise ValueError("Checkpoint tree prior metadata missing")
            digest = tree_metadata.get("training_sha256")
            if not isinstance(digest, str) or len(digest) != 64 or digest != checkpoint["fit_diagnostics"].get("tree_prior_training_sha256"):
                raise ValueError("Checkpoint tree training hash mismatch")
            for key, field in (("tree_log_prior_center", "log_center"), ("tree_log_prior_scale", "log_scale")):
                value = state.get(key)
                if (not isinstance(value, torch.Tensor) or value.dtype != torch.float64 or value.shape != (12,)
                        or not bool(torch.isfinite(value).all()) or (field == "log_scale" and bool((value < 1e-8).any()))
                        or not np.array_equal(value.numpy(), tree_metadata.get(field))
                        or not np.array_equal(value.numpy(), checkpoint["fit_diagnostics"].get(key))):
                    raise ValueError("Checkpoint tree log preprocessing mismatch")
        elif tree_metadata is not None or "tree_log_prior_center" in state or "tree_log_prior_scale" in state:
            raise ValueError("Default checkpoint cannot contain tree prior preprocessing")
        reference = state.get("coefficient_initial_variance")
        if model.coefficient_parameterization == "log_expectation":
            if (not isinstance(reference, torch.Tensor) or reference.dtype != torch.float32
                    or reference.shape != (12,) or not bool(torch.isfinite(reference).all())
                    or bool((reference <= 0).any())):
                raise ValueError("Invalid checkpoint coefficient initial variance buffer")
            for metadata in (checkpoint, checkpoint["fit_diagnostics"]):
                saved = np.asarray(metadata.get("coefficient_initial_variance"), dtype=np.float64)
                if saved.shape != (12,) or not np.array_equal(saved, reference.numpy()):
                    raise ValueError("Checkpoint coefficient initial variance metadata disagree")
        elif reference is not None or "coefficient_initial_variance" in checkpoint or "coefficient_initial_variance" in checkpoint["fit_diagnostics"]:
            raise ValueError("Default coefficient parameterization cannot carry a variance reference")
        metadata = checkpoint.get("covariate_metadata")
        if metadata is not None:
            n, width = metadata["n_covariates"], metadata["feature_count"]
            if (isinstance(n, bool) or not isinstance(n, int) or n < 0
                    or isinstance(width, bool) or not isinstance(width, int)
                    or width != n * (2 if model.covariate_preprocessing == "internal" else 1)):
                raise ValueError("Invalid conditional rate checkpoint covariate dimensions")
            center, scale = np.asarray(metadata["center"], dtype=float), np.asarray(metadata["scale"], dtype=float)
            expected_shape = (n,) if model.covariate_preprocessing == "internal" else (0,)
            if (center.shape != expected_shape or scale.shape != expected_shape
                    or not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale <= 0).any()):
                raise ValueError("Invalid conditional rate checkpoint covariate preprocessing")
            model.n_covariates_, model.covariate_feature_count_ = n, width
            model.covariate_center_, model.covariate_scale_ = center, scale
        elif model.covariate_preprocessing == "external":
            raise ValueError("External conditional rate checkpoint lacks covariate metadata")
        expected_input = 8 * len(state["report_days"]) + 1 + model.covariate_feature_count_
        if model.history_rate_features != "none":
            expected_input += 6
        if model.tree_static_mode != "none":
            expected_input += 12
        if model.earlier_history_features != "none":
            expected_input += EARLIER_FEATURE_COUNT
        if state["network.hidden.0.weight"].shape[1] != expected_input:
            raise ValueError("Conditional rate checkpoint network and covariate dimensions disagree")
        head = state.get("report_scale_head.weight")
        if model.report_scale_mode != "global":
            if (not isinstance(head, torch.Tensor) or head.dtype != torch.float32
                    or head.shape != (3, expected_input) or not bool(torch.isfinite(head).all())
                    or hashlib.sha256(head.contiguous().numpy().tobytes()).hexdigest()
                       != checkpoint["report_scale_head_sha256"]):
                raise ValueError("Invalid contextual report scale checkpoint head")
            for raw_key, fit_key in (("noise_abs_raw", "noise_abs_scaled"), ("noise_rel_raw", "noise_rel")):
                value = state.get(raw_key)
                recorded = np.asarray(fit_metadata.get(fit_key))
                if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.shape != (3,)
                        or not bool(torch.isfinite(value).all())
                        or recorded.shape != (3,) or not np.isfinite(recorded).all() or np.any(recorded <= 0)
                        or not np.array_equal(recorded.astype(np.float32).astype(np.float64), recorded)
                        or not np.allclose((torch.nn.functional.softplus(value)+model.noise_floor).numpy(),
                                           recorded, atol=0, rtol=4*torch.finfo(torch.float32).eps)):
                    raise ValueError("Contextual report checkpoint global base noise metadata disagree")
        elif head is not None:
            raise ValueError("Global report checkpoint cannot contain a contextual head")
        model._initialize(state["report_days"], state["volume_scale"], state["increment_scales"])
        if (model.covariance_mode != "bounded_cholesky"
                and model.initial_parameter_sha256_ != fit_metadata.get("initial_parameter_sha256")):
            raise ValueError("Checkpoint covariance initialization differs from configured parameters")
        if model.covariance_mode != "bounded_cholesky":
            # Cross-device legacy kernels can produce a slightly different
            # initialization anchor. Validate that discrepancy at float32
            # precision, then load the STORED float64 scalar unchanged below.
            if not torch.allclose(covariance_anchor.to(model.device), model.network.covariance_diagonal_anchor_logit,
                                  atol=0, rtol=4*torch.finfo(torch.float32).eps):
                raise ValueError("Checkpoint covariance anchor differs from the legacy initialization")
        if model.report_scale_mode != "global" and (
                model.common_initial_parameter_sha256_ != checkpoint["common_initial_parameter_sha256"]
                or model.initial_parameter_sha256_ != fit_metadata.get("initial_parameter_sha256")):
            raise ValueError("Checkpoint report scale hashes differ from configured initialization")
        if model.earlier_history_features != "none" and (
                model.common_initial_parameter_sha256_ != checkpoint["common_initial_parameter_sha256"]
                or model.initial_parameter_sha256_ != fit_metadata.get("initial_parameter_sha256")):
            raise ValueError("Checkpoint earlier history hashes differ from configured initialization")
        if reference is not None:
            # CPU/CUDA elementary kernels can differ by float32 rounding.
            # Check the configured initialization, then load the STORED bytes;
            # never re-anchor a trained model to its new execution device.
            if not torch.allclose(reference.to(model.device), model.coefficient_initial_variance,
                                  atol=0, rtol=4*torch.finfo(torch.float32).eps):
                raise ValueError("Checkpoint coefficient variance reference differs from configured initialization")
        model.load_state_dict(state)
        if model.report_scale_mode != "global" and model._report_scale_base_raw_sha256() != checkpoint["report_scale_base_raw_sha256"]:
            raise ValueError("Contextual report checkpoint base raw noise hash disagrees")
        model.origins_ = np.asarray(checkpoint["origins"])
        model.cells_ = checkpoint["cells"]
        model.fit_diagnostics_ = checkpoint["fit_diagnostics"]
        model.training_diagnostics_ = model.fit_diagnostics_
        if model.history_head_mode != "shared":
            for key in ("common_initial_parameter_sha256", "initial_output_head_sha256", "no_history_head_initial_sha256"):
                setattr(model, key + "_", checkpoint[key])
        if model.tree_static_mode != "none":
            model.tree_prior_training_sha256_ = tree_metadata["training_sha256"]
            model.common_initial_parameter_sha256_ = checkpoint["fit_diagnostics"]["common_initial_parameter_sha256"]
        model.scale_provenance_ = checkpoint["scale_provenance"]
        model.eval()
        return model
