"""A positive-definite covariance factor with bounded marginal scales.

For every row, a free lower-triangular direction is normalized to unit length
and multiplied by its marginal standard deviation. In exact arithmetic this
parameterizes every positive-definite covariance with marginal standard
deviations in (.025, 1.2), independently of coordinate order. Reordering also
relabels the associated latent coordinates; it does not make physical phases
exchangeable. This pointwise covariance claim is not a claim about nesting
finite neural-network function classes or forecast performance.

No eigenvalue floor or conditioning guarantee is imposed. A covariance can
approach singularity. Floating-point sigmoid saturation can attain marginal
endpoints, and conversion to the input dtype adds ordinary rounding. Any
nonfinite factor or nonpositive returned diagonal is rejected. A positive
returned factor diagonal does not guarantee that explicitly forming L L^T
and factorizing that rounded matrix is numerically stable.
"""
from __future__ import annotations

import math
import struct

import torch


DIMENSION = 12
TRIANGULAR_COORDINATES = 78
MIN_MARGINAL_STD = .025
MAX_MARGINAL_STD = 1.2
OFFDIAGONAL_RAW_SCALE = .8
DIAGONAL_LINK_SCALE = 235 / 247
DIAGONAL_ANCHOR_RAW_FLOAT32 = struct.unpack('<f', struct.pack('<f', math.log(.3)))[0]
DIAGONAL_ANCHOR_NOMINAL_LOGIT = math.log(9 / 38)


def initial_diagonal_anchor(device) -> torch.Tensor:
    """Return a detached float64 logit matching the legacy float32 start.

    Preserve the old *separate* sigmoid, multiplication and addition on the
    training device. Their rounding gives .25 on the checked CPU and the next
    float32 value above .25 on the checked CUDA backend. A production model
    stores this result once; moving a fitted model must not recompute it.
    """
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Covariance initialization requires CPU or CUDA')
    with torch.no_grad():
        raw = torch.tensor(DIAGONAL_ANCHOR_RAW_FLOAT32, dtype=torch.float32, device=device)
        sigma0 = .025 + .975 * torch.sigmoid(raw)
        sigma0 = sigma0.to(torch.float64)
        anchor = torch.log((sigma0 - MIN_MARGINAL_STD) / (MAX_MARGINAL_STD - sigma0))
    if not bool(torch.isfinite(anchor)):
        raise FloatingPointError('Legacy initial covariance anchor is not finite')
    return anchor


def _anchor_tensor(value, device):
    if value is None:
        return initial_diagonal_anchor(device)
    if isinstance(value, torch.Tensor):
        if (value.ndim != 0 or value.dtype != torch.float64 or value.requires_grad
                or value.device.type not in ('cpu', 'cuda')):
            raise ValueError('The persisted covariance anchor must be a detached float64 scalar')
        anchor = value.to(device=device)
    elif isinstance(value, float):
        anchor = torch.tensor(value, dtype=torch.float64, device=device)
    else:
        raise ValueError('The persisted covariance anchor must be a float or detached float64 scalar')
    if not bool(torch.isfinite(anchor)):
        raise ValueError('The persisted covariance anchor must be finite')
    return anchor


def normalized_row_metadata(diagonal_anchor_logit=None):
    """Fresh JSON-safe description for fit/checkpoint/prediction provenance."""
    # Decode recorded metadata on CPU without replacing its actual anchor by
    # the CPU legacy anchor. The result is the original initial float32 SD.
    anchor = _anchor_tensor(diagonal_anchor_logit, torch.device('cpu'))
    sigma0 = (.025 + 1.175 * torch.sigmoid(anchor)).to(torch.float32).item()
    return dict(
        covariance_factor_meaning='marginal_scales_times_unit_lower_triangular_rows',
        covariance_factor_dimension=DIMENSION,
        covariance_factor_raw_coordinates=TRIANGULAR_COORDINATES,
        covariance_marginal_std_bounds=[MIN_MARGINAL_STD, MAX_MARGINAL_STD],
        covariance_offdiagonal_raw_scale=OFFDIAGONAL_RAW_SCALE,
        covariance_diagonal_link_scale=DIAGONAL_LINK_SCALE,
        covariance_diagonal_anchor_raw_float32=DIAGONAL_ANCHOR_RAW_FLOAT32,
        covariance_diagonal_anchor_logit=anchor.item(),
        covariance_initial_legacy_std_float32=sigma0,
        covariance_diagonal_anchor_nominal_logit=DIAGONAL_ANCHOR_NOMINAL_LOGIT,
        covariance_diagonal_anchor_rule='original_float32_initial_sigma_on_training_device; float64_log((sigma0-.025)/(1.2-sigma0)); persisted_for_fit_prediction_and_device_moves',
        covariance_factor_arithmetic='float64_link_and_scaled_row_normalization_then_input_dtype',
        covariance_factor_formula='L_i=sigma_i*(v_i,1)/sqrt(1+sum(v_i^2)); v_i=.8*raw_off_i',
        covariance_diagonal_link_formula='.025+1.175*sigmoid((235/247)*(raw_diag-h0_float32)+persisted_anchor_logit)',
        covariance_conditioning_limit='positive_diagonal_required; no_jitter_or_eigenvalue_floor; near_singularity_allowed',
        covariance_support_limit='pointwise_coordinate_relabeling_invariance; finite_MLP_function_class_nesting_not_claimed',
    )


def normalized_row_cholesky(raw: torch.Tensor, diagonal_anchor_logit=None) -> torch.Tensor:
    """Decode raw ``[...,78]`` coordinates to ``[...,12,12]`` lower factors.

    Input/output dtype is float32 or float64 on the same CPU/CUDA device. All
    link and normalization arithmetic uses float64. At the retained float32
    raw initialization (diagonals float32(log(.3)), off-diagonals zero), the
    float32 result exactly matches the old same-device factor. Its nominal
    value is .25 I; the old CUDA float32 expression can differ by one ULP.
    The link and .8 direction scale match the old factor's local initial
    Jacobian to float32 rounding precision. A production caller supplies the
    anchor stored at initialization, retaining it across checkpoint/device
    moves; the default is for standalone same-device initialization only.

    Scaling the direction before its norm avoids squaring extreme values.
    That numerical scale cancels algebraically; it changes neither the law
    nor its derivative. Its gradient is detached because the final normalized
    vector is homogeneous of degree zero in this auxiliary scale. This also
    avoids overflow in derivatives of a large squared scaling denominator.

    For the support claim, take the positive-diagonal Cholesky factor A of
    any desired covariance. Set sigma_i=||A_i|| and v_ij=A_ij/A_ii. Then the
    normalized direction is A_i/||A_i||. The logistic link and free raw
    off-diagonals invert these choices whenever .025 < sigma_i < 1.2.
    """
    if (not isinstance(raw, torch.Tensor) or raw.dtype not in (torch.float32, torch.float64)
            or raw.device.type not in ('cpu', 'cuda') or raw.ndim < 1
            or raw.shape[-1] != TRIANGULAR_COORDINATES or raw.numel() == 0):
        raise ValueError('Require nonempty CPU/CUDA float32/float64 raw [...,78] coordinates')
    if not bool(torch.isfinite(raw).all()):
        raise ValueError('Covariance raw coordinates must be finite')
    anchor = _anchor_tensor(diagonal_anchor_logit, raw.device)
    values = raw.to(torch.float64)
    rows, columns = torch.tril_indices(DIMENSION, DIMENSION, device=raw.device)
    diagonal = rows == columns
    sigma = MIN_MARGINAL_STD + (MAX_MARGINAL_STD - MIN_MARGINAL_STD) * torch.sigmoid(
        DIAGONAL_LINK_SCALE * (values[..., diagonal] - DIAGONAL_ANCHOR_RAW_FLOAT32)
        + anchor)
    direction = values.new_zeros((*values.shape[:-1], DIMENSION, DIMENSION))
    direction[..., rows[~diagonal], columns[~diagonal]] = OFFDIAGONAL_RAW_SCALE * values[..., ~diagonal]
    direction[..., torch.arange(DIMENSION, device=raw.device), torch.arange(DIMENSION, device=raw.device)] = 1.
    # Each direction contains a unit diagonal, so the scale and norm are
    # positive without imposing any additional floor on the covariance.
    scale = direction.abs().amax(dim=-1, keepdim=True).detach()
    scaled = direction / scale
    unit_rows = scaled / torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    lower = (sigma[..., :, None] * unit_rows).to(raw.dtype)
    if not bool(torch.isfinite(lower).all()) or not bool((lower.diagonal(dim1=-2, dim2=-1) > 0).all()):
        raise FloatingPointError('Covariance factor exceeded finite positive-diagonal numerical support')
    return lower
