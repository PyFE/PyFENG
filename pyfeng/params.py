"""
Parameter dataclasses for PyFENG option pricing models.

Each concrete model family has a dedicated dataclass that:
  - declares parameters with types and defaults,
  - validates constraints in ``__post_init__``, and
  - precomputes model-parameter-only constants (e.g. ``_mgf1_correction``)
    so pricing calls do not repeat the work.

``intr``, ``divr``, and ``is_fwd`` are **keyword-only** in every class
(requires Python ≥ 3.10).

Hierarchy::

    BaseParams                      single vol parameter + intr/divr/is_fwd
    ├── BsmParams                   Black-Scholes-Merton
    ├── NormParams                  Bachelier (Normal)
    ├── CevParams                   CEV  (+beta)
    │   └── SabrParams              SABR = stochastic CEV  (+vov, rho)
    │       └── NsvhParams          NSVH = Normal SABR (+lam; beta fixed to 0)
    ├── SvParams                    stochastic-vol base  (+vov, rho, mr, theta)
    │   ├── HestonParams            Heston
    │   │   └── RoughHestonParams   Rough Heston  (+alpha)
    │   ├── GarchParams             GARCH diffusion
    │   ├── OusvParams              OU stochastic vol
    │   └── Sv32Params              3/2 model
    ├── VarGammaParams              Variance Gamma subordinated BM  (+nu, theta)
    ├── NigParams                   NIG subordinated BM  (+nu, theta)
    ├── SviParams                   SVI  (+vov, rho, smooth, shift)
    ├── SpreadParams                two-asset spread/max  (+sigma2, rho)
    └── MaParams                    multi-asset base: sigma array + cor → cor_m/cov_m/chol_m

    CgmyParams                      CGMY — no sigma; standalone
"""

from __future__ import annotations

import dataclasses
import warnings
import numpy as np
import scipy.special as spsp
from dataclasses import KW_ONLY, dataclass, field
from typing import ClassVar


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────

def _params_kw(obj) -> dict:
    """Return a dict of all *init* fields, keyword-only fields last.

    Excludes ``field(init=False)`` entries (precomputed constants).
    Keyword-only fields (``intr``, ``divr``, ``is_fwd``) are sorted to the end
    so the dict reads in natural model-parameter order.
    """
    positional = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj) if f.init and not f.kw_only}
    kw_only    = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj) if f.init and f.kw_only}
    return {**positional, **kw_only}


# ──────────────────────────────────────────────────────────────────────────────
# Base: single vol parameter
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BaseParams:
    """
    Base parameter class for single-asset option models.

    ``intr``, ``divr``, and ``is_fwd`` are keyword-only so they cannot be
    passed positionally, preventing silent argument-order mistakes.
    """
    sigma: float
    _: KW_ONLY
    intr: float = 0.0
    divr: float = 0.0
    is_fwd: bool = False

    # Benchmark file name relative to the ``data/`` directory.
    # None (default) means derive from model_type: ``{model_type.lower()}_benchmark.xlsx``.
    # Override with a literal filename in subclasses that share another model's benchmark file.
    _benchmark_file: ClassVar[str] = None

    def params_kw(self) -> dict:
        """Model parameters as a keyword-argument dictionary."""
        return _params_kw(self)

    def params_hash(self) -> int:
        """Stable hash of the model parameters, suitable as a cache key."""
        dct = self.params_kw()
        return hash((frozenset(dct.keys()), frozenset(dct.values())))

    @classmethod
    def init_benchmark(cls, set_no=None):
        """
        Instantiate a model with stored benchmark parameter sets.

        Args:
            set_no: set number. If None, return the full parameter table.

        Returns:
            DataFrame of all benchmark sets if ``set_no`` is None;
            ``(model, result_df, info_dict)`` otherwise, where ``info_dict``
            contains ``args_pricing``, ``ref``, ``val``, and ``is_iv``.
        """
        import os
        import pandas as pd

        this_dir = os.path.dirname(os.path.abspath(__file__))
        fname = cls._benchmark_file or f"{cls.model_type.lower()}_benchmark.xlsx"
        file = os.path.join(this_dir, "data", fname)
        df_param = pd.read_excel(file, sheet_name="Param", index_col="Sheet")

        if set_no is None:
            return df_param

        df_val = pd.read_excel(file, sheet_name=str(set_no))
        param = df_param.loc[set_no].to_dict()

        valid_keys = {f.name for f in dataclasses.fields(cls) if f.init}
        args_model = {k: param[k] for k in param.keys() & valid_keys}
        args_pricing = {k: param[k] for k in param.keys() & {"texp", "spot"}}

        assert df_val.columns[0] == "Strike"
        args_pricing["strike"] = df_val.values[:, 0]

        if "CP" in df_val.columns:
            args_pricing["cp"] = df_val["CP"].values

        val = df_val[param["col_name"]].values
        is_iv = param["col_name"].startswith("IV")

        return cls(**args_model), df_val, {
            "args_pricing": args_pricing,
            "ref": param["Reference"],
            "val": val,
            "is_iv": is_iv,
        }

    @classmethod
    def from_model(cls, other):
        """
        Create a new instance by copying parameters from another instance of the same model type.

        Args:
            other: source param or model instance.

        Returns:
            New instance of ``cls`` with parameters copied from ``other``.

        Raises:
            TypeError: if source and target have incompatible model types.
        """
        cls_type = getattr(cls, 'model_type', None)
        other_type = getattr(type(other), 'model_type', None)

        if isinstance(cls_type, str) and isinstance(other_type, str):
            # Compare model_type strings: allows cross-algorithm copies (e.g. HestonFft → HestonMC)
            if cls_type != other_type:
                raise TypeError(
                    f"Model type mismatch: source is '{other_type}' ({type(other).__name__}) "
                    f"but target is '{cls_type}' ({cls.__name__})."
                )
        elif not isinstance(other, cls):
            raise TypeError(
                f"Cannot copy from '{type(other).__name__}' to '{cls.__name__}'."
            )

        return cls(**other.params_kw())


# ──────────────────────────────────────────────────────────────────────────────
# Simple closed-form models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BsmParams(BaseParams):
    """Parameters for the Black-Scholes-Merton (BSM / lognormal) model."""
    model_type: ClassVar[str] = "Bsm"


@dataclass
class NormParams(BaseParams):
    """Parameters for the Bachelier (Normal / absolute-diffusion) model."""
    model_type: ClassVar[str] = "Norm"


@dataclass
class InvGamParams(BaseParams):
    """Parameters for the Inverse Gamma distribution option pricing model."""
    model_type: ClassVar[str] = "InvGam"


@dataclass
class InvGaussParams(BaseParams):
    """Parameters for the Inverse Gaussian distribution option pricing model."""
    model_type: ClassVar[str] = "InvGauss"


@dataclass
class CevParams(BaseParams):
    """
    Parameters for the Constant Elasticity of Variance (CEV) model.

    The local-vol SDE is :math:`dS = S^\\beta\\,\\sigma\\,dW`.
    ``beta = 1`` recovers BSM; ``beta = 0`` recovers Normal.
    """
    model_type: ClassVar[str] = "Cev"
    beta: float = 0.5


# ──────────────────────────────────────────────────────────────────────────────
# SABR — stochastic CEV: shares beta with CevParams
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SabrParams(CevParams):
    """
    Parameters for SABR stochastic-volatility models.

    SABR is a stochastic CEV model, so it inherits ``beta`` from
    :class:`CevParams` (default ``0.5``).  ``beta = 1`` gives the lognormal
    (Black) SABR; ``beta = 0`` gives the Normal SABR.

    Field order in ``__init__``:
    ``(sigma, beta=0.5, vov=0.1, rho=0.0, *, intr=0.0, divr=0.0, is_fwd=False)``
    """
    model_type: ClassVar[str] = "SABR"
    var_process: ClassVar[bool] = False
    vov: float = 0.1
    rho: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Stochastic-volatility base
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SvParams(BaseParams):
    """
    Base parameter class for mean-reverting stochastic-volatility models.

    ``theta`` defaults to ``None``; ``__post_init__`` replaces it with
    ``sigma`` so the long-run mean equals the initial variance/vol by default.
    """
    var_process: ClassVar[bool]   # True if SDE drives variance; False if volatility
    vov: float = 0.01
    rho: float = 0.0
    mr: float = 0.01
    theta: float = field(default=None)

    def __post_init__(self):
        if self.theta is None:
            self.theta = self.sigma



# ──────────────────────────────────────────────────────────────────────────────
# Concrete SV models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HestonParams(SvParams):
    """
    Parameters for the Heston stochastic-volatility model.

    Raises:
        ValueError: if any constraint is violated:
            ``sigma > 0``, ``vov > 0``, ``mr > 0``, ``theta > 0``,
            ``rho ∈ (−1, 1)``.
    """
    model_type: ClassVar[str] = "Heston"
    var_process: ClassVar[bool] = True

    def __post_init__(self):
        super().__post_init__()   # sets self.theta = self.sigma when theta is None
        if self.sigma <= 0.0:
            raise ValueError(f"sigma (initial variance) must be > 0, got {self.sigma}")
        if self.vov <= 0.0:
            raise ValueError(f"vov (vol-of-vol) must be > 0, got {self.vov}")
        if self.mr <= 0.0:
            raise ValueError(f"mr (mean-reversion) must be > 0, got {self.mr}")
        if self.theta <= 0.0:
            raise ValueError(f"theta (long-run variance) must be > 0, got {self.theta}")
        if not (-1.0 < self.rho < 1.0):
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")


@dataclass
class GarchParams(SvParams):
    """Parameters for the GARCH diffusion model."""
    model_type: ClassVar[str] = "GarchDiff"
    var_process: ClassVar[bool] = True


@dataclass
class OusvParams(SvParams):
    """Parameters for the Ornstein-Uhlenbeck stochastic-volatility (OUSV) model."""
    model_type: ClassVar[str] = "OUSV"
    var_process: ClassVar[bool] = False


@dataclass
class Sv32Params(SvParams):
    """Parameters for the 3/2 stochastic-volatility model."""
    model_type: ClassVar[str] = "3/2"
    var_process: ClassVar[bool] = True


@dataclass
class RoughHestonParams(HestonParams):
    """
    Parameters for the Rough Heston model.

    Extends :class:`HestonParams` with the Hurst-like exponent ``alpha``;
    the variance process is driven by a fractional Brownian motion with
    :math:`H = \\alpha - \\tfrac{1}{2}`.
    """
    model_type: ClassVar[str] = "rHeston"
    alpha: float = 0.62


# ──────────────────────────────────────────────────────────────────────────────
# Subordinated-BM (Lévy time-change) models
# Both VG and NIG are drifted BMs time-changed by a subordinator.
# Parameters: sigma (BM vol), nu (subordinator variance rate), theta (drift/skew).
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VarGammaParams(BaseParams):
    """
    Parameters for the Variance Gamma (VG) model.

    The log price is

    .. math::

        \\log(F_T/F_0) = \\omega T + \\theta G_T + \\sigma W_{G_T},

    where :math:`G_T \\sim \\mathrm{Gamma}(T/\\nu,\\, \\nu)` is the Gamma subordinator
    with mean :math:`T` and variance :math:`\\nu T`, and the drift

    .. math::

        \\omega = \\frac{1}{\\nu}\\ln\\!\\left(1 - \\theta\\nu - \\tfrac{1}{2}\\sigma^2\\nu\\right)

    is chosen so that :math:`E[F_T] = F_0` (martingale condition).

    * ``sigma`` (:math:`\\sigma`): volatility of the Brownian component.
    * ``nu`` (:math:`\\nu`): variance rate of the Gamma subordinator (controls
      excess kurtosis); :math:`\\nu > 0`.
    * ``theta`` (:math:`\\theta`): drift of the Gamma component (controls skewness).

    Constraint: :math:`1 - \\theta\\nu - \\tfrac{1}{2}\\sigma^2\\nu > 0`.

    Precomputes ``_mgf1_correction = -omega * nu = -ln(1 - theta*nu - sigma^2*nu/2)``.
    """
    model_type: ClassVar[str] = "VarGamma"
    nu: float = 0.01
    theta: float = 0.0
    _mgf1_correction: float = field(init=False, repr=False)

    def __post_init__(self):
        arg = 1.0 - self.theta * self.nu - 0.5 * self.sigma**2 * self.nu
        if arg <= 0:
            raise ValueError(
                f"VarGamma constraint violated: "
                f"1 - theta*nu - sigma^2*nu/2 = {arg:.6g} <= 0 "
                f"(sigma={self.sigma}, nu={self.nu}, theta={self.theta}). "
                f"Reduce |theta|, nu, or sigma."
            )
        self._mgf1_correction = -np.log(arg)

    def sv_params_dict(self):
        """
        Return the SV parametrization ``(sigma_bs, rho, vov)`` as a dict.

        The conversion requires :math:`\\nu(2\\theta + \\sigma^2) \\le 1`, which is
        stricter than the VG model constraint (:math:`\\le 2`).

        Returns:
            ``{'sigma_bs': ..., 'rho': ..., 'vov': ...}``

        Raises:
            ValueError: if :math:`\\nu(2\\theta + \\sigma^2) > 1`.
        """
        cond = self.nu * (2.0 * self.theta + self.sigma**2)
        if cond > 1.0:
            raise ValueError(
                f"Cannot convert to SV parametrization: "
                f"nu*(2*theta + sigma^2) = {cond:.6g} > 1."
            )
        vov = np.sqrt(self.nu)
        c = 1.0 - np.sqrt(1.0 - cond)
        denom = np.sqrt(c**2 + self.sigma**2 * self.nu)
        return {'sigma_bs': float(denom / vov), 'rho': float(c / denom), 'vov': float(vov)}

    @staticmethod
    def to_sv_param(sigma, theta, nu):
        """Convert ``(sigma, theta, nu)`` to ``(sigma_bs, rho, vov)``.

        Raises ``ValueError`` if :math:`\\nu(2\\theta+\\sigma^2) > 1`
        (stricter than the VG model constraint of ``< 2``).
        """
        cond = nu * (2.0 * theta + sigma**2)
        if cond > 1.0:
            raise ValueError(
                f"Cannot convert to SV parametrization: "
                f"nu*(2*theta + sigma^2) = {cond:.6g} > 1."
            )
        vov = np.sqrt(nu)
        c = 1.0 - np.sqrt(1.0 - cond)
        denom = np.sqrt(c**2 + sigma**2 * nu)
        return float(denom / vov), float(c / denom), float(vov)

    @staticmethod
    def to_orig_param(sigma_bs, rho, vov):
        """Convert ``(sigma_bs, rho, vov)`` to ``(sigma, theta, nu)``."""
        nu = float(vov**2)
        sigma = float(np.sqrt(1.0 - rho**2) * sigma_bs)
        theta = float(rho * sigma_bs / vov - 0.5 * sigma_bs**2)
        return sigma, theta, nu

    @classmethod
    def from_sv_param(cls, sigma_bs, rho, vov, **kwargs):
        """Construct an instance from the SV parametrization ``(sigma_bs, rho, vov)``."""
        sigma, theta, nu = cls.to_orig_param(sigma_bs, rho, vov)
        return cls(sigma=sigma, theta=theta, nu=nu, **kwargs)


@dataclass
class NigParams(BaseParams):
    """
    Parameters for the Normal Inverse Gaussian (NIG) model.

    The log price is

    .. math::

        \\log(F_T/F_0) = \\omega T + \\theta I_T + \\sigma W_{I_T},

    where :math:`I_T \\sim \\mathrm{IG}(T,\\, T^2/\\nu)` is the Inverse Gaussian subordinator
    with mean :math:`T` and variance :math:`\\nu T`, and the drift

    .. math::

        \\omega = -\\frac{1}{\\nu}\\left(1 - \\sqrt{1 - 2\\theta\\nu - \\sigma^2\\nu}\\right)

    is chosen so that :math:`E[F_T] = F_0` (martingale condition).

    * ``sigma`` (:math:`\\sigma`): volatility of the Brownian component.
    * ``nu`` (:math:`\\nu`): variance rate of the IG subordinator (controls
      excess kurtosis); :math:`\\nu > 0`.
    * ``theta`` (:math:`\\theta`): drift of the IG component (controls skewness).

    Constraint: :math:`1 - 2\\theta\\nu - \\sigma^2\\nu > 0`.

    Precomputes ``_mgf1_correction = -omega * nu = 1 - sqrt(1 - 2*theta*nu - sigma^2*nu)``.
    """
    model_type: ClassVar[str] = "NIG"
    nu: float = 0.01
    theta: float = 0.0
    _mgf1_correction: float = field(init=False, repr=False)

    def __post_init__(self):
        arg = 1.0 - 2.0 * self.theta * self.nu - self.sigma**2 * self.nu
        if arg <= 0:
            raise ValueError(
                f"NIG constraint violated: "
                f"1 - 2*theta*nu - sigma^2*nu = {arg:.6g} <= 0 "
                f"(sigma={self.sigma}, nu={self.nu}, theta={self.theta}). "
                f"Reduce |theta|, nu, or sigma."
            )
        self._mgf1_correction = 1.0 - np.sqrt(arg)

    def sv_params_dict(self):
        """
        Return the SV parametrization ``(sigma_bs, rho, vov)`` as a dict.

        For NIG the model constraint :math:`\\nu(2\\theta + \\sigma^2) < 1` coincides
        with the SV conversion condition, so this is always valid for valid parameters.

        Returns:
            ``{'sigma_bs': ..., 'rho': ..., 'vov': ...}``
        """
        vov = np.sqrt(self.nu)
        cond = self.nu * (2.0 * self.theta + self.sigma**2)
        c = 1.0 - np.sqrt(1.0 - cond)
        denom = np.sqrt(c**2 + self.sigma**2 * self.nu)
        return {'sigma_bs': float(denom / vov), 'rho': float(c / denom), 'vov': float(vov)}

    @staticmethod
    def to_sv_param(sigma, theta, nu):
        """Convert ``(sigma, theta, nu)`` to ``(sigma_bs, rho, vov)``.

        For NIG the model constraint :math:`\\nu(2\\theta+\\sigma^2) < 1` coincides
        with the SV conversion condition, so no additional guard is needed.
        """
        vov = np.sqrt(nu)
        cond = nu * (2.0 * theta + sigma**2)
        c = 1.0 - np.sqrt(1.0 - cond)
        denom = np.sqrt(c**2 + sigma**2 * nu)
        return float(denom / vov), float(c / denom), float(vov)

    @staticmethod
    def to_orig_param(sigma_bs, rho, vov):
        """Convert ``(sigma_bs, rho, vov)`` to ``(sigma, theta, nu)``."""
        nu = float(vov**2)
        sigma = float(np.sqrt(1.0 - rho**2) * sigma_bs)
        theta = float(rho * sigma_bs / vov - 0.5 * sigma_bs**2)
        return sigma, theta, nu

    @classmethod
    def from_sv_param(cls, sigma_bs, rho, vov, **kwargs):
        """Construct an instance from the SV parametrization ``(sigma_bs, rho, vov)``."""
        sigma, theta, nu = cls.to_orig_param(sigma_bs, rho, vov)
        return cls(sigma=sigma, theta=theta, nu=nu, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# NSVH  (sigma + vov/rho/lam; no mr, theta, or beta)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NsvhParams(SabrParams):
    """
    Parameters for the Normal Stochastic-Volatility with Hyperbolic (NSVH) model.

    NSVH is the Normal (``beta = 0``) special case of SABR extended with the
    hyperbolic shape parameter ``lam`` (λ).  ``beta`` is always fixed to ``0``
    and excluded from ``__init__``; ``vov`` and ``rho`` are inherited from
    :class:`SabrParams`.

    ``lam = 0`` recovers the Normal SABR; ``lam = 1`` gives Johnson's SU.

    Field order in ``__init__``:
    ``(sigma, vov=0.1, rho=0.0, lam=0.0, *, intr=0.0, divr=0.0, is_fwd=False)``
    """
    model_type: ClassVar[str] = "Nsvh"
    _benchmark_file: ClassVar[str] = "sabr_benchmark.xlsx"
    beta: float = field(default=0.0, init=False)   # NSVH is always Normal (beta=0)
    lam: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# SVI
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SviParams(BaseParams):
    """
    Parameters for the Stochastic-Volatility-Inspired (SVI) parametrisation.

    Raw SVI: :math:`w(k) = a + b(\\rho(k-m) + \\sqrt{(k-m)^2 + s^2})`
    where ``sigma`` = a (level), ``vov`` = b (slope), ``rho`` = rotation,
    ``smooth`` = s (smoothness), ``shift`` = m (translation).
    """
    model_type: ClassVar[str] = "Svi"
    vov: float = 0.4
    rho: float = -0.4
    smooth: float = 0.1
    shift: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# CGMY — standalone (no sigma)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CgmyParams(BaseParams):
    """
    Parameters for the CGMY (Carr-Geman-Madan-Yor) Lévy model.

    The Lévy measure is
    :math:`C[e^{-Mx}/x^{1+Y}\\mathbf{1}_{x>0} + e^{Gx}/|x|^{1+Y}\\mathbf{1}_{x<0}]\\,dx`.

    ``sigma`` is inherited from :class:`BaseParams` but unused (set to ``None``);
    the CGMY process has no Brownian diffusion component.

    Constraints: ``C > 0``, ``G > 0``, ``M > 1`` (for finite martingale
    correction), ``Y < 2``.

    Precomputes:

    * ``_gam_Y_C = C·Γ(-Y)``  — shared factor in every MGF call.
    * ``_M_pow_Y = M^Y``, ``_G_pow_Y = G^Y``  — reused in every MGF call.
    * ``_mgf1_correction = κ(1)/(C·Γ(-Y)) = -ω/(C·Γ(-Y))``.
    """
    model_type: ClassVar[str] = "CGMY"
    sigma: float = None          # CGMY has no diffusion component
    C: float = field(kw_only=True)
    G: float = field(kw_only=True)
    M: float = field(kw_only=True)
    Y: float = field(kw_only=True)

    _gam_Y_C: float = field(init=False, repr=False)
    _M_pow_Y: float = field(init=False, repr=False)
    _G_pow_Y: float = field(init=False, repr=False)
    _mgf1_correction: float = field(init=False, repr=False)

    def __post_init__(self):
        if self.C <= 0:
            raise ValueError(f"C = {self.C} must be positive.")
        if self.G <= 0:
            raise ValueError(f"G = {self.G} must be positive.")
        if self.M <= 1:
            raise ValueError(
                f"M = {self.M} must be greater than 1 for the martingale "
                f"correction kappa(1) to be finite."
            )
        if self.Y >= 2:
            raise ValueError(f"Y = {self.Y} must be less than 2.")

        self._gam_Y_C = spsp.gamma(-self.Y) * self.C
        self._M_pow_Y = np.power(self.M, self.Y)
        self._G_pow_Y = np.power(self.G, self.Y)
        self._mgf1_correction = (
            np.power(self.M - 1.0, self.Y) - self._M_pow_Y
            + np.power(self.G + 1.0, self.Y) - self._G_pow_Y
        )


# ──────────────────────────────────────────────────────────────────────────────
# Two-asset spread / max models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SpreadParams(BaseParams):
    """
    Parameters for two-asset spread and max option models.

    ``sigma`` is a 2-element array ``[σ₁, σ₂]``; ``rho`` is the correlation
    between the two assets.

    Generated ``__init__`` signature::

        SpreadParams(sigma, rho=0.0, *, intr=0.0, divr=0.0, is_fwd=False)
    """
    model_type: ClassVar[str] = "Spread"
    n_asset: ClassVar[int] = 2
    weight: ClassVar[np.ndarray] = np.array([1., -1.])

    rho: float = 0.0

    def __post_init__(self):
        self.sigma = np.atleast_1d(self.sigma).astype(float)
        if len(self.sigma) != 2:
            raise ValueError(f"sigma must be a 2-element array for SpreadParams; got length {len(self.sigma)}.")


# ──────────────────────────────────────────────────────────────────────────────
# Multi-asset base
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MaParams(BaseParams):
    """
    Base parameter class for multi-asset option models.

    ``sigma`` (inherited from :class:`BaseParams`) is a 1-D array of
    per-asset volatilities.  Exactly one of ``rho``, ``cor_m``, or ``cov_m``
    may be supplied to describe the correlation/covariance structure:

    * ``rho`` — scalar applied uniformly to every off-diagonal entry.
    * ``cor_m`` — full (n_asset × n_asset) correlation matrix.
    * ``cov_m`` — full (n_asset × n_asset) covariance matrix; ``sigma`` is
      derived from its diagonal and overrides the positional argument.

    Omitting all three is valid only for a single asset; for ``n_asset > 1``
    it raises :class:`ValueError`.

    ``weight`` controls how assets are combined in basket/spread payoffs.
    ``None`` defaults to equal weights ``1/n_asset``.

    Generated ``__init__`` signature::

        MaParams(sigma, rho=None, cor_m=None, cov_m=None,
                 *, weight=None, intr=0.0, divr=0.0, is_fwd=False)

    The following attributes are always available after construction:

    * ``n_asset`` — number of assets (``init=False``).
    * ``rho`` — scalar correlation for 2-asset or uniform-ρ models, else ``None``.
    * ``cor_m`` — (n_asset × n_asset) correlation matrix.
    * ``cov_m`` — (n_asset × n_asset) covariance matrix.
    * ``chol_m`` — lower-triangular Cholesky factor of ``cov_m`` (``init=False``).
    * ``weight`` — (n_asset,) asset-weight vector.
    """
    rho: float | None = None              # scalar ρ for all off-diagonal entries
    cor_m: np.ndarray | None = None       # full (n×n) correlation matrix
    cov_m: np.ndarray | None = None       # full (n×n) covariance matrix
    _: KW_ONLY
    weight: np.ndarray | float | None = None

    # ── always derived, never in __init__ ─────────────────────────────────────
    n_asset: int = field(init=False, repr=False)
    chol_m: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.sigma = np.atleast_1d(self.sigma)
        self.n_asset = len(self.sigma)

        # ── mutual exclusivity check ──────────────────────────────────────────
        n_given = sum(x is not None for x in (self.rho, self.cor_m, self.cov_m))
        if n_given > 1:
            raise ValueError("At most one of rho, cor_m, cov_m may be specified.")

        # ── resolve to cor_m + cov_m ──────────────────────────────────────────
        if self.cov_m is not None:
            self.cov_m = np.asarray(self.cov_m, dtype=float)
            if self.cov_m.shape != (self.n_asset, self.n_asset):
                raise ValueError(
                    f"cov_m must have shape ({self.n_asset}, {self.n_asset}); "
                    f"got {self.cov_m.shape}."
                )
            # derive sigma and cor_m from cov_m
            self.sigma = np.sqrt(np.diag(self.cov_m))
            self.cor_m = self.cov_m / (self.sigma[:, None] * self.sigma)
            self.rho = float(self.cor_m[0, 1]) if self.n_asset == 2 else None
        elif self.cor_m is not None:
            self.cor_m = np.asarray(self.cor_m, dtype=float)
            if self.cor_m.shape != (self.n_asset, self.n_asset):
                raise ValueError(
                    f"cor_m must have shape ({self.n_asset}, {self.n_asset}); "
                    f"got {self.cor_m.shape}."
                )
            self.cov_m = self.sigma * self.cor_m * self.sigma[:, None]
            self.rho = float(self.cor_m[0, 1]) if self.n_asset == 2 else None
        else:
            # rho path (or single-asset identity)
            rho = self.rho if self.rho is not None else 0.0
            self.cor_m = rho * np.ones((self.n_asset, self.n_asset)) + (
                1.0 - rho
            ) * np.eye(self.n_asset)
            self.cov_m = self.sigma * self.cor_m * self.sigma[:, None]
            # rho field already set (or stays None for single asset)

        self.chol_m = np.linalg.cholesky(self.cov_m)

        # ── weight ────────────────────────────────────────────────────────────
        if self.weight is None:
            self.weight = np.ones(self.n_asset) / self.n_asset
        elif np.isscalar(self.weight):
            self.weight = np.full(self.n_asset, float(self.weight))
        else:
            self.weight = np.asarray(self.weight, dtype=float)
            if self.weight.shape != (self.n_asset,):
                raise ValueError(
                    f"weight must have shape ({self.n_asset},); got {self.weight.shape}."
                )

    def params_kw(self) -> dict:
        """Return parameters as a keyword-argument dict.

        ``cov_m`` is always used as the canonical correlation representation
        so that ``cls(**obj.params_kw())`` round-trips cleanly regardless of
        whether ``rho``, ``cor_m``, or ``cov_m`` was originally provided.
        """
        d = _params_kw(self)
        # Drop rho and cor_m — cov_m (always populated) is sufficient.
        d.pop("rho", None)
        d.pop("cor_m", None)
        return d
