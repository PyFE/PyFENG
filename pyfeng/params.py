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
    ├── SvParams                    stochastic-vol base  (+vov, rho, mr, theta)
    │   ├── HestonParams            Heston
    │   │   └── RoughHestonParams   Rough Heston  (+alpha)
    │   ├── GarchParams             GARCH diffusion
    │   ├── OusvParams              OU stochastic vol
    │   └── Sv32Params              3/2 model
    ├── VarGammaParams              Variance Gamma subordinated BM  (+nu, theta)
    ├── NigParams                   NIG subordinated BM  (+nu, theta)
    ├── NsvhParams                  NSVH  (+vov, rho, lam)
    └── SviParams                   SVI  (+vov, rho, smooth, shift)

    CgmyParams                      CGMY — no sigma; standalone
"""

from __future__ import annotations

import dataclasses
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

    def params_kw(self) -> dict:
        """Model parameters as a keyword-argument dictionary."""
        return _params_kw(self)

    def params_hash(self) -> int:
        """Stable hash of the model parameters, suitable as a cache key."""
        dct = self.params_kw()
        return hash((frozenset(dct.keys()), frozenset(dct.values())))

    @classmethod
    def from_param(cls, other):
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
    :class:`CevParams`.  The default ``beta = 1.0`` gives the lognormal
    (Black) SABR; ``beta = 0`` gives the Normal SABR.

    Field order in ``__init__``:
    ``(sigma, beta=1.0, vov=0.1, rho=0.0, *, intr=0.0, divr=0.0, is_fwd=False)``
    """
    model_type: ClassVar[str] = "SABR"
    var_process: ClassVar[bool] = False
    beta: float = 1.0           # override CevParams default (0.5 → 1.0)
    vov: float = 0.1
    rho: float = 0.0

    @classmethod
    def init_benchmark(cls, set_no=None):
        """
        Instantiate a SABR model with stored benchmark parameter sets.

        Args:
            set_no: set number. If None, return the full parameter table.

        Returns:
            DataFrame of all benchmark sets if set_no is None;
            (model, result DataFrame, param dict) otherwise.

        References:
            - Antonov et al. (2013). SABR spreads its wings. Risk.
            - Antonov et al. (2019). Modern SABR Analytics. Springer.
            - Cai et al. (2017). Exact Simulation of the SABR Model. Operations Research.
            - Korn & Tang (2013). Exact analytical solution for the normal SABR model. Wilmott.
            - Lewis (2016). Option valuation under stochastic volatility II.
        """
        import os
        import pandas as pd

        this_dir = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(this_dir, f"data/{cls.model_type.lower()}_benchmark.xlsx")
        df_param = pd.read_excel(file, sheet_name="Param", index_col="Sheet")

        if set_no is None:
            return df_param

        df_val = pd.read_excel(file, sheet_name=str(set_no))
        param = df_param.loc[set_no].to_dict()

        valid_keys = {f.name for f in dataclasses.fields(cls)}
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

    @classmethod
    def init_benchmark(cls, set_no=None):
        """
        Instantiate a model with stored benchmark parameter sets.

        Args:
            set_no: set number. If None, return the full parameter table.

        Returns:
            DataFrame of all benchmark sets if set_no is None;
            (model, result DataFrame, param dict) otherwise.
        """
        import os
        import pandas as pd

        this_dir = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(this_dir, f"data/{cls.model_type.lower()}_benchmark.xlsx")
        df_param = pd.read_excel(file, sheet_name="Param", index_col="Sheet")

        if set_no is None:
            return df_param

        df_val = pd.read_excel(file, sheet_name=str(set_no))
        param = df_param.loc[set_no].to_dict()

        valid_keys = {f.name for f in dataclasses.fields(cls)}
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


# ──────────────────────────────────────────────────────────────────────────────
# Concrete SV models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HestonParams(SvParams):
    """Parameters for the Heston stochastic-volatility model."""
    model_type: ClassVar[str] = "Heston"
    var_process: ClassVar[bool] = True


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

    The log price is :math:`\\theta G_T + \\sigma W_{G_T} + \\omega T` where
    :math:`G_T \\sim \\mathrm{Gamma}(T/\\nu,\\, \\nu)` is the Gamma subordinator
    with mean :math:`T` and variance :math:`\\nu T`.

    * ``sigma`` (:math:`\\sigma`): volatility of the Brownian component.
    * ``nu`` (:math:`\\nu`): variance rate of the Gamma subordinator (controls
      excess kurtosis); :math:`\\nu > 0`.
    * ``theta`` (:math:`\\theta`): drift of the Gamma component (controls skewness).

    Constraint: :math:`1 - \\theta\\nu - \\tfrac{1}{2}\\sigma^2\\nu > 0`.

    Precomputes ``_mgf1_correction = -ln(1 - theta*nu - sigma^2*nu/2) = kappa(1)*nu``.
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


@dataclass
class NigParams(BaseParams):
    """
    Parameters for the Normal Inverse Gaussian (NIG) model.

    The log price is :math:`\\theta I_T + \\sigma W_{I_T} + \\omega T` where
    :math:`I_T \\sim \\mathrm{IG}(T,\\, T^2/\\nu)` is the Inverse Gaussian subordinator
    with mean :math:`T` and variance :math:`\\nu T`.

    * ``sigma`` (:math:`\\sigma`): volatility of the Brownian component.
    * ``nu`` (:math:`\\nu`): variance rate of the IG subordinator (controls
      excess kurtosis); :math:`\\nu > 0`.
    * ``theta`` (:math:`\\theta`): drift of the IG component (controls skewness).

    Constraint: :math:`1 - 2\\theta\\nu - \\sigma^2\\nu > 0`.

    Precomputes ``_mgf1_correction = 1 - sqrt(1 - 2*theta*nu - sigma^2*nu) = kappa(1)*nu``.
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


# ──────────────────────────────────────────────────────────────────────────────
# NSVH  (sigma + vov/rho/lam; no mr, theta, or beta)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NsvhParams(BaseParams):
    """
    Parameters for the Normal Stochastic-Volatility with Hyperbolic (NSVH) model.

    Uses ``lam`` (λ) as the shape parameter.
    ``lam = 0`` gives the Normal SABR; ``lam = 1`` gives Johnson's SU.
    """
    model_type: ClassVar[str] = "Nsvh"
    var_process: ClassVar[bool] = False
    vov: float = 0.1
    rho: float = 0.0
    lam: float = 0.0

    @classmethod
    def init_benchmark(cls, set_no=None):
        """
        Instantiate an NSvh model with stored benchmark parameter sets.

        NSvh benchmarks are stored in the shared SABR benchmark file.

        Args:
            set_no: set number. If None, return the full parameter table.

        Returns:
            DataFrame of all benchmark sets if set_no is None;
            (model, result DataFrame, param dict) otherwise.

        References:
            - Choi, Liu & Seo (2019). Hyperbolic normal stochastic volatility model.
              J Futures Mark 39:186–204.
        """
        import os
        import pandas as pd

        this_dir = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(this_dir, "data/sabr_benchmark.xlsx")
        df_param = pd.read_excel(file, sheet_name="Param", index_col="Sheet")

        if set_no is None:
            return df_param

        df_val = pd.read_excel(file, sheet_name=str(set_no))
        param = df_param.loc[set_no].to_dict()

        valid_keys = {f.name for f in dataclasses.fields(cls)}
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
