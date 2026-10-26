# config.py
from dataclasses import dataclass
import torch as th


@dataclass
class DABSIConfig:
    L: int  # FIR length (= hop size, frame length 2L)
    eta: float = 0.98  # forgetting factor of the covariance EMA
    rho: float = 1.0  # ADMM penalty
    mu: float = 0.4  # local (BFGS) step size
    lambd: float = 0.0  # prior regularisation weight
    gev_reg: float = 1e-9  # regularisation of the GEV ratio (a+reg)/(b+reg)
    gev_lim: float = 4.0  # upper clip of the GEV ratio
    bfgs_tol: float = 1e-10  # curvature gate for the BFGS inverse-Hessian update
    local_iters: int = 1
    da: bool = (
        True  # distributed averaging of the consensus norm (False: exact global norm)
    )
    real_dtype: th.dtype = th.float64
    device: str = "cpu"

    @property
    def complex_dtype(self) -> th.dtype:
        return th.complex128 if self.real_dtype == th.float64 else th.complex64
