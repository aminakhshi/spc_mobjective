"""
A modular script for implementing SDE solvers. It supports both Itô (Euler, SRI2) and Stratonovich (Heun, SRS2) methods.

- Allows choosing between white noise, power-law (1/f^β) noise, or OU noise internally.
- Optionally accepts user-supplied increments (dW).

copyright: 2024, @aminakhshi
"""

import logging
import numpy as np
from collections import namedtuple
from typing import Callable, Tuple, Optional, Union, List, Any, Iterable, Dict
from solvers.base_solver import Solver

# Configure module-level logger
logger = logging.getLogger(__name__)

Solution = namedtuple('Solution', ['t', 'y'])

# Utilities

def _check_args(
    F: Callable[[float, np.ndarray], np.ndarray],
    G: Union[Callable, List[Callable]],
    y0: np.ndarray,
    tspan: np.ndarray,
    dW: Optional[np.ndarray],
    seed: Optional[int] = None
) -> Tuple[int, int, Callable, Union[Callable, List[Callable]], np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
    """
    Basic argument checks to validate and process arguments for stochastic solvers.

    Parameters:
    -----------
    F : Callable
        Deterministic function f(t, y).
    G : Union[Callable, List[Callable]]
        Stochastic function(s) G(t, y).
    y0 : np.ndarray
        Initial state vector.
    tspan : np.ndarray
        Array of time points.
    dW : Optional[np.ndarray]
        Wiener increments.


    Returns:
    --------
    Tuple containing:
        - d: int, dimension of y0.
        - m: int, number of Wiener processes.
        - F: Callable, deterministic function.
        - G: Union[Callable, List[Callable]], stochastic function(s).
        - y0: np.ndarray, initial state vector.
        - tspan: np.ndarray, array of time points.
        - dW: Optional[np.ndarray], Wiener increments.
    """
    logger.debug("Validating input arguments.")
    if not callable(F):
        logger.error("Invalid argument: F must be a callable function returning an array.")
        raise ValueError("F must be a callable function returning an array.")
    if not (callable(G) or isinstance(G, list)):
        logger.error("Invalid argument: G must be callable or a list of callables.")
        raise ValueError("G must be callable or a list of callables.")

    y0 = np.asarray(y0)
    if y0.ndim != 1:
        logger.error("Invalid argument: y0 must be a 1-dimensional array.")
        raise ValueError("y0 must be a 1-dimensional array.")

    tspan = np.asarray(tspan)
    if tspan.ndim != 1 or len(tspan) < 2:
        logger.error("Invalid argument: tspan must be 1-dimensional with at least two points.")
        raise ValueError("tspan must be 1-dimensional with at least two points.")

    d = len(y0) # dimension of state
    if isinstance(G, list):
        m = len(G)
        for idx, g in enumerate(G):
            if not callable(g):
                logger.error(f"Invalid argument: G[{idx}] is not callable. All elements in G list must be callable functions.")
                raise ValueError(f"G[{idx}] is not callable. All elements in G list must be callable functions.")
    else:
        # Evaluate G at initial condition to determine shape
        G_matrix = G(tspan[0], y0)
        if not isinstance(G_matrix, np.ndarray) or G_matrix.ndim != 2:
            logger.error("Invalid argument: G(t, y) must return a 2-dimensional array with shape (d, m).")
            raise ValueError("G(t, y) must return a 2-dimensional array with shape (d, m).")
        d_check, m = G_matrix.shape
        if d_check != d:
            logger.error("Dimension mismatch: y0 dimension does not match the first dimension of G(t, y).")
            raise ValueError("Dimension mismatch: y0 dimension does not match the first dimension of G(t, y).")
    
    if seed is None:
        logger.debug("No seed provided; using default_rng.")
    
    logger.debug("Input arguments validated successfully.")
    return d, m, F, G, y0, tspan, dW, seed

def _powerlaw_noise(
    beta: float,
    N: Union[int, Iterable[int]],
    dt: float,
    generator: np.random.Generator,
    fmin: float = 0.1) -> np.ndarray:
    """
    Powerlaw (1/f)**beta noise. This code is implemented based on the algorithm in:
    Timmer, J. and Koenig, M., On generating power law noise. Astron. Astrophys. 300, 707-710 (1995)
    Normalised to unit variance
    
    Parameters:
    -----------
    beta : float
        The power-spectrum of the generated noise is proportional to
        S(f) = (1 / f)**beta
        Flicker / pink noise:   beta = 1
        Brown noise:            beta = 2
        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.
    N : int or Iterable[int]
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.
    dt : float
        Time step between samples.
    generator : np.random.Generator
        A NumPy random number generator instance.
    fmin : float, optional
        Low-frequency cutoff.
        Default: 0.1. It is not actually zero, but 1/samples.
    
    Returns
    -------
    out : np.ndarray
        The samples.
    """
    # Ensure N is a list to handle multiple dimensions
    try:
        size = list(N)
    except TypeError:
        size = [N]
    
    # The number of samples in each time series
    samples = size[-1]
    
    # Calculate Frequencies (assuming a sample rate of one)
    f = np.fft.rfftfreq(samples, dt)
    
    # Build scaling factors for all frequencies
    s_scale = f.copy()
    fmin = max(fmin, 1.0 / samples)  # Low frequency cutoff
    ix = np.sum(s_scale < fmin)      # Index of the cutoff
    if ix > 0 and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale**(-beta / 2.0)
    
    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].copy()
    if samples % 2 == 0:
        w[-1] *= 0.5  # Correct f = +-0.5 for even sample counts
    sigma = 2 * np.sqrt(np.sum(w**2)) / samples
    
    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)
    
    # Add empty dimension(s) to broadcast s_scale along last dimension
    dims_to_add = len(size) - 1
    s_scale = s_scale[(np.newaxis,) * dims_to_add + (...,)]
    
    # Generate scaled random power + phase using the provided generator
    sr = generator.normal(scale=s_scale, size=size)
    si = generator.normal(scale=s_scale, size=size)
    
    # If the signal length is even, the Nyquist frequency component must be real
    if samples % 2 == 0:
        si[..., -1] = 0.0
    
    # The DC component must always be real
    si[..., 0] = 0.0
    
    # Combine power and phase to form Fourier components
    s = sr + 1j * si
    
    # Transform to real time series and scale to unit variance
    y = np.fft.irfft(s, n=samples, axis=-1) / sigma
    return y

def _wiener_increments(
        N: int,
        m: int,
        dt: float,
        generator: np.random.Generator
        ) -> np.ndarray:
    """
    Generate standard Wiener increments with shape (N, m).
    """
    logger.debug(f"Generating Wiener increments with N={N}, m={m}, dt={dt}.")
    return generator.normal(loc=0.0, scale=np.sqrt(dt), size=(N, m))


def _ornstein_uhlenbeck_sequence(
        n_steps: int,
        m: int,
        dt: float,
        generator: np.random.Generator,
        tau: float,
        sigma: float = 1.0,
        mu: float = 0.0
    ) -> np.ndarray:
    """Generate OU process samples with stationary variance ``sigma``.

    Parameters
    ----------
    n_steps : int
        Number of time steps to generate.
    m : int
        Number of independent OU processes.
    dt : float
        Integration step size.
    generator : np.random.Generator
        Random number generator instance.
    tau : float
        Correlation time constant (must be positive).
    sigma : float, optional
        Stationary standard deviation of the OU process (default 1.0).
    mu : float, optional
        Mean value of the OU process (default 0.0).

    Returns
    -------
    np.ndarray
        Array of shape ``(n_steps, m)`` containing OU samples.
    """
    if tau <= 0.0:
        raise ValueError("OU correlation time 'tau' must be positive.")

    rho = np.exp(-dt / tau)
    innovation_scale = sigma * np.sqrt(max(0.0, 1.0 - rho ** 2))

    # Start in the stationary distribution for zero transients
    state = generator.normal(loc=mu, scale=sigma, size=m)
    samples = np.empty((n_steps, m), dtype=float)

    for idx in range(n_steps):
        innovation = generator.normal(size=m)
        state = mu + rho * (state - mu) + innovation_scale * innovation
        samples[idx] = state

    return samples
# ---------------------------------------------------------------------------------
#                                SDE Solver
# ---------------------------------------------------------------------------------

class SDESolver(Solver):
    """
    A lightweight SDE solver supporting multiple methods:
    - Ito Euler-Maruyama ('itoEuler')
    - Stratonovich Heun ('stratHeun')
    - Ito SRI2 ('itoSRI2')
    - Stratonovich SRS2 ('stratSRS2')

    Allows choosing 'white', 'powerlaw' (1/f^beta), or 'ou' noise if no
    increments are supplied (dW=None). If dW is provided, the solver uses it as-is.
    """

    def __init__(
        self,
        method: str = "itoEuler",
        step_size: float = 0.01,
        seed: Optional[int] = None,
        noise_type: str = "white",
        beta: float = 0.0,
        noise_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Parameters
        ----------
        method : str
            SDE scheme: 'itoEuler', 'stratHeun', 'itoSRI2', 'stratSRS2'
        step_size : float
            Integration time step, dt
        seed : int, optional
            rng seed for the random number generator to ensure reproducibility.
        noise_type : str
            Name of the internal noise generator if dW is not supplied.
            Supported values: 'white', 'powerlaw', 'ou'.
        beta : float
            Default 1/f exponent used when noise_type is 'powerlaw'.
        noise_kwargs : dict, optional
            Additional parameters for noise generation (e.g., {'tau': 5.0} for OU).
        """
        valid_methods = ["itoEuler", "stratHeun", "itoSRI2", "stratSRS2"]
        if method not in valid_methods:
            logger.error("Invalid method '%s'. Choose from %s.", method, valid_methods)
            raise ValueError(f"Invalid method '{method}'. Choose from {valid_methods}.")
        
        self.method = method
        self.step_size = step_size
        self.seed = seed
        self.noise_type = (noise_type or "white").lower()
        self.beta = beta
        self.noise_kwargs: Dict[str, Any] = dict(noise_kwargs or {})

        # Initialize random number generator
        self.generator = np.random.default_rng(seed)

        self._stepper = getattr(self, f"_step_{method}", None)
        if self._stepper is None:
            logger.error("Solver method '_%s' is not implemented.", self.method)
            raise NotImplementedError(f"Solver method '_{self.method}' is not implemented.")    

        # Map method name -> step function
        self._steppers = getattr(self, f"_step_{self.method}", None)

    def solve(
        self,
        model: Any,
        t_span: Tuple[float, float],
        y0: np.ndarray,
        t_eval: np.ndarray,
        dW: Optional[np.ndarray] = None,
    ) -> Solution:
        """
        Solve the SDE function using the selected stochastic integration method.
        dy = f(t,y) dt + G(t,y) dW

        Parameters
        ----------
        model : Any
            The neuron model instance containing f and G functions.
        t_span : Tuple[float, float]
            Tuple representing the start and end times (t0, tf).
        y0 : np.ndarray
            Initial state vector.
        t_eval : np.ndarray
            Array of time points at which to store the computed solutions.
        f : Callable
            Drift term, signature f(t, y) -> (d,)
        G : Callable
            Diffusion term, signature G(t, y) -> (d, m)
        y0 : np.ndarray
            Initial condition, shape (d,)
        t_eval : np.ndarray
            Array of times at which solution is desired (must be equally spaced)
        dW : np.ndarray, optional
            User-supplied increments, shape (len(t_eval)-1, m).
            If None, solver will generate increments internally
            based on noise_type.
        solver_kwargs : dict
                Additional keyword arguments for the solver.

        Returns
        -------
        Solution
            A namedtuple containing:
                - t: np.ndarray, array of time points.
                - y: np.ndarray, array of state vectors corresponding to each time point.
        """
        logger.debug(
            "Starting SDESolver.solve with method='%s', t_span=%s, y0=%s, t_eval_length=%d",
            self.method, t_span, y0, len(t_eval)
        )
        try:
            f = model.f  # Drift term, signature f(t, y) -> (d,)
            G = model.G  # Diffusion term, signature G(t, y) -> (d, m)
            d, m, f, G, y0, tspan, dW, seed = _check_args(f, G, y0, t_eval, dW, self.seed)
            dt = self.step_size
            N = len(t_eval)
                
            # Generate or reuse increments
            increments = self._noise_generator(
                n_steps = N - 1,
                m = m,
                dt = dt,
                dW_user = dW,
                model = model
                )
                       
            # Perform the integration via a generic stepper
            y = self._sde_stepper(f, G, y0, t_eval, increments, self._stepper)
            
            logger.debug(
                "SDESolver.solve completed successfully using method='%s'.",
                self.method
            )
            
            return Solution(t=t_eval, y=y.T)
        except Exception as e:
            logger.error("Exception encountered during solve: %s", e)
            raise

    # -------------------------- Internal Helpers --------------------------

    def _resolve_noise_param(self, key: str, default: Any, model: Any = None) -> Any:
        """Resolve noise-related parameters with priority: solver kwargs -> model."""
        search_candidates = []
        if self.noise_type:
            search_candidates.append(f"{self.noise_type}_{key}")
        search_candidates.append(f"noise_{key}")
        search_candidates.append(key)

        if self.noise_kwargs:
            for candidate in search_candidates:
                if candidate in self.noise_kwargs:
                    return self.noise_kwargs[candidate]

        params = getattr(model, "parameters", None)
        if isinstance(params, dict):
            noise_cfg = params.get("noise_params")
            if isinstance(noise_cfg, dict):
                for candidate in search_candidates:
                    if candidate in noise_cfg:
                        return noise_cfg[candidate]
            for candidate in search_candidates:
                if candidate in params:
                    return params[candidate]

        return default

    def _noise_generator(
        self,
        n_steps: int,
        m: int,
        dt: float,
        dW_user: Optional[np.ndarray] = None,
        model: Any = None
    ) -> np.ndarray:
        """
        Return increments of shape (n_steps, m).
        If dW_user is given, just return that.
        Otherwise generate noise according to self.noise_type.
        """
        if dW_user is not None:
            dW_arr = np.asarray(dW_user, dtype=float)
            if dW_arr.shape != (n_steps, m):
                raise ValueError(
                    f"Provided increments have shape {dW_arr.shape}, expected {(n_steps, m)}"
                )
            return dW_arr

    
        noise_type = self.noise_type

        if noise_type in ("white", "wiener", "additive", "brownian"):
            return _wiener_increments(n_steps, m, dt, self.generator)

        if noise_type in ("powerlaw", "pink", "colored"):
            beta_default = self.beta if self.beta else 1.0
            beta_value = float(self._resolve_noise_param("beta", beta_default, model))
            fmin_value = self._resolve_noise_param("fmin", None, model)

            raw = _powerlaw_noise(
                beta=beta_value,
                N=(m, n_steps),
                dt=dt,
                generator=self.generator,
                fmin=fmin_value if fmin_value is not None else 1.0 / max(n_steps, 1)
            )
            if raw.ndim == 1:
                raw = raw[np.newaxis, :]

            # Ensure zero mean and unit variance before scaling by sqrt(dt)
            raw = raw - raw.mean(axis=-1, keepdims=True)
            std = raw.std(axis=-1, keepdims=True)
            std[std == 0.0] = 1.0
            normalized = raw / std
            return normalized.T * np.sqrt(dt)

        if noise_type in ("ou", "ornstein-uhlenbeck", "ornstein_uhlenbeck"):
            tau_value = float(self._resolve_noise_param("tau", 1.0, model))
            sigma_value = float(self._resolve_noise_param("sigma", 1.0, model))
            mu_value = float(self._resolve_noise_param("mu", 0.0, model))

            ou_samples = _ornstein_uhlenbeck_sequence(
                n_steps=n_steps,
                m=m,
                dt=dt,
                generator=self.generator,
                tau=tau_value,
                sigma=sigma_value,
                mu=mu_value
            )
            centered = ou_samples - mu_value
            return centered * np.sqrt(dt)

        raise ValueError(f"Unknown noise_type '{self.noise_type}'.")

    def _sde_stepper(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        G: Callable[[float, np.ndarray], np.ndarray],
        y0: np.ndarray,
        t_eval: np.ndarray,
        increments: np.ndarray,
        step_func: Callable[[float, np.ndarray, float, Callable, Callable, np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """
        Main stepping loop, independent of integration scheme.
        step_func is a single-step function (tn, yn, dt, f, G, dWn) -> y_{n+1}.
        """
        N = len(t_eval)
        d = len(y0)
        dt = self.step_size

        y = np.empty((N, d), dtype=float)
        y[0] = y0

        for n in range(N - 1):
            tn = t_eval[n]
            yn = y[n]
            dWn = increments[n]
            y[n + 1] = step_func(tn, yn, dt, f, G, dWn)

            # Example debug logging every 1000 steps
            if n % 1000 == 0 and n > 0:
                logger.debug(f"Step {n}/{N - 1}, t={tn:.4f}")

        return y

    # -------------------------- Single-Step Methods --------------------------
    
    def _step_itoEuler(
        self,
        tn: float,
        yn: np.ndarray,
        dt: float,
        f: Callable[[float, np.ndarray], np.ndarray],
        G: Callable[[float, np.ndarray], np.ndarray],
        dWn: np.ndarray
    ) -> np.ndarray:
        """
        Euler-Maruyama solver for the Ito SDE:
            y_{n+1} = y_n + f(t_n, y_n)*dt + G(t_n, y_n) * dW_n
        """
        fy = f(tn, yn)  # shape (d,)
        G_mat = G(tn, yn)  # shape (d, m)
        return yn + fy * dt + G_mat @ dWn

    def _step_stratHeun(
        self,
        tn: float,
        yn: np.ndarray,
        dt: float,
        f: Callable[[float, np.ndarray], np.ndarray],
        G: Callable[[float, np.ndarray], np.ndarray],
        dWn: np.ndarray
    ) -> np.ndarray:
        """
        Stratonovich Heun (predictor-corrector) solver:
            y_pred = y_n + f(t_n, y_n)*dt + G(t_n, y_n)*dW_n
            y_{n+1} = y_n + 0.5 [f(t_n, y_n) + f(t_{n+1}, y_pred)] * dt
                               + 0.5 [G(t_n, y_n) + G(t_{n+1}, y_pred)] * dW_n
        """
        fy_n = f(tn, yn)
        G_n = G(tn, yn)

        y_pred = yn + fy_n * dt + G_n @ dWn

        # Evaluate at t_{n+1} = tn + dt
        fy_pred = f(tn + dt, y_pred)
        G_pred = G(tn + dt, y_pred)

        return yn + 0.5 * (fy_n + fy_pred) * dt + 0.5 * (G_n + G_pred) @ dWn

    def _step_itoSRI2(
        self,
        tn: float,
        yn: np.ndarray,
        dt: float,
        f: Callable[[float, np.ndarray], np.ndarray],
        G: Callable[[float, np.ndarray], np.ndarray],
        dWn: np.ndarray
    ) -> np.ndarray:
        """
        Ito Stochastic Runge-Kutta Integration 2 (SRI2) solver.

        Predictor:
            y_pred = y_n + f(t_n, y_n)*dt + G(t_n, y_n)*dW_n
        Corrector:
            y_{n+1} = y_n + 0.5 [f(t_n, y_n) + f(t_{n+1}, y_pred)] * dt
                                + 0.5 * G(t_n, y_n)*dW_n
        """
        fy_n = f(tn, yn)
        G_n = G(tn, yn)

        # Predictor
        y_pred = yn + fy_n * dt + G_n @ dWn

        # Evaluate f at t_{n+1}, y_pred
        fy_pred = f(tn + dt, y_pred)

        # Correct
        return yn + 0.5 * (fy_n + fy_pred) * dt + 0.5 * (G_n @ dWn)

    def _step_stratSRS2(
        self,
        tn: float,
        yn: np.ndarray,
        dt: float,
        f: Callable[[float, np.ndarray], np.ndarray],
        G: Callable[[float, np.ndarray], np.ndarray],
        dWn: np.ndarray
    ) -> np.ndarray:
        """
        Stratonovich Stochastic Runge-Kutta 2 (SRS2) solver.

        Predictor:
            y_pred = y_n + f(t_n, y_n)*dt + G(t_n, y_n)*dW_n
        Corrector:
            y_{n+1} = y_n + 0.5 [f(t_n, y_n) + f(t_{n+1}, y_pred)] * dt
                                + 0.5 [G(t_n, y_n) + G(t_{n+1}, y_pred)] * dW_n
        Note: This is equivalent to Stratonovich Heun in many references, but provided here for flexibility.
        """
        fy_n = f(tn, yn)
        G_n = G(tn, yn)

        y_pred = yn + fy_n * dt + G_n @ dWn

        # Evaluate at t_{n+1} = tn + dt
        fy_pred = f(tn + dt, y_pred)
        G_pred = G(tn + dt, y_pred)

        return yn + 0.5 * (fy_n + fy_pred) * dt + 0.5 * (G_n + G_pred) @ dWn

