import logging
import numpy as np
from collections import namedtuple
from typing import Callable, Tuple, Optional, Union, List, Any, Iterable
from solvers.base_solver import Solver

logger = logging.getLogger(__name__)

Solution = namedtuple('Solution', ['t', 'y'])


def _check_args(
    F: Callable[[float, np.ndarray], np.ndarray],
    G: Union[Callable, List[Callable]],
    y0: np.ndarray,
    tspan: np.ndarray,
    dW: Optional[np.ndarray],
    seed: Optional[int] = None
) -> Tuple[int, int, Callable, Union[Callable, List[Callable]], np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
    """
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
        N: int or Iterable[int],
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
    
    Examples:
    ---------
    # Generate 1/f noise (pink noise)
    >>> import numpy as np
    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> y = _powerlaw_noise(beta=1, N=1000, dt=1.0, generator=rng)
    """
    try:
        size = list(N)
    except TypeError:
        size = [N]
    
    # The number of samples in each time series
    samples = size[-1]
    
    f = np.fft.rfftfreq(samples, dt)
    
    s_scale = f.copy()
    fmin = max(fmin, 1.0 / samples)  # Low frequency cutoff
    ix = np.sum(s_scale < fmin)      
    if ix > 0 and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale**(-beta / 2.0)
    
    w = s_scale[1:].copy()
    if samples % 2 == 0:
        w[-1] *= 0.5  # Correct f = +-0.5 for even sample counts
    sigma = 2 * np.sqrt(np.sum(w**2)) / samples
    
    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)
    
    dims_to_add = len(size) - 1
    s_scale = s_scale[(np.newaxis,) * dims_to_add + (...,)]
    
    # Generate scaled random power + phase using the provided generator
    sr = generator.normal(scale=s_scale, size=size)
    si = generator.normal(scale=s_scale, size=size)
    
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

class CustomSDE(Solver):
    """
    A lightweight SDE solver supporting multiple methods:
    - Ito Euler-Maruyama ('itoEuler')
    - Stratonovich Heun ('stratHeun')
    - Ito SRI2 ('itoSRI2')
    - Stratonovich SRS2 ('stratSRS2')

    Allows choosing 'white' or 'powerlaw' noise if no increments are supplied
    (dW=None). If dW is provided, the solver uses it as-is.
    """

    def __init__(
        self,
        method: str = "itoEuler",
        step_size: float = 0.01,
        seed: Optional[int] = None,
        noise_type: str = "white",
        beta: float = 0.0
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
            'white' or 'powerlaw'
        beta : float
            noise exponent (used if noise_type !='white')
        """
        valid_methods = ["itoEuler", "stratHeun", "itoSRI2", "stratSRS2"]
        if method not in valid_methods:
            logger.error("Invalid method '%s'. Choose from %s.", method, valid_methods)
            raise ValueError(f"Invalid method '{method}'. Choose from {valid_methods}.")

        self.method = method
        self.step_size = step_size
        self.seed = seed
        self.noise_type = noise_type
        self.beta = beta

        # Initialize random number generator
        self.generator = np.random.default_rng(seed)

        if self.method not in valid_methods:
            logger.error("Unsupported method '%s'.", self.method)
            raise ValueError(f"Unsupported method '{self.method}'.")

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
            f = model.f  # Drift term
            G = model.G  # Diffusion term
            d, m, f, G, y0, tspan, dW, seed = _check_args(f, G, y0, t_eval, dW, self.seed)
            dt = self.step_size
            N = len(t_eval)

            ##TODO: fix the spacing of t_eval to avoid this warning
            # # Ensure that t_eval is equally spaced and matches step_size
            # expected_N = int(np.ceil((tspan[1] - tspan[0]) / dt)) + 1
            # if not np.isclose(len(t_eval), expected_N):
            #     logger.warning(
            #         "t_eval length %d does not match expected number of steps %d based on step_size=%.5f",
            #         len(t_eval), expected_N, dt
            #     )
            
                
            # Generate or reuse increments
            increments = self._noise_generator(
                n_steps = N - 1,
                m = m,
                dt = dt,
                dW_user = dW,
                model = model
                )
                       
            y = self._sde_stepper(f, G, y0, t_eval, increments, self._stepper)
            
            logger.debug(
                "SDESolver.solve completed successfully using method='%s'.",
                self.method
            )
            
            return Solution(t=t_eval, y=y.T)
        except Exception as e:
            logger.error("Exception encountered during solve: %s", e)
            raise

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
            return dW_user

        if model.__class__.__name__ == "SPCvivo":
            seed = 0 if self.seed is None else self.seed
            self.beta = model.parameters['beta'] if 'beta' in model.parameters and model.parameters['beta'] is not None else self.beta
            rng0 = np.random.default_rng(seed)
            rng1 = np.random.default_rng(seed + 1)
            rng2 = np.random.default_rng(seed + 2)
            dW = np.zeros((n_steps, 3))
            # White col 0
            dW[:, 0] = rng0.normal(0.0, np.sqrt(dt), size=n_steps)
            # White col 1
            dW[:, 1] = rng1.normal(0.0, np.sqrt(dt), size=n_steps)
            # Power-law col 2
            dW[:, 2] = _powerlaw_noise(self.beta, n_steps, dt, rng2)
            return dW
    
        if self.noise_type == "white":
            return _wiener_increments(n_steps, m, dt, self.generator)
        
        elif self.noise_type == "powerlaw":
            array_list = []
            for _ in range(m):
                arr = _powerlaw_noise(self.beta, n_steps, dt, self.generator)
                array_list.append(arr)
            return np.column_stack(array_list)
        else:
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

            if n % 1000 == 0 and n > 0:
                logger.debug(f"Step {n}/{N - 1}, t={tn:.4f}")

        return y

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

