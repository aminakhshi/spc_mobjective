# solvers/helper_functions.py

import numpy as np
import logging

logger = logging.getLogger(__name__)

class SDEValueError(Exception):
    """Custom exception for SDE solver errors."""
    pass

def _check_args(f, G, y0, tspan, dW, IJ):
    """
    Validate and process arguments for stochastic solvers.
    
    Args:
        f: Callable representing the deterministic part.
        G: Callable or list of callables representing the stochastic part.
        y0: Initial condition array.
        tspan: Array of time points.
        dW: Optional array of Wiener increments.
        IJ: Optional array of repeated integrals.
    
    Returns:
        Tuple containing processed arguments.
    
    Raises:
        SDEValueError: If input arguments are invalid.
    """
    if not callable(f):
        raise SDEValueError("f must be a callable function returning an array.")
    if not (callable(G) or isinstance(G, list)):
        raise SDEValueError("G must be a callable function or a list of callables.")
    y0 = np.asarray(y0)
    if y0.ndim != 1:
        raise SDEValueError("y0 must be a 1-dimensional array.")
    tspan = np.asarray(tspan)
    if tspan.ndim != 1 or len(tspan) < 2:
        raise SDEValueError("tspan must be a 1-dimensional array with at least two points.")
    d = len(y0)
    if isinstance(G, list):
        m = len(G)
        for g in G:
            if not callable(g):
                raise SDEValueError("All elements in G list must be callable functions.")
    else:
        G = np.asarray(G)
        if G.ndim != 2:
            raise SDEValueError("G must return a 2-dimensional array with shape (d, m).")
        d, m = G.shape
    if d != len(y0):
        raise SDEValueError("Dimension of y0 does not match the first dimension of G.")
    return (d, m, f, G, y0, tspan, dW, IJ)

def deltaW(N, m, h, generator=None):
    """
    Generate Wiener increments.
    
    Args:
        N: Number of time steps.
        m: Number of independent Wiener processes.
        h: Time step size.
        generator: NumPy random generator instance.
    
    Returns:
        Array of shape (N, m) with Wiener increments.
    """
    if generator is None:
        generator = np.random.default_rng()
    return generator.normal(loc=0.0, scale=np.sqrt(h), size=(N, m))
