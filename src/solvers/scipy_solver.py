"""
This file contains the implementation different solvers in the package. Available models are:
- RK45
- Euler method
- Euler-Maruyama

## TODO: adding more solvers. 
 
copyright: 2024, @aminakhshi
"""

import logging
from typing import Any, Dict, Tuple
import numpy as np
from scipy.integrate import solve_ivp
from collections import namedtuple

from solvers.base_solver import Solver

logger = logging.getLogger(__name__)
Solution = namedtuple('Solution', ['t', 'y'])

class ScipySolver(Solver):
    def __init__(self, method: str = 'Radau', **kwargs):
        """
        Initializes the ScipySolver with optional solver keyword arguments.

        Parameters:
        -----------
        method : str, optional
            The integration method to use. Defaults to 'Radau'.
        **kwargs: Keyword arguments for scipy's solve_ivp.
        """
        logger.debug("Initializing ScipySolver with method='%s' and kwargs=%s", method, kwargs)
        self.method = method
        self.solver_kwargs = kwargs

    def solve(
        self,
        model: Any,
        t_span: Tuple[float, float],
        y0: np.ndarray,
        t_eval: np.ndarray,
        **additional_kwargs
    ) -> Solution:
        """
        Solve the given model dynamics over time using scipy's solve_ivp.

        Parameters:
        -----------
        model : NeuronModel
            The model to simulate.
        t_span : tuple
            Start and end time.
        y0 : np.ndarray
            Initial conditions.
        t_eval : np.ndarray
            Time points to evaluate the solution.
        **additional_kwargs : dict
            Additional keyword arguments for solve_ivp.

        Returns:
        --------
        Solution
            Time points (t) and the solution (y) at those time points.
        """
        logger.debug(
            "Starting solve_ivp with t_span=%s, y0=%s, t_eval_length=%d, additional_kwargs=%s",
            t_span, y0, len(t_eval), additional_kwargs
        )
        try:
            # Update solver_kwargs with additional_kwargs
            kwargs = {**self.solver_kwargs, **additional_kwargs}
            kwargs['method'] = self.method  # Ensure the method is set correctly
            logger.debug("Combined solver kwargs: %s", kwargs)

            result = solve_ivp(model.simulate, t_span=t_span, y0=y0, t_eval=t_eval, **kwargs)
            if not result.success:
                logger.error("solve_ivp failed: %s", result.message)
                raise RuntimeError(f"solve_ivp failed: {result.message}")

            logger.debug("solve_ivp completed successfully.")
            return Solution(t=result.t, y=result.y)
        except Exception as e:
            logger.critical("Failed to solve model using solve_ivp: %s", e)
            raise RuntimeError(f"Failed to solve model using solve_ivp: {e}") from e