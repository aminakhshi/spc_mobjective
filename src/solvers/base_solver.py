"""
This file contains the implementation different solvers in the package. Available models are:
- RK45
- Euler method
- Euler-Maruyama

## TODO: adding more solvers. 
 
copyright: 2024, @aminakhshi
"""
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class Solver(ABC):
    @property
    def use_jax(self) -> bool:
        """Indicates whether the solver uses JAX."""
        return False  # Default to False for NumPy-based solvers

    @abstractmethod
    def solve(self, model, t_span, y0, t_eval):
        """Solve the given model dynamics over time."""
        pass