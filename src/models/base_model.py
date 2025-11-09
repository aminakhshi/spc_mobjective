"""
This file contains the base class for the implementation of neuronal models. Available models are:
- Hodgkin-Huxley
- Leaky Integrate-and-Fire
- Izhikevich
- Morris-Lecar
- FitzHugh-Nagumo
- Hindmarsh-Rose
## TODO: adding more models. 
 
copyright: 2024, @aminakhshi
"""

import numpy as np
from abc import ABC, abstractmethod
from solvers.base_solver import Solver
import logging

logger = logging.getLogger(__name__)

class NeuronModel(ABC):
    def __init__(self, solver: Solver, **params):
        self.parameters = params
        self.solver = solver
        self.additional_args = None 
    
    @abstractmethod
    def set_parameters(self, **params):
        """Set the parameters for the neuron model."""
        pass
    
    @abstractmethod
    def simulate(self, t: float, y: np.ndarray) -> np.ndarray:
        """Compute the derivatives of the model at time t for state y."""
        pass
    
    @abstractmethod
    def get_num_state_vars(self) -> int:
        """Return the number of state variables for the model."""
        pass
    
    def set_args(self, *args):
        """
        Store additional arguments for the model's simulation.
        These can be used within the simulate method as needed.
        """
        self.additional_args = args
