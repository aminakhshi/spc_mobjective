"""
Class file for the implementation of the Modified Hindmarsh-Rose model.

Author: 2024, @aminakhshi
"""

import logging
import os
from typing import Any, Dict
import numpy as np
from models.base_model import NeuronModel
from solvers.base_solver import Solver
from collections import namedtuple

logger = logging.getLogger(__name__)

class Modified_HR(NeuronModel):
    def __init__(self, solver: Any, **params):
        """
        Initialize the Modified Hindmarsh-Rose Model with default and user-provided parameters.

        Parameters:
        -----------
        solver : Solver
            An instance of the solver to be used for simulation.
        **params : dict
            Arbitrary keyword arguments to update model parameters.
        """
        # Default parameters
        default_params = {
            'a': 1.0,
            'b': 3.0,
            'c': 1.0,
            'd': 5.0,
            'r': 0.005,
            's': 4.0,
            'v_R': -1.6,
            'I_app': 1.34, 
            'c_n': 0.5,
            'k_n': 0.06,
            'x_v': 0.5,
            'c_h': 0.5,
            'k_h': 0.13,
            'x_z': 0.5,
            'alpha_z': 0.2,
            'gamma_u': 0.9,      # Rate of increase during burst
            'gamma_d': 0.05,      # Rate of decay after burst
            # Noise parameters for diffusion
            'sigma_v': 0.05,
            'sigma_y': 0.0,
            'sigma_z': 0.01,
            'sigma_u': 0.05,
        }
        # Update default parameters with user-provided parameters
        default_params.update(params)

        super().__init__(solver, **default_params)
        logger.debug("Modified_HR model parameters after update: %s", self.parameters)

    def set_parameters(self, **params):
        """
        Update model parameters.

        Parameters:
        -----------
        **params : dict
            Arbitrary keyword arguments to update model parameters.
        """
        logger.debug("Updating Modified HR model parameters with: %s", params)
        self.parameters.update(params)
        # logger.debug("Modified_HR model parameters after update: %s", self.parameters)

    @staticmethod
    def hinf_z(z, parameters, mode='tanh'):
        """
        Inactivation gating function based on z. 
        
        Returns:
        --------
        float or np.ndarray
            Gating function output 
        """
        k_h = parameters.get('k_h', 0.13)
        x_z = parameters.get('x_z', 0.5)
        c_h = parameters.get('c_h', 0.3)

        
        if mode == 'tanh':
            hz = c_h * (1 + np.tanh((z - x_z) / k_h))
            return hz

        elif mode == 'sigmoid':
            hz = c_h / (1 + np.exp(- k_h * (z - x_z)))
            return hz

    @staticmethod
    def ninf_v(v, parameters, mode='tanh'):
        """
        Activation gating function based on v.

        Returns:
        --------
        float or np.ndarray
            Gating function output
        """
        k_n = parameters.get('k_n', 0.067)
        x_v = parameters.get('x_v', 0.5)
        c_n = parameters.get('c_n', 0.6)

        if mode == 'tanh':
            nv = c_n * (1 + np.tanh((v - x_v) / k_n))
            return nv
        
        elif mode == 'sigmoid':
            nv = c_n / (1 + np.exp(- k_n * (v - x_v)))
            return nv
    
    def f(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Compute the deterministic part of the Modified Hindmarsh-Rose model equations.

        Parameters:
        -----------
        t : float
            Current time.
        y : np.ndarray
            Current values of [v, y, z, u].

        Returns:
        --------
        np.ndarray
            Derivatives [dv/dt, dy/dt, dz/dt, du/dt].
        """
        # logger.debug("Simulating at time t=%.6f with state y=%s", t, y)
        try:
            vs, ys, zs, us = y

            nv = self.ninf_v(vs, self.parameters)
            hz = self.hinf_z(zs, self.parameters)
            # Differential equations
            dvdt = ys - self.parameters['a'] * vs**3 + self.parameters['b'] * vs**2 - zs - us + self.parameters['I_app']
            dydt = self.parameters['c'] - self.parameters['d'] * vs**2 - ys
            dzdt = self.parameters['r'] * (self.parameters['s'] * (vs - self.parameters['v_R']) - zs)            
            dudt = - self.parameters['gamma_u'] * nv + self.parameters['alpha_z'] * hz - self.parameters['gamma_d'] * us 

            derivatives = np.array([dvdt, dydt, dzdt, dudt])
            logger.debug("Computed derivatives: %s", derivatives)
            return derivatives
        except Exception as e:
            logger.critical("Error during simulation at time t=%.6f with state y=%s: %s", t, y, e, exc_info=True)
            raise

    def G(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Compute the stochastic part (diffusion matrix) of the model equations.

        Parameters:
        -----------
        t : float
            Current time.
        y : np.ndarray
            State variable values.

        Returns:
        --------
        np.ndarray
            Diffusion matrix of shape (n_var, n_var).
        """
        logger.debug("Computing diffusion matrix at time t=%.6f with state y=%s", t, y)
        try:
            n_var = self.get_num_state_vars()
            # Retrieve noise parameters
            sigma_v = self.parameters.get('sigma_v', 0.0)
            sigma_y = self.parameters.get('sigma_y', 0.0)
            sigma_z = self.parameters.get('sigma_z', 0.0)
            sigma_u = self.parameters.get('sigma_u', 0.0)

            G_matrix = np.zeros((n_var, n_var))
            G_matrix[0, 0] = sigma_v  # diffusion element for v
            G_matrix[1, 1] = sigma_y  # diffusion element for y
            G_matrix[2, 2] = sigma_z  # diffusion element for z
            G_matrix[3, 3] = sigma_u  # diffusion element for u

            logger.debug("Computed Diffusion Matrix: %s", G_matrix)
            return G_matrix
        except Exception as e:
            logger.critical(
                "Error during diffusion computation at time t=%.6f with state y=%s: %s",
                t, y, e, exc_info=True
            )
            raise

    def simulate(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Wrapper function for computing the deterministic derivatives of the model at time t for state y.
        Used when the problem is not SDE and utilizes the f(t, y) method. See f(t, y) for details.
        """
        return self.f(t, y)
    
    def get_num_state_vars(self) -> int:
        """
        Return the number of state variables for the model.

        Returns:
        --------
        int
            Number of state variables (4: v, y, z, u).
        """
        logger.debug("Returning number of state variables: 4")
        return 4 
