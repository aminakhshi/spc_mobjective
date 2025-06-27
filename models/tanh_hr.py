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

class Tanh_HR(NeuronModel):
    def __init__(self, solver: Any, **params):
        """
        Initialize the Tanh Hindmarsh-Rose Model with default and user-provided parameters.

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
            'x_R': -1.6,
            'I_app': 1.34,     
            'k_n': 11.5,
            'x_v': 0.5,
            'ninf_c': 0.5,
            'k_h': 15.0,
            'x_z': 0.5,
            'hinf_c': 0.5,
            'alpha_z': 0.01,
            'gamma_u': 0.8,      # Rate of increase during burst
            'gamma_d': 0.05,      # Rate of decay after burst
            # Noise parameters for diffusion
            'sigma_x': 0.05,
            'sigma_y': 0.0,
            'sigma_z': 0.01,
            'sigma_u': 0.05,
        }
        # Update default parameters with user-provided parameters
        default_params.update(params)

        super().__init__(solver, **default_params)
        logger.debug("Tanh_HR model parameters after update: %s", self.parameters)

    def set_parameters(self, **params):
        """
        Update model parameters.

        Parameters:
        -----------
        **params : dict
            Arbitrary keyword arguments to update model parameters.
        """
        logger.debug("Updating Tanh HR model parameters with: %s", params)
        self.parameters.update(params)

    @staticmethod
    def hz_inf(z, parameters, mode='sigmoid'):
        """
        Activation gating function based on z. It can choose between sigmoid or tanh

        Returns:
        --------
        float or np.ndarray
            Gating function output between 0 and 1 (if sigmoid) or -1 and 1 (if tanh)
        """
        k_h = parameters.get('k_h', 15.0)
        x_z = parameters.get('x_z', 0.5)
        hinf_c = parameters.get('hinf_c', 0.5)
        
        if mode == 'tanh':
            activation = hinf_c * (1 + np.tanh( k_h * (z - x_z)))
            return activation
        
        elif mode == 'sigmoid':
            activation = hinf_c / (1 + np.exp(- k_h * (z - x_z)))
            return activation

    @staticmethod
    def nv_inf(x, parameters, mode='sigmoid'):
        """
        Activation gating function based on x. It can choose between sigmoid or tanh

        Returns:
        --------
        float or np.ndarray
            Gating function output between 0 and 1 (if sigmoid) or -1 and 1 (if tanh)
        """
        k_n = parameters.get('k_n', 15.0)
        x_v = parameters.get('x_v', 0.5)
        ninf_c = parameters.get('ninf_c', 0.5)

        if mode == 'tanh':
            activation = ninf_c * (1 + np.tanh(k_n * (x - x_v)))
            return activation
        
        elif mode == 'sigmoid':
            activation = ninf_c / (1 + np.exp(- k_n * (x - x_v)))
            return activation
    
    def f(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Compute the deterministic part of the Tanh Hindmarsh-Rose model equations.

        Parameters:
        -----------
        t : float
            Current time.
        y : np.ndarray
            Current values of [x, y, z, u, v].

        Returns:
        --------
        np.ndarray
            Derivatives [dx/dt, dy/dt, dz/dt, du/dt, dv/dt].
        """
        logger.debug("Simulating at time t=%.6f with state y=%s", t, y)
        try:
            x, y_var, z, u = y

            # Activation function to smoothly transition u
            nv = self.nv_inf(x, self.parameters)
            hz = self.hz_inf(z, self.parameters)
            # Differential equations
            dx_dt = y_var - self.parameters['a'] * x**3 + self.parameters['b'] * x**2 - z - u + self.parameters['I_app']
            dy_dt = self.parameters['c'] - self.parameters['d'] * x**2 - y_var
            dz_dt = self.parameters['r'] * (self.parameters['s'] * (x - self.parameters['x_R']) - z)            
            du_dt = - self.parameters['gamma_u'] * nv + self.parameters['alpha_z'] * hz - self.parameters['gamma_d'] * u 

            derivatives = np.array([dx_dt, dy_dt, dz_dt, du_dt])
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
            sigma_x = self.parameters.get('sigma_x', 0.0)
            sigma_y = self.parameters.get('sigma_y', 0.0)
            sigma_z = self.parameters.get('sigma_z', 0.0)
            sigma_u = self.parameters.get('sigma_u', 0.0)

            G_matrix = np.zeros((n_var, n_var))
            G_matrix[0, 0] = sigma_x  # diffusion element for x
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
        This method is used when the problem is not SDE and utilizes the f(t, y) method. See f(t, y) for details.
        """
        return self.f(t, y)
    
    def get_num_state_vars(self) -> int:
        """
        Return the number of state variables for the model.

        Returns:
        --------
        int
            Number of state variables (4: x, y, z, u).
        """
        logger.debug("Returning number of state variables: 4")
        return 4 
