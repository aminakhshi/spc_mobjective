"""
Class file for the implementation of the GhostBurster model. This model is based on the following paper:

- Doiron, Brent, et al. "Ghostbursting: a novel neuronal burst mechanism." Journal of computational neuroscience 12 (2002): 5-25.

copyright: 2024, @aminakhshi
"""

import logging
import os
from typing import Any, Dict
import numpy as np
from models.base_model import NeuronModel
from solvers.base_solver import Solver
from collections import namedtuple

logger = logging.getLogger(__name__)

class GhostBurst(NeuronModel):
    def __init__(self, solver: Solver, **params):
        """
        Defaul values are provided from Doiron et al. (2002)
        """
        default_params = {
           'Iapp': 9.0, 'C_m': 1.0, 'kappa': 0.4, 'gc': 1.0,
           'gNaS': 55.0, 'gDrS': 20.0, 'gleak': 0.18,
           'gNaD': 5.0, 'gDrD': 15.0,
           'VNa': 40.0, 'VK': -88.5, 'Vleak': -70.0,
           'tau_ns': 0.39,
           'tau_hd': 1.0, 'tau_nd': 0.9, 'tau_pd': 5.0,
           'V_ms': 40.0, 's_ms': 3.0,
           'V_ns': 40.0, 's_ns': 3.0,
           'V_md': 40.0, 's_md': 5.0,
           'V_hd': 52.0, 's_hd': 5.0,
           'V_nd': 40.0, 's_nd': 5.0,
           'V_pd': 65.0, 's_pd': 6.0,
            # Noise parameters for diffusion
            'sigma_Vs': 0.05, 'sigma_Vd': 0.0,
            'sigma_ns': 0.00,
            'sigma_hd': 0.00, 'sigma_nd': 0.0, 'sigma_pd': 0.0
            }
        # Update default parameters with user-provided parameters
        default_params.update(params)

        super().__init__(solver, **default_params)
        logger.debug("Ghostburst HH model parameters after update: %s", self.parameters)

    def set_parameters(self, **params):
        """
        Update model parameters.

        Parameters:
        -----------
        **params : dict
            Arbitrary keyword arguments to update model parameters.
        """
        logger.info("Updating Ghostburst HH model parameters with: %s", params)
        self.parameters.update(params)

    def ms_inf(self, V):
        """ steady-state conductance activation curve for Na+ channel in the Soma. """
        msi = 1 / (1 + np.exp(-(V + self.parameters['V_ms']) / self.parameters['s_ms']))
        logger.debug("SS activation for Na+ in the Soma.: %s", msi)
        return msi
    
    def ns_inf(self, V):
        """ steady-state conductance inactivation curve for K+ channel in the Soma. """
        nsi = 1 / (1 + np.exp(-(V + self.parameters['V_ns']) / self.parameters['s_ns']))
        logger.debug("SS inactivation for K+ in the Soma.: %s", nsi)
        return nsi
    
    def md_inf(self, V):
        """ steady-state conductance activation curve for Na+ channel in the Dendrite. """
        mdi = 1 / (1 + np.exp(-(V + self.parameters['V_md']) / self.parameters['s_md']))
        logger.debug("SS activation for Na+ in the Dendrite.: %s", mdi)
        return mdi
    
    def hd_inf(self, V):
        """ steady-state conductance inactivation curve for Na+ channel in the Dendrite. """ 
        hdi = 1 / (1 + np.exp((V + self.parameters['V_hd']) / self.parameters['s_hd']))
        logger.debug("SS inactivation for Na+ in the Dendrite.: %s", hdi)
        return hdi
    
    def nd_inf(self, V):
        """ steady-state conductance activation curve for K+ channel in the Dendrite. """ 
        ndi = 1 / (1 + np.exp(-(V + self.parameters['V_nd']) / self.parameters['s_nd']))
        logger.debug("SS activation for K+ in the Dendrite.: %s", ndi)
        return ndi
    
    def pd_inf(self, V):
        """ steady-state conductance inactivation curve for K+ channel in the Dendrite."""
        pdi = 1 / (1 + np.exp((V + self.parameters['V_pd']) / self.parameters['s_pd']))
        logger.debug("SS inactivation for K+ in the Dendrite.: %s", pdi)
        return pdi

    def f(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Compute the deterministic part of the GhostBurst model equations.

        Parameters:
        -----------
        t : float
            Current time.
        y : np.ndarray
            State variable values of [Vs, Vd, ns, hd, nd, pd].

        Returns:
        --------
        np.ndarray
            Derivatives [dVs/dt, dVd/dt, dns/dt, dhd/dt, dnd/dt, dpd/dt].
        """
        logger.debug("Simulating at time t=%.6f with state y=%s", t, y)
        try:
            Vs, Vd, ns, hd, nd, pd = y
            
            # Ionic currents in soma and dendrite
            I_NaS = self.parameters['gNaS'] * (self.ms_inf(Vs) ** 2) * (1 - ns) * (Vs - self.parameters['VNa'])
            I_DrS = self.parameters['gDrS'] * (ns ** 2) * (Vs - self.parameters['VK'])
            I_leakS = self.parameters['gleak'] * (Vs - self.parameters['Vleak'])
            I_conS = (self.parameters['gc'] / self.parameters['kappa']) * (Vs - Vd)
            I_NaD = self.parameters['gNaD'] * (self.md_inf(Vd) ** 2) * hd * (Vd - self.parameters['VNa'])
            I_DrD = self.parameters['gDrD'] * (nd ** 2) * pd * (Vd - self.parameters['VK'])
            I_leakD = self.parameters['gleak'] * (Vd - self.parameters['Vleak'])
            I_conD = (self.parameters['gc'] / (1 - self.parameters['kappa'])) * (Vd - Vs)
            
            # Differential equations for voltage and gating variables
            derivatives = [
                (self.parameters['Iapp'] - I_NaS - I_DrS - I_leakS - I_conS) / self.parameters['C_m'],
                (-I_NaD - I_DrD - I_leakD - I_conD) / self.parameters['C_m'],
                (self.ns_inf(Vs) - ns) / self.parameters['tau_ns'],
                (self.hd_inf(Vd) - hd) / self.parameters['tau_hd'],
                (self.nd_inf(Vd) - nd) / self.parameters['tau_nd'],
                (self.pd_inf(Vd) - pd) / self.parameters['tau_pd']
                ]
            logger.debug("Computed derivatives: %s", derivatives)
            return np.array(derivatives)
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
            sigma_vs = self.parameters.get('sigma_Vs', 0.0)
            sigma_vd = self.parameters.get('sigma_Vd', 0.0)
            sigma_ns = self.parameters.get('sigma_ns', 0.0)
            sigma_hd = self.parameters.get('sigma_hd', 0.0)
            sigma_nd = self.parameters.get('sigma_nd', 0.0)
            sigma_pd = self.parameters.get('sigma_pd', 0.0)

            G_matrix = np.zeros((n_var, n_var))
            G_matrix[0, 0] = sigma_vs  # diffusion element for Vs
            G_matrix[1, 1] = sigma_vd  # diffusion element for Vd
            G_matrix[2, 2] = sigma_ns  # diffusion element for ns
            G_matrix[3, 3] = sigma_hd  # diffusion element for hd
            G_matrix[4, 4] = sigma_nd  # diffusion element for nd
            G_matrix[5, 5] = sigma_pd  # diffusion element for pd

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
        """Return the number of state variables for the model."""
        logger.debug("Returning number of state variables: 6")
        return 6  # Vs, Vd, ns, hd, nd, pd