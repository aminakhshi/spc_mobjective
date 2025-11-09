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

class SPCvivo(NeuronModel):
    def __init__(self, solver: Solver, **params):
        """
        Defaul values are provided from Doiron et al. (2002)
        """
        default_params = {
           'Iapp': 9.0, 'C_m': 1.0, 'kappa': 0.4, 'gc': 1.0,
           'gNaS': 55.0, 'gDrS': 20.0, 'gleak': 0.25,
           'gNaD': 5.0, 'gDrD': 15.0, 'gSK': 1.5, 'gNMDA': 10.1,
           'VNa': 40.0, 'VK': -88.5, 'Vleak': -48.0, 'VCa': 100.0,
           'tau_ns': 0.39,
           'tau_hd': 1.0, 'tau_nd': 0.9, 'tau_pd': 5.0,
           'tau_sd': 1.1, 'kCa': 0.4,
           'V_ms': 40.0, 's_ms': 3.0,
           'V_ns': 40.0, 's_ns': 3.0,
           'V_md': 40.0, 's_md': 5.0,
           'V_hd': 52.0, 's_hd': 5.0,
           'V_nd': 40.0, 's_nd': 5.0,
           'V_pd': 65.0, 's_pd': 6.0,
           'R_b': 45.9, 'R_u': 12.9,
           'R_o': 46.5, 'R_c': 73.8,
           'R_r': 6.8, 'R_d': 8.4,
           'Mg_o': 2.0,
           'nu_PMCA': 30.0, 'nu_Serca': 22.5,
           'k_PMCA': 0.45, 'k_Serca': 0.105,
           'nu_INleak': 0.03, 'nu_ERleak': 0.03,
           'nu_IP3': 15.0,
           'd_1': 0.13,
           'd_2': 1.049,
           'd_3': 0.9434,
           'd_5': 0.08234,
           'a2': 0.2,
           'IP3': 0.3,
           'f_c': 0.05,
           'f_ER': 0.025,
           'gamma': 9.0,
           'alpha': 0.3,
           'beta': 1.8,
           'lambda_glu': 30.0,
           'tau_glu': 5,
           'v_threshold': 0.9,
           'reset_voltage': 0.0,
           'refractory': 0.5,
           'tau_pre_neuron': 1.0,
           'sigma_d': 1.0,
            # Noise parameters for diffusion
            'sigma_Vs': 0.05, 'sigma_Vd': 0.05,
            'dt': 0.01
            }

        # Update default parameters with user-provided parameters
        default_params.update(params)
        self.record_pre = params.get("record_pre", False)
        self.pre_spikes_history = []
        self.glu_history = []

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
        """ steady-state conductance activation curve for K+ channel in the Soma. """
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

    def sd_inf(self, Ca):
        """Steady-state conductance activation curve for SK channel in the Dendrite."""
        sdi = 0.81 * (Ca ** 5 / (Ca ** 5 + self.parameters['kCa'] ** 5))
        logger.debug("SS activation for SK in the Dendrite.: %s", sdi)
        return sdi
    
    @property
    def Q2(self):
        """Calcium influx rate."""
        q2 = self.parameters['d_2'] * (
            (self.parameters['IP3'] + self.parameters['d_1']) /
            (self.parameters['IP3'] + self.parameters['d_3'])
            )
        logger.debug("Calcium influx rate: %s", q2)
        return q2
    
    @property
    def minf_IP3(self):
        """Steady-state activation curve for IP3R."""
        mip3 = self.parameters['IP3'] / (self.parameters['IP3'] + self.parameters['d_1'])
        logger.debug("Steady-state activation curve for IP3R: %s", mip3)
        return mip3
    
    def ninf_IP3(self, Ca):
        """Steady-state activation curve for Calcium."""
        nip3 = Ca / (Ca + self.parameters['d_5'])
        logger.debug("Steady-state activation curve for Calcium: %s", nip3)
        return nip3
    
    def hinf_IP3(self, Ca):
        """Steady-state inactivation curve for IP3R."""
        hip3 = self.Q2 / (self.Q2 + Ca)
        logger.debug("Steady-state inactivation curve for IP3R: %s", hip3)
        return hip3
    
    def B(self, V):
        """Magnesium block function."""
        bmg = 1 / (1 + (self.parameters['Mg_o'] * np.exp(-0.062 * V)) / 3.57)
        logger.debug("Magnesium block function: %s", bmg)
        return bmg
    
    def poisson_input(self, time_scale=1e-3):
        return np.random.rand() < self.parameters['lambda_glu'] * self.parameters['dt'] * time_scale
    
    def pre_neuron(self, t, **kwargs):
        # tau_pre_neuron = kwargs.get('tau_pre_neuron', self.tau_pre_neuron)
        # v_threshold = kwargs.get('v_threshold', self.v_threshold)
        # refractory = kwargs.get('refractory', self.refractory)
        # reset_voltage = kwargs.get('reset_voltage', self.reset_voltage)

        if not hasattr(self, 'last_spike'):
            self.last_spike = -np.inf
        if not hasattr(self, 'v_pre_neuron'):
            self.v_pre_neuron = 0
        if t - self.last_spike > self.parameters['refractory']:
            self.v_pre_neuron += -self.v_pre_neuron / self.parameters['tau_pre_neuron'] * self.parameters['dt']
            if self.poisson_input():
                # self.poisson_input_record[t] = 1
                self.v_pre_neuron += 1.0
            # self.v_pre_neuron += dv
            if self.v_pre_neuron > self.parameters['v_threshold']:
                self.v_pre_neuron = self.parameters['reset_voltage']
                self.last_spike = t
        else:
            self.v_pre_neuron = self.parameters['reset_voltage']
        
        if self.record_pre and  t == self.last_spike:
            self.pre_spikes_history.append(t)
        return self.v_pre_neuron
    
    def glurelease(self, t, **kwargs):
        if not hasattr(self, 'g_glu'):
            self.g_glu = 0
        self.g_glu += (-self.g_glu / self.parameters['tau_glu']) * self.parameters['dt']
        if t == self.last_spike:
            self.g_glu += 0.3
        if self.record_pre:
            self.glu_history.append(self.g_glu)
        return self.g_glu

    # def synaptic_in(self, dw, t):
    #     if callable(dw):
    #         I_syn = self.parameters['sigma_d'] * dw(t)
    #     else:
    #         I_syn = self.parameters['sigma'] * dw
    #     return I_syn

    def f(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Compute the deterministic part of the GhostBurst model equations.

        Parameters:
        -----------
        t : float
            Current time.
        y : np.ndarray
            [
            Vs, Vd, ns, hd, nd, pd, sd, Ca, CaER, hca, 
            C0, C1, C2, Ds, Os
            ].
            
        Returns:
        --------
        np.ndarray
            Derivatives of the same length.
        """
        logger.debug("Simulating at time t=%.6f with state y=%s", t, y)
        try:
            Vs, Vd, ns, hd, nd, pd, sd, Ca, CaER, hca, C0, C1, C2, Ds, Os = y

            self.pre_neuron(t)
            glu = self.glurelease(t)
            # I_syn = 0 if dw is None else self.synaptic_in(dw, t)

            # Ionic currents in soma and dendrite
            I_NaS = self.parameters['gNaS'] * (self.ms_inf(Vs) ** 2) * (1 - ns) * (Vs - self.parameters['VNa'])
            I_DrS = self.parameters['gDrS'] * (ns ** 2) * (Vs - self.parameters['VK'])
            I_leakS = self.parameters['gleak'] * (Vs - self.parameters['Vleak'])
            I_conS = (self.parameters['gc'] / self.parameters['kappa']) * (Vs - Vd)
            I_NaD = self.parameters['gNaD'] * (self.md_inf(Vd) ** 2) * hd * (Vd - self.parameters['VNa'])
            I_DrD = self.parameters['gDrD'] * (nd ** 2) * pd * (Vd - self.parameters['VK'])
            I_leakD = self.parameters['gleak'] * (Vd - self.parameters['Vleak'])
            I_conD = (self.parameters['gc'] / (1 - self.parameters['kappa'])) * (Vd - Vs)
            I_SK = self.parameters['gSK'] * sd * (Vd - self.parameters['VK'])
            I_NMDA = self.parameters['gNMDA'] * self.B(Vd) * Os * (Vd - self.parameters['VCa'])
            J_IP3 = self.parameters['nu_IP3'] * (self.minf_IP3 ** 3) * (self.ninf_IP3(Ca) ** 3) * (hca ** 3) * (CaER - Ca)
            J_PMCA = self.parameters['nu_PMCA'] * (Ca ** 2) / (Ca ** 2 + self.parameters['k_PMCA'] ** 2)
            J_Serca = self.parameters['nu_Serca'] * (Ca ** 2) / (Ca ** 2 + self.parameters['k_Serca'] ** 2)
            J_leak = self.parameters['nu_ERleak'] * (CaER - Ca)
            
            # Differential equations for voltage and gating variables
            derivatives = [
                (self.parameters['Iapp'] - I_NaS - I_DrS - I_leakS - I_conS) / self.parameters['C_m'],
                (-I_NaD - I_DrD - I_leakD - I_conD - I_NMDA - I_SK) / self.parameters['C_m'],
                (self.ns_inf(Vs) - ns) / self.parameters['tau_ns'],
                (self.hd_inf(Vd) - hd) / self.parameters['tau_hd'],
                (self.nd_inf(Vd) - nd) / self.parameters['tau_nd'],
                (self.pd_inf(Vd) - pd) / self.parameters['tau_pd'],
                (self.sd_inf(Ca) - sd) / self.parameters['tau_sd'],
                self.parameters['f_c'] * (-self.parameters['alpha'] * I_NMDA + J_IP3 - J_Serca - J_PMCA + J_leak),
                self.parameters['f_ER'] * self.parameters['gamma'] * (-J_IP3 + J_Serca - J_leak),
                (self.hinf_IP3(Ca) - hca) * self.parameters['a2'] * (self.Q2 + Ca),
                -(self.parameters['R_b'] * glu * C0) + (self.parameters['R_u'] * C1),
                -((self.parameters['R_b'] * glu + self.parameters['R_u']) * C1) + (self.parameters['R_b'] * glu * C0 + self.parameters['R_u'] * C2),
                -((self.parameters['R_o'] + self.parameters['R_d'] + self.parameters['R_u']) * C2) + (self.parameters['R_b'] * glu * C1 + self.parameters['R_c'] * Os + self.parameters['R_r'] * Ds),
                -(self.parameters['R_r'] * Ds) + (self.parameters['R_d'] * C2),
                -(self.parameters['R_c'] * Os) + (self.parameters['R_o'] * C2)
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
        columns:
          column 0 => white noise #1 for Vs
          column 1 => white noise #2 for Vd
          column 2 => power-law noise for Vd
        """
        logger.debug("Computing diffusion matrix at time t=%.6f with state y=%s", t, y)
        try:
            n_var = self.get_num_state_vars()
            # Retrieve noise parameters
            # Retrieve noise amplitudes from self.parameters
            sigma_vs = self.parameters.get('sigma_Vs', 0.05)  # e.g. 0.05
            sigma_vd = self.parameters.get('sigma_Vd', 0.05)  # e.g. 0.05 or something
            sigma_d  = self.parameters.get('sigma_d',  0.0)  # power-law amplitude

            # Initialize G_matrix with shape (15, 3)
            G_matrix = np.zeros((n_var, 3))

            # Noise dimension #0 => affects Vs
            #   Vs is index 0 in your state vector
            G_matrix[0, 0] = sigma_vs

            # Noise dimension #1 => affects Vd
            #   Vd is index 1 in your state vector
            G_matrix[1, 1] = sigma_vd

            # Noise dimension #2 => also affects Vd, but with amplitude sigma_d
            G_matrix[1, 2] = sigma_d
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
        return 15  # Vs, Vd, ns, hd, nd, pd