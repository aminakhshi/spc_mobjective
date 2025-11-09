# neuron_simulation/handlers/sensitivity_analysis.py

import logging
from pathlib import Path
import yaml
import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class SensitivityAnalysis:
    def __init__(self, joint_distribution: cp.J, param_names: List[str], samples: np.ndarray, outputs: List[float], save_dir: str):
        """
        Initialize the SensitivityAnalysis with necessary data.

        Parameters:
        -----------
        joint_distribution : cp.J
            The joint distribution of parameters.
        param_names : List[str]
            Names of the parameters.
        samples : np.ndarray
            Sampled parameter sets.
        outputs : List[float]
            Simulation outputs corresponding to each sample.
        save_dir : str
            Directory to save sensitivity analysis results.
        """
        self.joint_distribution = joint_distribution
        self.param_names = param_names
        self.samples = samples
        self.outputs = outputs
        self.save_dir = Path(save_dir).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.poly_order = 3  # Can be parameterized if needed

    def perform_pce(self):
        """
        Perform Polynomial Chaos Expansion (PCE) fitting.

        Returns:
        --------
        cp.J
            The fitted PCE model.
        """
        logger.info("Starting Polynomial Chaos Expansion (PCE) fitting.")
        try:
            poly_expansion = cp.expansion.stieltjes(self.poly_order, self.joint_distribution)
            logger.debug("PCE expansion order set to %d.", self.poly_order)

            # Regression to fit PCE coefficients
            approx_model = cp.fit_regression(poly_expansion, self.samples.T, self.outputs)
            logger.info("PCE fitting completed successfully.")
            return approx_model
        except Exception as e:
            logger.exception("Error during PCE fitting: %s", e)
            raise

    def compute_sobol_indices(self, approx_model: cp.J):
        """
        Compute first-order Sobol sensitivity indices.

        Parameters:
        -----------
        approx_model : cp.J
            The fitted PCE model.

        Returns:
        --------
        np.ndarray
            First-order Sobol sensitivity indices.
        """
        logger.info("Computing first-order Sobol sensitivity indices.")
        try:
            sensitivity_indices = cp.Sens_m(approx_model, self.joint_distribution)
            logger.info("Sensitivity indices computed successfully.")
            return sensitivity_indices
        except Exception as e:
            logger.exception("Error computing sensitivity indices: %s", e)
            raise

    def plot_sensitivity_indices(self, sensitivity_indices: np.ndarray):
        """
        Plot and save the sensitivity indices.

        Parameters:
        -----------
        sensitivity_indices : np.ndarray
            First-order Sobol sensitivity indices.
        """
        logger.info("Plotting sensitivity indices.")
        try:
            plt.figure(figsize=(10, 6))
            plt.bar(self.param_names, sensitivity_indices, color='skyblue')
            plt.xlabel('Parameters')
            plt.ylabel('First-order Sobol Sensitivity Index')
            plt.title('Parameter Sensitivity Analysis')
            plt.xticks(rotation=45)
            plt.tight_layout()
            fig_path = self.save_dir / 'sensitivity_indices.png'
            plt.savefig(fig_path, dpi=300)
            plt.close()
            logger.info(f"Sensitivity indices plot saved to {fig_path}")
        except Exception as e:
            logger.exception("Error plotting sensitivity indices: %s", e)
            raise
