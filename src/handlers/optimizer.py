#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 00:50:59 2025

@author: amin
"""

import os
import uuid
import json
import pickle
import logging
import yaml
import optuna
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union, Optional, List, Tuple, NamedTuple, Callable
from collections import namedtuple
import models
import solvers
from handlers.data_handler import DataHandlerOptim, DataHandler
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Retrieve available model and solver classes
model_classes = models.__all__
solver_classes = solvers.__all__

Solution = namedtuple('Solution', ['t', 'y'])
# Define default feature columns (can be overridden via kwargs or class attributes)
DEFAULT_SPIKE_FEATURE_COLS = ['threshold_v', 'peak_v', 'trough_v', 'upstroke_v', 'downstroke_v', 'adp_v']
DEFAULT_SPIKETIME_FEATURE_COLS = ['mean_rate', 'std_isi', 'mean_isi', 'median_isi', 'burst_fraction'] # Example
EPSILON = 1e-9 # For safe division in loss calculation

@dataclass
class ConfigLoader:
    """
    Configuration file (YAML, JSON, PKL) for handling simulation, feature extraction and optimization
    """
    cfg_path: Optional[str] = None # Path to the single config file
    _configs: Optional[Dict[str, Any]] = field(init=False, default=None) # Stores loaded file content
    _default_getters: Dict[str, Callable[[], Dict[str, Any]]] = field(init=False, repr=False)

    def __post_init__(self):
        """Loads the configuration file specified by cfg_path."""
        # Initialize the map to default getter methods
        self._default_getters = {
            "simulation_config": self._get_default_simulation_config,
            "data_feature_config": self._get_default_data_feature_config,
            "optimizer_config": self._get_default_optimizer_config,
            "model_feature_config": self._get_default_model_feature_config,
        }

        if self.cfg_path:
            self._load_file()
        else:
            logger.warning("Configuration file not provided. Use defaults.")
            self._configs = {}
            
    def _load_file(self):
        """Loads and parse the configuration file into self._configs."""
        if not self.cfg_path or not os.path.exists(self.cfg_path):
            logger.error(f"Configuration file not found: {self.cfg_path}")
            self._configs = {} 
            return

        _, ext = os.path.splitext(self.cfg_path)
        ext = ext.lower()
        load_cfg = None

        if ext in ['.yaml', '.yml']:
            with open(self.cfg_path, 'r') as f:
                load_cfg = yaml.safe_load(f)
        elif ext == '.json':
            with open(self.cfg_path, 'r') as f:
                load_cfg = json.load(f)
        elif ext in ['.pkl', '.pickle']:
            with open(self.cfg_path, 'rb') as f:
                load_cfg = pickle.load(f)
        else:
            logger.error(f"Unsupported config file: {ext}. Use .yaml, .json, or .pkl")
            self._configs = {}
            return

        if not isinstance(load_cfg, dict):
            logger.error(f"Config file '{self.cfg_path}' is not dict.")
            self._configs = {}
            return

        self._configs = load_cfg # Store loaded data
        logger.debug(f"Successful loading configuration: {self.cfg_path}")

    def get_section(self, section_key: str) -> Dict[str, Any]:
        """
        Retrieves the configuration for a specific section.

        Args:
            section_key: Key identifier of the configuration section.

        Returns:
            A dictionary containing the configuration for the section.

        Raises:
            ValueError: If the section_key is not recognized (i.e., no default getter).
        """
        if section_key not in self._default_getters:
            available_keys = list(self._default_getters.keys())
            raise ValueError(f"Unknown configuration section key: '{section_key}'. "
                             f"Available keys: {available_keys}")

        # Get the default dictionary
        default_config = self._default_getters[section_key]()
        new_configs = self._configs.get(section_key, {})
        default_config.update(new_configs)
        logger.debug(f"Update configuration for section: '{section_key}'")
        return default_config
    
    @staticmethod
    def _get_default_simulation_config() -> Dict[str, Any]:
        """Default configuration for model simulation."""
        return {'dt': 0.01, 'fs': 1000 / 0.01,
                't_span': [0, 7000],
                'y0': [0.0, 0.0, 0.0, 0.5],
                'noise_type': None,
                'seed': None}

    @staticmethod
    def _get_default_data_feature_config() -> Dict[str, Any]:
        """Default configuration for analyzing data features."""
        return {'cutoff': (19.80, 24.80), 'sta_win': 10.0, 'filter': 5.0, 'dv_cutoff': 20.0,
                'max_interval': 0.005, 'min_height': 2.0, 'min_peak': -30.0, 'thresh_frac': 0.05}

    @staticmethod
    def _get_default_model_feature_config() -> Dict[str, Any]:
        """Default configuration for analyzing model features."""
        return {'cutoff': (2.0, 7.0), 'sta_win': 10.0, 'filter': 5.0, 'dv_cutoff': 2.0,
                'max_interval': 0.005, 'min_height': 1.0, 'min_peak': -1.0, 'thresh_frac': 0.05,
                'fs': 1000 / 0.01 } 

    @staticmethod
    def _get_default_optimizer_config() -> Dict[str, Any]:
        """Default configuration for optimization process."""
        return {
            'n_trials': 1000, 'n_jobs': -1,
            'storage': "postgresql://admin_optneurjit:admin@localhost/optuna_db",
            'load_if_exists': True,
            'direction': "minimize", 'directions': None, 'seed': 0,
            'sampler': None, 'pruner': None,
            'w_spike_features': None,
            'w_spiketime_features': None,
            'max_loss': 1e5,
            'seed': 0,
            'early_stop_threshold': None,
            'spike_feature_cols': DEFAULT_SPIKE_FEATURE_COLS,
            'spiketime_feature_cols': DEFAULT_SPIKETIME_FEATURE_COLS
        }

class NeuronSimulationOptimizer:
    def __init__(self,
                 model_name: str,
                 solver_name: str,
                 data_file: str, # Path to pickled target features
                 params_file: str, # Path to JSON parameter file
                 fit_mode: str = 'all', # 'isi', 'spike', or 'all'
                 config_file: Optional[str] = None,
                 study_name: Optional[str] = None,
                 result_dir: Optional[str] = None,
                 **kwargs):
        
        # Simulation setup
        self.model_name = model_name
        self.solver_name = solver_name
        self.fit_mode = fit_mode.lower()
        if self.fit_mode not in ['isi', 'spike', 'all']:
            raise ValueError("fit_mode must be 'isi', 'spike', or 'all'")

        # Load data file
        self.data_file = Path(data_file)
        self._load_data(self.data_file)
        
        # Load parameter ranges
        self.params_file = Path(params_file)
        self.parameter_ranges = self._load_params(self.params_file)
        
        # Setup optuna study
        self.study_name = study_name if study_name is not None else f"{model_name}_{uuid.uuid4().hex[:8]}"
        self.result_dir = Path(result_dir) if result_dir is not None else None
        if result_dir is None:
            self.save_result = False
        else:
            self.save_result = True
            os.makedirs(self.result_dir, exist_ok=True)
            logger.debug("Results will be saved to: %s", self.result_dir)

        # Load configurations
        config_loader = ConfigLoader(cfg_path=config_file)
        self.optimizer_config = config_loader.get_section('optimizer_config')
        self.simulation_config = config_loader.get_section('simulation_config')
        self.model_feature_config = config_loader.get_section('model_feature_config')
        self.data_feature_config = config_loader.get_section('data_feature_config')
        logging.debug("Configurations retrieved from ConfigLoader.")


        
        # Feature Column
        self.spike_feature_cols = self.optimizer_config.get('spike_feature_cols', None)
        self.spiketime_feature_cols = self.optimizer_config.get('spiketime_feature_cols', None)
        self.w_spike_features = self.optimizer_config.get('w_spike_features', None)
        self.w_spiketime_features = self.optimizer_config.get('w_spiketime_features', None)
        logger.debug("Using spike feature columns: %s", self.spike_feature_cols)
        logger.debug("Using spiketime feature columns: %s", self.spiketime_feature_cols)
        
        self.seed = self.optimizer_config.get('seed', 0)
        self.max_loss = self.optimizer_config.get('max_loss', 1e6)
        self.early_stop_threshold = self.optimizer_config.get('early_stop_threshold', None)
        self.multi_objective = False if self.optimizer_config["directions"] is None else True
            

    def _load_params(self, file_path: Path) -> Dict[str, Tuple[float, float]]:
        """Read parameter ranges from a JSON file."""
        if not file_path.is_file():
            raise FileNotFoundError(f"Parameter range file not found: {file_path}")
        try:
            with open(file_path, 'r') as f:
                ranges = json.load(f)
            # Validation (check if values are lists/tuples of size 2)
            for key, value in ranges.items():
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    raise ValueError(f"Invalid range format for parameter '{key}' in {file_path}. Expected [min, max].")
            return ranges
        except json.JSONDecodeError as e:
            logger.critical(f"Failed to decode JSON from {file_path}: {e}")
            raise
        except Exception as e:
            logger.critical(f"Error loading parameter ranges from {file_path}: {e}")
            raise

    def _load_data(self, file_path: Path):
        """Parse data and features from a pickle file."""
        if not file_path.is_file():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            if not isinstance(data, dict):
                 raise TypeError(f"Expected pickled data to be a dictionary, got {type(data)}")

            # Store target features based on what's needed
            if self.fit_mode in ['spike', 'all']:
                self.spike_feature_data = data.get('spike_features')
                if not isinstance(self.spike_feature_data, pd.DataFrame):
                     logger.warning(f"'spike_features' in {file_path} is not a DataFrame. Trying to convert.")
                     try:
                         self.spike_feature_data = pd.DataFrame(self.spike_feature_data)
                     except Exception as conv_err:
                         logger.error(f"Could not convert target 'spike_features' to DataFrame: {conv_err}")
                         self.spike_feature_data = None # Reset if conversion fails

            if self.fit_mode in ['isi', 'all']:
                self.spiketime_feature_data = data.get('spiketrain_features') 
                if not isinstance(self.spiketime_feature_data, pd.DataFrame):
                     logger.warning(f"'spiketrain_features' in {file_path} is not a DataFrame. Trying to convert.")
                     try:
                         self.spiketime_feature_data = pd.DataFrame(self.spiketime_feature_data)
                     except Exception as conv_err:
                         logger.error(f"Could not convert target 'spiketrain_features' to DataFrame: {conv_err}")
                         self.spiketime_feature_data = None # Reset if conversion fails

            # Validation: Check if we have the necessary target data for the requested fitting
            if self.fit_mode in ['spike', 'all'] and self.spike_feature_data is None:
                 raise ValueError("Fitting requires spike features, but spike features could not be loaded from data.")
            if self.fit_mode in ['isi', 'all'] and self.spiketime_feature_data is None:
                 raise ValueError("Fitting requires spiketime features, but spiketrain features could not be loaded from data.")
        except Exception as e:
            logger.critical(f"Error loading data from {file_path}: {e}")
            raise

    def run_simulation(self, params: Dict[str, Any], run_id: int = 1) -> Tuple[int, Union[Solution, Dict[str, Any]]]:
        """Runs a single simulation with the given parameters and configurations."""
        
        config = self.simulation_config
        seed = config.get('seed', run_id)
        
        # Instantiate the model and the solver
        try:
            solver_class = getattr(solvers, self.solver_name)
            solver = solver_class(step_size=config['dt'], seed = seed, noise_type = config['noise_type'])
            model_class = getattr(models, self.model_name)
            model = model_class(solver=solver, **params)
        except Exception as e:
            logger.critical(f"Failed to instantiate model or solver: {e}", exc_info=True)
            return (run_id, {"error": f"Instantiation failed: {e}"})

        logger.debug(f"Starting simulation run {run_id} for model '{self.model_name}' with {self.solver_name}.")
        try:
            # Ensure t_eval uses simulation config
            t_start, t_end = config['t_span']
            dt = config['dt']
            t_eval = np.arange(t_start, t_end + dt, dt)

            solution = solver.solve(
                model=model,
                t_span=config['t_span'],
                y0=config.get('y0'),
                t_eval=t_eval,
            )
            logger.debug(f"Run_id={run_id}: Simulation successful.")
            return (run_id, solution)
        except Exception as e:
            logger.error(f"Simulation run {run_id} failed: {e}", exc_info=True)
            return (run_id, {"error": str(e)})
    
    def feature_extractor(self,
                          data: Union[Solution, Dict[str, np.ndarray]],
                          run_id: int) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Extracts features from simulation results."""
        # Use the specific config for model feature extraction
        config = self.model_feature_config

        if isinstance(data, tuple) and hasattr(data, '_fields') and 't' in data._fields and 'y' in data._fields: 
            sim_data_dict = {'t': data.t / 1000, 'y': data.y[0]} # Assuming time in ms
        elif isinstance(data, dict) and 't' in data and 'y' in data:
             sim_data_dict = data
        else:
            logger.error(f"Run {run_id}: Invalid simulation output format received by feature_extractor: {type(data)}")
            return (None, None)

        logger.debug(f"Run {run_id}: Extracting features ('{self.fit_mode}') from simulation output.")
        try:
            # Assuming DataHandlerOptim takes simulation data and extracts based on 'feature' argument
            data_handler = DataHandlerOptim(data=sim_data_dict,
                                           data_type='simulation', 
                                           config=config,
                                           feature=self.fit_mode) 

            spike_features_model = getattr(data_handler, 'spike_features', None)
            spiketime_features_model = getattr(data_handler, 'spike_train_features', None)

            return (spike_features_model, spiketime_features_model)

        except Exception as e:
            logger.error(f"Run {run_id}: Feature extraction failed: {e}", exc_info=True)
            return (None, None)

    def get_normalized_loss(self,
                            data_features: pd.DataFrame,
                            model_features: pd.DataFrame,
                            feature_cols: List[str],
                            feature_type: str) -> Tuple[float, int]:
        """
        Computes normalized weighted loss for a given set of features (spike or spiketime).
        Returns the loss and the number of features used in the calculation.
        """
        if data_features is None or model_features is None or data_features.empty or model_features.empty:
            logger.debug(f"Cannot compute {feature_type} loss: Missing or empty dataframes.")
            return self.max_loss, 0
        
        config_weights = self.w_spike_features if feature_type == "spike" else self.w_spiketime_features
        # Drop columns with any NaN
        data_features = data_features.dropna(axis=1, how='any')
        model_features = model_features.dropna(axis=1, how='any')

        # If specific columns are specified
        if feature_cols is not None:
            # Keep intersection only
            common_features = data_features.columns.intersection(feature_cols).intersection(model_features.columns)
        else:
            common_features = data_features.columns.intersection(model_features.columns)
        
        if len(common_features) == 0:
            logger.warning("Data features and model features have no feature in common. Return high error")
            return self.max_loss, 0

        # Select columns
        data_features_means = data_features[common_features].mean()
        model_features_means = model_features[common_features].mean()

        total_loss = 0.0
        feature_count = 0
        use_config_weights = False
        
        if isinstance(config_weights, (list, tuple, np.ndarray)):
            if len(config_weights) == len(common_features):
                feature_weights = config_weights
                use_config_weights = True
        elif isinstance(config_weights, (int, float)):
                use_config_weights = True
                feature_weights = [config_weights] * len(common_features)

        logger.debug(f"Calculating {feature_type} loss using features: {common_features}")
        for i, col in enumerate(common_features):
            data_mean = data_features_means[col]
            model_mean = model_features_means[col]
            
            squared_diff = (model_mean - data_mean)**2
            w_i = feature_weights[i] if use_config_weights else 1.0 / max(abs(data_mean), EPSILON)

            total_loss += w_i * squared_diff
            feature_count += 1
            
        if feature_count == 0:
            logger.warning(f"No valid finite features were available to compute {feature_type} loss.")
            return self.max_loss, 0 # Return max loss if no features contributed

        # Average normalized squared error over the features
        mean_loss = total_loss / feature_count
        return mean_loss, feature_count
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        run_id = trial.number

        # Suggest parameters using pre-loaded ranges
        params = {
            key: trial.suggest_float(key, *value)
            for key, value in self.parameter_ranges.items()
        }

        # Run Simulation
        logger.debug(f"Trial {run_id}: Running simulation with {len(params)} suggested parameters.")
        sim_run_id, solution = self.run_simulation(params=params, run_id=run_id)

        if isinstance(solution, dict) and "error" in solution:
            logger.warning(f"Trial {run_id}: Simulation failed: {solution['error']}")
            return self.max_loss # Penalize failed simulations heavily

        # Extract Modeled Features
        try:
            spike_feature_model, spiketime_feature_model = self.feature_extractor(
                data=solution, run_id=run_id
            )
        except Exception as e:
            return self.max_loss

        # Calculate Losses
        loss_spike = 0 
        loss_spiketime = 0 
        features_used_spike = 0
        features_used_spiketime = 0

        # Compute spike features loss
        if self.fit_mode in ['spike', 'all']:
            spike_valid = all([self.spike_feature_data is not None and not self.spike_feature_data.empty,
                                spike_features_model is not None and not spike_features_model.empty
                                ])
            if spike_valid:
                loss_spike, features_used_spike = self.get_normalized_loss(
                    self.spike_feature_data,
                    spike_feature_model,
                    self.spike_feature_cols,
                    "spike"
                )
                logger.debug("Trial %d: mean spike feature loss: %f", run_id, loss_spike)
            else:
                logger.debug("Trial %d: Computing mean spike feature loss failed.", run_id)
                loss_spike = self.max_loss # Assign max loss if model fails to produce spikes

        # Compute spiketime features loss
        if self.fit_mode in ['isi', 'all']:
            spiketime_valid = all([self.spiketime_feature_data is not None and not self.spiketime_feature_data.empty,
                                spiketime_feature_model is not None and not spiketime_feature_model.empty
                                ])
            if spiketime_valid:
                loss_spiketime, features_used_spiketime = self.get_normalized_loss(
                    self.spiketime_feature_data,
                    spiketime_feature_model,
                    self.spiketime_feature_cols,
                    "spiketime"
                )
                logger.debug("Trial %d: mean spiketime feature loss: %f", run_id, loss_spiketime)
            else:
                logger.debug("Trial %d: Computing mean spiketime feature loss failed.", run_id)
                loss_spiketime = self.max_loss # Assign max loss

        # Compute total loss
        if self.multi_objective and self.fit_mode == "all":
            # For multi-objective, return losses separately
            final_loss = (loss_spike, loss_spiketime)
        else:
            # For single-objective, combine the losses
            if loss_spike != 0 and loss_spiketime != 0:
                final_loss = (loss_spike + loss_spiketime) / 2
            else:
                final_loss = loss_spike + loss_spiketime
                
        logger.info(f"Trial {run_id}: Loss_Spike={loss_spike:.4f}, Loss_SpikeTime={loss_spiketime:.4f} -> Total Loss={final_loss:.4f}")
        return final_loss
                
    def early_stop_callback(self, study: optuna.Study, trial: optuna.Trial) -> bool:
        """
        Callback to stop the study early if the loss reaches the early_stop_threshold.
        Compatible with single and multi-objective optimizations.
        """
        if self.early_stop_threshold is None:
            return False  # No early stopping

        if self.multi_objective:
            # Check if all objectives are below their respective thresholds
            # Assuming early_stop_threshold is a dict mapping objective index to threshold
            if not isinstance(self.early_stop_threshold, dict):
                logger.error("For multi-objective optimization, early_stop_threshold must be a dict mapping objective indices to thresholds.")
                return False
            met = True
            for i, threshold in self.early_stop_threshold.items():
                if i < len(trial.values):
                    if trial.values[i] > threshold:
                        met = False
                        break
                else:
                    met = False
                    break
            if met:
                logger.info("Early stopping: Trial %d has met the loss thresholds for all objectives.", trial.number)
                study.stop()
                return True
        else:
            # Single-objective optimization
            if len(trial.values) == 0:
                return False
            current_loss = trial.values[0]
            if current_loss <= self.early_stop_threshold:
                logger.info("Early stopping: Trial %d has met the loss threshold.", trial.number)
                study.stop()
                return True
        return False
    
    def configure_sampler(self,
                          sampler_override: Optional[optuna.samplers.BaseSampler] = None,
                          pruner_override: Optional[optuna.pruners.BasePruner] = None) -> Tuple[optuna.samplers.BaseSampler, optuna.pruners.BasePruner]:
        """Configures Optuna sampler and pruner."""
        # Sampler
        if sampler_override:
            sampler = sampler_override
            logger.debug("Using user-provided sampler.")
        elif self.multi_objective:
            # Use NSGAIISampler for multi-objective
            sampler = optuna.samplers.NSGAIISampler(seed=self.seed)
            logger.debug("Using NSGAII Sampler (multi-objective).")
        else:
            # Default to TPESampler for single-objective
            sampler = optuna.samplers.TPESampler(seed=self.seed, n_startup_trials=20)
            logger.debug("Using TPE Sampler (single-objective).")

        # Pruner
        if pruner_override:
            pruner = pruner_override
            logger.debug("Using user-provided pruner.")
        else:
            # Use MedianPruner, adjust parameters based on objective type potentially
            n_startup = 15 if self.multi_objective else 10
            n_warmup = 5
            pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup, n_warmup_steps=n_warmup, interval_steps=1)
            logger.debug(f"Using Median Pruner (startup={n_startup}, warmup={n_warmup}).")

        return sampler, pruner

    def run_optuna(self,
                   n_trials: int = 100,
                   n_jobs: int = 1,
                   storage: Optional[str] = None,
                   sampler: Optional[optuna.samplers.BaseSampler] = None,
                   pruner: Optional[optuna.pruners.BasePruner] = None,
                   direction: Optional[str] = None,           # For single-objective
                   directions: Optional[List[str]] = None,    # For multi-objective
                   load_if_exists: bool = False,
                   **kwargs):
        """Sets up and runs the Optuna optimization study."""
        config = self.optimizer_config           
        self.n_trials = config["n_trials"]
        self.n_jobs = config["n_jobs"]
        
        # Configure sampler and pruner
        sampler, pruner = self.configure_sampler(config["sampler"], config["pruner"])

        # Create or load the study
        try:
            study = optuna.create_study(
                study_name=self.study_name,
                storage=config["storage"],
                sampler=sampler,
                pruner=pruner,
                direction=config["direction"] if not self.multi_objective else None,
                directions=config["directions"] if self.multi_objective else None,
                load_if_exists=config["load_if_exists"]
            )
            logger.debug("Optuna study '%s' created or loaded successfully.", self.study_name)
        except Exception as e:
            logger.critical("Failed to create or load Optuna study: %s", e, exc_info=True)
            raise
            
        # Define callback
        def callback(study, trial):
            return self.early_stop_callback(study, trial)
    
        # Optimize
        logger.info(f"Starting optimization: {self.n_trials} trials, {self.n_jobs} jobs.")
        study.optimize(
            self.objective,
            n_trials=config["n_trials"],
            n_jobs=config["n_jobs"],
            callbacks=[callback]
            )
        logger.info("Optimization completed successfully.")

        # --- Save results (Best params, study, etc.) ---
        if self.save_result:
            self.log_best_trials(study)
        return study

    def log_best_trials(self, study: optuna.Study, return_solution=True, ode_run=True):
        """
        Logs and saves the best trials from the optimization study.
        Additionally, runs a simulation with the best parameters and saves the resulting plot.
        """
        logger.info("Optimization completed. Best trials (Pareto front):")

        if not study.best_trials:
            logger.warning("No best trials found.")
            return

        # Determine the number of objectives
        n_obj = len(study.best_trials[0].values)

        # Prepare data for CSV export
        data = {'Trial Number': [trial.number for trial in study.best_trials]}

        # Handle single or multi-objective cases
        if n_obj == 1:
            # Single-objective study
            data['Objective Value'] = [trial.values[0] for trial in study.best_trials]
            target_names = ['Objective Value']
        else:
            for i in range(n_obj):
                data[f'Objective_{i}'] = [trial.values[i] for trial in study.best_trials]
            target_names = [f'Objective_{i}' for i in range(n_obj)]

        # Add parameters to the data
        param_names = []
        if study.best_trials:
            param_names = list(study.best_trials[0].params.keys())
            for param in param_names:
                data[param] = [trial.params[param] for trial in study.best_trials]

        # Log the best trials
        for trial in study.best_trials:
            logger.info("Trial Number: %d | Values: %s | Params: %s", trial.number, trial.values, trial.params)

        # Convert to DataFrame and save to CSV
        df_best_trials = pd.DataFrame(data)
        logger.info("Best Trials DataFrame:")
        logger.info("\n%s", df_best_trials.to_string(index=False))

        csv_path = self.result_dir / "best_trials_pareto_front.csv"

        df_best_trials.to_csv(csv_path, index=False)
        logger.debug("Best trials saved to %s", csv_path)

        # Run simulation with best parameters and save the plot for each best trial
        for trial in study.best_trials:
            logger.info("Running simulation with best trial %d parameters...", trial.number)
            params = trial.params
            if ode_run:
                run_id = 00
                for key in params.keys():
                    if key.startswith("sigma"):
                        params[key] = 0.00
            
            # Run the simulation again using the best parameters
            run_id, solution = self.run_simulation(params=params, run_id=trial.number)

            if isinstance(solution, dict) and "error" in solution:
                logger.warning("Failed to run simulation for best trial %d: %s", trial.number, solution["error"])
            else:
                 data_handler = DataHandler(data={'t': solution.t / 1000, 'y': solution.y},
                                            params=params,
                                            config={**self.simulation_config, **self.model_feature_config},
                                            run_id=run_id,
                                            save_dir=str(self.result_dir),
                                            isi_plot_type='histogram',
                                            cutoff=(2.0, 7.0),
                                            save_result=self.save_result,
                                            threshold=0.0,
                                            distance=20.0
                                            )
        if return_solution:
            return solution, data_handler, params
