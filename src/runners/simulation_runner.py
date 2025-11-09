# neuron_simulation/runners/simulation_runner.py

import concurrent.futures
from typing import List, Dict, Any, Tuple
import joblib
import yaml
from pathlib import Path
import logging

import models
import solvers
from handlers.data_handler import DataHandler
from collections import namedtuple


model_classes = models.__all__
solver_classes = solvers.__all__

logger = logging.getLogger(__name__)
Solution = namedtuple('Solution', ['t', 'y'])

class SimulationRunner:
    def __init__(
        self,
        model_name: str,
        solver_name: str,
        base_params: Dict[str, float],
        base_config: Dict[str, Any],
        parameter_grid: List[Dict[str, float]] = None,
        save: bool = False,
        save_dir: str = "results",
        max_workers: int = None,
        isi_plot_type: str = 'histogram'
    ):
        """
        Initialize the SimulationRunner with selected model, solver, base parameters, and parameter grid.

        Parameters:
        -----------
        model_name : str
            Name of the model class to use (must be defined in models.__all__).
        solver_name : str
            Name of the solver class to use (must be defined in solvers.__all__).
        base_params : dict
            Base parameters for the NeuronModel. Individual simulations can override these.
        base_config : dict
            Base simulation configurations (e.g., t_span, y0, t_eval). Individual simulations can override these.
        parameter_grid : list of dict, optional
            A list where each dict represents a set of parameters to override base_params for a simulation.
        save : bool, optional
            Whether to save simulation results. Defaults to False.
        save_dir : str, optional
            Directory where results will be saved if save is True. Defaults to "results".
        max_workers : int, optional
            Maximum number of worker processes. Defaults to number of processors on the machine.
        isi_plot_type : str, optional
            Type of ISI plot: 'histogram' or 'kde'. Defaults to 'histogram'.
        """
        logger.debug("Initializing SimulationRunner with model_name='%s' and solver_name='%s'.", model_name, solver_name)

        # Validate model_name
        if model_name not in model_classes:
            logger.critical("Model '%s' is not available. Available models: %s", model_name, model_classes)
            raise ValueError(f"Model '{model_name}' is not available. Available models: {model_classes}")

        # Validate solver_name
        if solver_name not in solver_classes:
            logger.critical("Solver '%s' is not available. Available solvers: %s", solver_name, solver_classes)
            raise ValueError(f"Solver '{solver_name}' is not available. Available solvers: {solver_classes}")

        # Convert save_dir to absolute path
        self.save_dir = Path(save_dir).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("SimulationRunner: Save directory set to '%s'.", self.save_dir)

        self.model_name = model_name
        self.solver_name = solver_name

        # Instantiate the solver
        try:
            logger.debug("Instantiating solver '%s'.", solver_name)
            solver_class = getattr(solvers, solver_name)
            self.solver = solver_class()
            logger.debug("Solver '%s' instantiated successfully.", solver_name)
        except Exception as e:
            logger.critical("Failed to instantiate solver '%s': %s", solver_name, e, exc_info=True)
            raise

        self.base_params = base_params
        self.base_config = base_config
        self.save = save
        self.max_workers = max_workers or (concurrent.futures.ProcessPoolExecutor()._max_workers)
        self.isi_plot_type = isi_plot_type.lower()
        
        self.parameter_grid = parameter_grid
        if self.parameter_grid is not None:
            logger.info("Initialized SimulationRunner with %d parameter sets using model '%s' and solver '%s'.",
                        len(self.parameter_grid), model_name, solver_name)

    def run_simulation(self, params: Dict[str, Any], config: Dict[str, Any], run_id: int = 1) -> Tuple[int, Dict[str, Any]]:
        """
        Run a single simulation with given parameters and configurations.

        Parameters:
        -----------
        params : dict
            Parameters for the NeuronModel.
        config : dict
            Simulation configurations.
        run_id : int, optional
            Unique identifier for the simulation run.

        Returns:
        --------
        tuple
            run_id and ISI statistics or error message.
        """
        logger.info("Starting simulation run %d with params: %s and config: %s", run_id, params, config)
        try:
            # Initialize model dynamically         
            model_class = getattr(models, self.model_name)
            logger.debug("Run_id=%d: Initializing model '%s' with params: %s", run_id, self.model_name, params)
            model = model_class(solver=self.solver, **params)  

            # Run solver
            logger.debug("Run_id=%d: Running solver with config: %s", run_id, config)
            solution = self.solver.solve(
                model=model,
                t_span=config['t_span'],
                y0=config['y0'],
                t_eval=config['t_eval']
            )

            # Handle data
            logger.debug("Run_id=%d: Handling data with DataHandler.", run_id)
            data_handler = DataHandler(
                data={'t': solution.t, 'y': solution.y},
                params=params,
                config=config,
                run_id=run_id,
                save_dir=str(self.save_dir),
                isi_plot_type=self.isi_plot_type,
                cutoff_fr=config.get('cutoff_fr', 0.25),
                save_result=self.save,
                threshold=config.get('threshold', 0.0),
                distance=config.get('distance', 20)
            )

            logger.info("Completed simulation run %d successfully.", run_id)
            return (run_id, solution, data_handler.isi_stats)
            # return (run_id, data_handler.isi_stats)
        except Exception as e:
            logger.error("Simulation run %d failed: %s", run_id, e, exc_info=True)
            return (run_id, {"error": str(e)})

    def run_all(self) -> Dict[int, Dict[str, Any]]:
        """
        Execute all simulations in parallel.

        Returns:
        --------
        dict
            A dictionary mapping run_id to ISI statistics or error messages.
        """
        results = {}
        total_runs = len(self.parameter_grid)
        logger.info("Starting all simulations with max_workers=%d.", self.max_workers)

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Assign a unique run_id to each simulation (starting from 1)
            futures = {}
            for run_id in range(1, total_runs + 1):
                try:
                    params = self.merge_params(run_id)
                    config = self.merge_config(run_id)
                    future = executor.submit(
                        self.run_simulation,
                        params,
                        config,
                        run_id
                    )
                    futures[future] = run_id
                    logger.debug("Submitted simulation run_id=%d to the executor.", run_id)
                except Exception as e:
                    logger.error("Failed to submit simulation run_id=%d: %s", run_id, e, exc_info=True)
                    results[run_id] = {"error": str(e)}

            for future in concurrent.futures.as_completed(futures):
                run_id = futures[future]
                try:
                    run_id, isi_stats = future.result()
                    results[run_id] = isi_stats
                    if 'error' in isi_stats:
                        logger.error("Run_id=%d: Simulation failed with error: %s", run_id, isi_stats['error'])
                    else:
                        logger.info("Run_id=%d: Simulation completed successfully.", run_id)
                except Exception as e:
                    logger.error("Run_id=%d: Simulation generated an exception: %s", run_id, e, exc_info=True)
                    results[run_id] = {"error": str(e)}

        logger.info("All simulations completed.")
        return results

    def merge_params(self, run_id: int) -> Dict[str, Any]:
        """
        Merge base parameters with the specific parameter set for a run.

        Parameters:
        -----------
        run_id : int
            Unique identifier for the simulation run.

        Returns:
        --------
        dict
            Merged parameters.
        """
        # Ensure run_id is within the parameter_grid range
        if run_id < 1 or run_id > len(self.parameter_grid):
            logger.error("Run_id=%d is out of bounds for parameter_grid of size %d.", run_id, len(self.parameter_grid))
            raise IndexError(f"Run_id={run_id} is out of bounds for parameter_grid of size {len(self.parameter_grid)}.")

        merged = self.base_params.copy()
        merged.update(self.parameter_grid[run_id - 1])  # Adjust for 0-based index
        logger.debug("Run_id=%d: Merged parameters: %s", run_id, merged)
        return merged

    def merge_config(self, run_id: int) -> Dict[str, Any]:
        """
        Merge base configurations with the specific configuration for a run, if any.

        Parameters:
        -----------
        run_id : int
            Unique identifier for the simulation run.

        Returns:
        --------
        dict
            Merged simulation configurations.
        """
        merged_config = self.base_config.copy()
        logger.debug("Run_id=%d: Merged configuration: %s", run_id, merged_config)
        return merged_config