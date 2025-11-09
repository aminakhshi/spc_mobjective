"""
Created on Thu Dec  5 10:03:21 2024

@author: amin
"""

import os
import argparse
import logging
import uuid
from datetime import datetime
from pathlib import Path

import models
import solvers
# from handlers.data_handler import DataHandlerOptim, DataHandler
from utils.logger import set_logger
from handlers.optimizer import NeuronSimulationOptimizer

# Retrieve available model and solver classes
model_classes = models.__all__
solver_classes = solvers.__all__

def main():
    parser = argparse.ArgumentParser(description="Neuron Simulation CLI")

    # Model and Solver
    parser.add_argument('--model_name', type=str, default='Base_HR',
                        choices=model_classes, help='Neuron model class')
    parser.add_argument('--solver_name', type=str, default='SDESolver',
                        choices=solver_classes, help='Solver name.')

    # Required arguments
    parser.add_argument('--data_file', required=True, help='Path to the data in PKL format.')
    parser.add_argument('--params_file', required=True, help='Path to the JSON parameter ranges file.')
    parser.add_argument('--config_file', default=None, help='Path to the config file (YAML/JSON/PKL).')

    # Optional arguments
    parser.add_argument('--study_name', type=str, default=None, help='Optuna study name.')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='Directory to save results/logs. If None, results are not saved.')
    parser.add_argument('--fit_mode', type=str, default='isi', choices=['spike', 'isi', 'all'],
                        help='Fitting mode for optuna.')

    # Logging arguments
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
                        help='Logging level.')
    parser.add_argument('--logger_name', type=str, default='NeurJIT',
                        help='Logger name (all modules can share this).')
    parser.add_argument('--save_log', action='store_true',
                        help='If set, logs are also written to a file. Otherwise console-only.')

    args = parser.parse_args()

    # Generate a study name if not user-defined
    if not args.study_name:
        args.study_name = uuid.uuid4().hex[:8]

    # Create the result_dir
    if args.result_dir is not None:
        result_dir = Path(args.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
    else:
        result_dir = None

    # 7) Create a single logger for the entire run
    logger = set_logger(
        log_dir=str(result_dir) if result_dir else 'logs',
        level=args.log_level,
        run_id=args.study_name,
        logger_name=args.logger_name,
        save_log=args.save_log
    )

    logger.info("========================================")
    logger.info("Starting Neuron Simulation Optimization!")
    logger.info(f"Study Name: {args.study_name}")
    logger.info(f"Data File:  {args.data_file}")
    logger.info(f"Model:     {args.model_name}")
    logger.info(f"Solver:     {args.solver_name}")
    logger.info(f"Result Dir: {result_dir}")
    logger.info("========================================")

    # Instantiate your optimizer
    optimizer = NeuronSimulationOptimizer(
        model_name=args.model_name,
        solver_name=args.solver_name,
        data_file=args.data_file,
        params_file=args.params_file,
        fit_mode=args.fit_mode,
        config_file=args.config_file,
        study_name=args.study_name,
        result_dir=result_dir
    )

    # Run the optimization
    study = optimizer.run_optuna()
    # study = optuna.load_study(study_name=optimizer.study_name,
    #                                  storage="postgresql://admin:admin@localhost/optuna_db")
    
    # Log the best trials
    optimizer.log_best_trials(study, return_solution=False, ode_run=False)

                   
if __name__ == "__main__":
    
    main()