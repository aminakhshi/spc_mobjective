# neuron_simulation/utils/logger.py

import os
import logging
from typing import Union
import uuid
from datetime import datetime

def log_level_convert(level: int) -> str:
    """
    Converts an integer log level to its corresponding string representation.
    
    Parameters
    ----------
    level : int
        Integer representing the log level.
    
    Returns
    -------
    str
        String representation of the log level.
    """
    level_mapping = {
        0: 'NOTSET',
        1: 'DEBUG',
        2: 'INFO',
        3: 'WARNING',
        4: 'ERROR',
        5: 'CRITICAL'
    }
    return level_mapping.get(level, 'INFO')

def set_logger(
    log_dir: str = 'logs',
    level: Union[int, str] = 'INFO',
    run_id: Union[int, str] = None,
    log_file: str = None,
    logger_name: str = 'NeurJIT',
    save_log: bool = True,
) -> logging.Logger:
    """
    Sets up a logger to log messages to both the console and a file with defined verbosity.
    Logs are stored in a directory using the run ID and current datetime.
    
    Parameters
    ----------
    log_dir : str, optional
        Base directory where all logs are stored. Defaults to 'logs'.
    level : int or str, optional
        Logging level as an integer or string. Integer levels are converted to equivalent string values.
    run_id : int or str, optional
        Unique identifier for the current run. If not provided, a new run ID will be generated.
    log_file : str, optional
        Name of the log file. If not provided, defaults to 'simulation_{run_id}_{current_time}.log'.
    logger_name : str, optional
        Name of the logger. Defaults to the package name 'NeurJIT'.
    save_log : bool, optional
        Whether to save logs to a file. If False, logs are only output to the console.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    # Generate a unique run_id if not provided
    if run_id is None:
        # run_id = uuid.uuid4().hex[:8]  # 8-character hexadecimal run ID
        run_id = "no_id"
    elif isinstance(run_id, int):
        run_id = str(run_id)

    # Get current datetime in a readable format
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the base log directory if it doesn't exist
    if save_log:
        os.makedirs(log_dir, exist_ok=True)

    # Set default log_file name if not provided and if saving logs
    if save_log and log_file is None:
        log_file = f'simulation_{run_id}_{current_time}.log'

    # Full path for the log file
    log_path = os.path.join(log_dir, log_file)

    # Create or get the package-level logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  

    # Prevent adding multiple handlers if the logger is already configured
    if not logger.handlers:
        # Convert integer log level to string if necessary
        if isinstance(level, int):
            level = log_level_convert(level)
        elif isinstance(level, str):
            level = level.upper()
        else:
            level = 'INFO'

        # Get the logging level attribute, default to INFO if invalid
        log_level = getattr(logging, level, logging.INFO)

        # Define a consistent log message format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler for logging to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler for logging to a file if saving logs
        if save_log and log_path:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(log_path, maxBytes=10**6, backupCount=5)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Store run_id and log_dir as attributes
        logger.run_id = run_id
        logger.log_dir = log_dir if save_log else None

    return logger
