"""
List all available solver classes.
Available solvers are:
- Euler
- RungeKutta
- etc.
# TODO: Add more solvers.

Copyright: 2024, @aminakhshi
"""

import importlib
import inspect
import pkgutil
import logging
from typing import Any
from solvers.base_solver import Solver  

__all__ = []

# Configure logger for the solvers package
logger = logging.getLogger(__name__)

for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    # Exclude 'base_solver' module and any sub-packages
    if module_name == 'base_solver' or is_pkg:
        continue

    try:
        # Dynamically import the module within the 'solvers' package
        module = importlib.import_module(f'.{module_name}', package=__name__)
    except ImportError as e:
        # Log the import error and skip the problematic module
        logger.warning(f"Could not import module '{module_name}': {e}")
        continue

    # Iterate through all classes defined in the module
    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Check if the class is a subclass of Solver and defined in this module
        if issubclass(obj, Solver) and obj.__module__ == module.__name__:
            # Add the class to the solvers namespace and __all__
            globals()[name] = obj
            __all__.append(name)
