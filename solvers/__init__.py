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

logger = logging.getLogger(__name__)

for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    if module_name == 'base_solver' or is_pkg:
        continue

    try:
        module = importlib.import_module(f'.{module_name}', package=__name__)
    except ImportError as e:
        logger.warning(f"Could not import module '{module_name}': {e}")
        continue

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, Solver) and obj.__module__ == module.__name__:
            globals()[name] = obj
            __all__.append(name)

