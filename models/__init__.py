"""
List all available model classes.
Available models are:
- Hodgkin-Huxley
- Leaky Integrate-and-Fire
- Izhikevich
- Morris-Lecar
- FitzHugh-Nagumo
- Hindmarsh-Rose
## TODO: adding more models.  
copyright: 2024, @aminakhshi
"""

import importlib
import inspect
import pkgutil
from typing import Any
from models.base_model import NeuronModel

__all__ = []

for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    # Exclude base_model module containing the NeuronModel
    if module_name in ['base_model'] or is_pkg:
        continue

    try:
        module = importlib.import_module(f'.{module_name}', package=__name__)
    except ImportError as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not import module {module_name}: {e}")
        continue

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, NeuronModel) and obj.__module__ == module.__name__:
            globals()[name] = obj
            __all__.append(name)

