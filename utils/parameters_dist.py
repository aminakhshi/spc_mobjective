"""
This module provides the ParametersDist class to generate Chaospy distributions for model parameters,
facilitating sensitivity analysis by sampling from various probability distributions.

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
from __future__ import absolute_import, division, print_function, unicode_literals
import chaospy as cp
import numpy as np
import logging
from typing import Union, List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class ParametersDist:
    """
    A class to generate Chaospy distributions for model parameters, facilitating
    sensitivity analysis by sampling from various probability distributions.

    Attributes
    ----------
    distribution_type : str or List[str]
        The type of distribution to use (e.g., 'uniform', 'normal', 'beta', etc.).
        Can be a single string for all parameters or a list specifying per-parameter distributions.
    params : dict or List[dict]
        Additional parameters required to define the distribution.
        Can be a single dictionary for all parameters or a list specifying per-parameter parameters.

    Methods
    -------
    get_distribution(parameters, additional_args=None):
        Generates and returns the Chaospy distribution(s) based on the distribution type and parameter value(s).
    """

    def __init__(self, distribution_type: Union[str, List[str]] = 'uniform', **params):
        """
        Initializes the ParametersDist with a specified distribution type and parameters.

        Parameters
        ----------
        distribution_type : str or List[str], optional
            The type of distribution to use (e.g., 'uniform', 'normal', 'beta', etc.).
            Defaults to 'uniform'. If a list is provided, it should match the number of parameters.
        **params :
            Additional keyword arguments specific to the distribution type.
            For example, 'interval' for uniform distributions, 'alpha' and 'beta_param' for beta distributions, etc.
            If multiple parameters are used, pass a list of dictionaries under the key 'params_list'.
        """
        logger.info("Initializing ParametersDist class.")
        self.distribution_type = distribution_type
        self.params = params

        # Validate distribution_type and params
        if isinstance(self.distribution_type, list):
            logger.debug("Multiple distribution types provided.")
            if 'params_list' in self.params:
                if not isinstance(self.params['params_list'], list):
                    logger.error("'params_list' must be a list of dictionaries when multiple distribution types are provided.")
                    raise ValueError("'params_list' must be a list of dictionaries when multiple distribution types are provided.")
            else:
                self.params['params_list'] = [{} for _ in self.distribution_type]
                logger.debug("No 'params_list' provided. Using default empty dictionaries for each distribution type.")
        else:
            logger.debug("Single distribution type provided.")
            # Ensure params_list is a list with one dictionary
            self.params['params_list'] = [self.params]
            logger.debug("Using a single distribution type for all parameters.")

        logger.info("ParametersDist initialized with distribution_type: %s and params: %s", self.distribution_type, self.params)

    def get_distribution(
        self,
        parameters: Union[float, List[float]],
        additional_args: Union[None, List[Tuple[str, Dict[str, Any]]]] = None
    ) -> Union[cp.Distribution, cp.J]:
        """
        Generates the Chaospy distribution(s) based on the distribution type and parameter value(s).

        Parameters
        ----------
        parameters : float or List[float]
            The central value(s) around which the distribution(s) are defined.
        additional_args : list of tuples, optional
            When parameters is a list, additional_args can specify per-parameter distribution types and parameters.
            Each tuple should be (distribution_type, params_dict).
            If not provided, the default distribution_type and params are used.

        Returns
        -------
        chaospy.distributions.Distribution or chaospy.distributions.J
            The generated Chaospy distribution object or joint distribution.

        Raises
        ------
        ValueError
            If any parameter value is invalid for the specified distribution type or if inputs are inconsistent.
        """
        logger.info("Generating distribution(s) for parameters: %s", parameters)

        # Handle single parameter dist
        if isinstance(parameters, (int, float)):
            logger.info("Single parameter detected.")
            distribution = self._create_single_distribution(parameters, 0)
            logger.info("Generated single distribution: %s", distribution)
            return distribution

        # Handle list of parameters
        elif isinstance(parameters, list):
            logger.info("Multiple parameters detected.")

            distributions = []
            param_names = []
            num_params = len(parameters)

            # Determine distribution types and params per parameter
            if isinstance(self.distribution_type, list):
                if len(self.distribution_type) != num_params:
                    logger.error("Length of distribution_type list (%d) does not match number of parameters (%d).", len(self.distribution_type), num_params)
                    raise ValueError("Length of distribution_type list does not match number of parameters.")
                distribution_types = self.distribution_type
                params_list = self.params.get('params_list', [{} for _ in range(num_params)])
            else:
                distribution_types = [self.distribution_type] * num_params
                params_list = [self.params.get('params_list', [{}])[0]] * num_params

            # Override with additional_args if provided
            if additional_args:
                if len(additional_args) != num_params:
                    logger.error("Length of additional_args (%d) does not match number of parameters (%d).", len(additional_args), num_params)
                    raise ValueError("Length of additional_args does not match number of parameters.")
                for i, (dist_type, dist_params) in enumerate(additional_args):
                    distribution_types[i] = dist_type
                    params_list[i] = dist_params
                    logger.debug("Overriding parameter %d with distribution_type: %s and params: %s", i, dist_type, dist_params)

            for idx, (param, dist_type, dist_params) in enumerate(zip(parameters, distribution_types, params_list)):
                logger.debug("Creating distribution for parameter %d: value=%s, type=%s, params=%s", idx, param, dist_type, dist_params)
                distribution = self._create_single_distribution(param, idx, dist_type, dist_params)
                distributions.append(distribution)
                param_names.append(f"param_{idx}")

            # Create joint distribution
            joint_dist = cp.J(*distributions)
            logger.info("Generated joint distribution: %s", joint_dist)
            return joint_dist

        else:
            logger.error("Parameters must be a float or a list of floats. Received type: %s", type(parameters))
            raise TypeError("Parameters must be a float or a list of floats.")

    def _create_single_distribution(
        self,
        parameter: float,
        index: int = 0,
        dist_type: Union[str, None] = None,
        dist_params: Union[Dict[str, Any], None] = None
    ) -> cp.Distribution:
        """
        Helper method to create a single Chaospy distribution based on the distribution type and parameter.

        Parameters
        ----------
        parameter : float
            The central value around which the distribution is defined.
        index : int, optional
            The index of the parameter (used for logging). Defaults to 0.
        dist_type : str, optional
            The type of distribution to use. If None, uses the class's distribution_type.
        dist_params : dict, optional
            Additional parameters for the distribution. If None, uses the class's params.

        Returns
        -------
        chaospy.distributions.Distribution
            The generated Chaospy distribution object.

        Raises
        ------
        ValueError
            If the parameter value is invalid for the specified distribution type.
        """
        if dist_type is None:
            if isinstance(self.distribution_type, list):
                dist_type = self.distribution_type[index].lower()
            else:
                dist_type = self.distribution_type.lower()

        if dist_params is None:
            if isinstance(self.distribution_type, list):
                dist_params = self.params.get('params_list', [{}])[index]
            else:
                dist_params = self.params.get('params_list', [{}])[0]

        logger.debug("Parameter %d: Using distribution_type='%s' with params=%s", index, dist_type, dist_params)

        # Avoid generating distributions for invalid parameter values
        if parameter == 0 and dist_type in ['uniform', 'normal', 'beta', 'exponential', 'lognormal']:
            logger.error("Parameter %d: Cannot create a '%s' distribution around 0.", index, dist_type)
            raise ValueError(f"Cannot create a '{dist_type}' distribution around 0 for parameter index {index}.")

        if parameter < 0 and dist_type in ['uniform', 'normal', 'beta', 'exponential', 'lognormal']:
            logger.error("Parameter %d: Negative values are not allowed for '%s' distribution.", index, dist_type)
            raise ValueError(f"Negative values are not allowed for '{dist_type}' distribution for parameter index {index}.")

        # Generate the distribution based on type
        dist_type_lower = dist_type.lower()
        if dist_type_lower == 'uniform':
            distribution = self._uniform(parameter, dist_params, index)
        elif dist_type_lower == 'normal':
            distribution = self._normal(parameter, dist_params, index)
        elif dist_type_lower == 'beta':
            distribution = self._beta(parameter, dist_params, index)
        elif dist_type_lower == 'exponential':
            distribution = self._exponential(parameter, dist_params, index)
        elif dist_type_lower == 'lognormal':
            distribution = self._lognormal(parameter, dist_params, index)
        else:
            distribution = self._generic_distribution(parameter, dist_type, dist_params, index)

        logger.debug("Parameter %d: Distribution created: %s", index, distribution)
        return distribution

    def _uniform(self, parameter: float, params: Dict[str, Any], index: int) -> cp.Distribution:
        """
        Creates a uniform distribution around the parameter value.

        Parameters
        ----------
        parameter : float
            The central value for the uniform distribution.
        params : dict
            Additional parameters for the uniform distribution.
        index : int
            The index of the parameter.

        Returns
        -------
        chaospy.distributions.Uniform
            The uniform distribution.

        Raises
        ------
        ValueError
            If the parameter is zero.
        """
        interval = params.get('interval', 0.1)
        logger.debug("Parameter %d: Creating uniform distribution with interval=%s", index, interval)

        lower = parameter - abs(interval / 2.0 * parameter)
        upper = parameter + abs(interval / 2.0 * parameter)

        # Handle cases where parameter is very small to avoid lower == upper
        if lower == upper:
            upper += 1e-6
            logger.warning("Parameter %d: Adjusted upper bound to avoid zero width uniform distribution.", index)

        logger.info("Parameter %d: Uniform distribution bounds: lower=%.6f, upper=%.6f", index, lower, upper)
        return cp.Uniform(lower, upper)

    def _normal(self, parameter: float, params: Dict[str, Any], index: int) -> cp.Distribution:
        """
        Creates a normal (Gaussian) distribution centered at the parameter value.

        Parameters
        ----------
        parameter : float
            The mean of the normal distribution.
        params : dict
            Additional parameters for the normal distribution.
        index : int
            The index of the parameter.

        Returns
        -------
        chaospy.distributions.Normal
            The normal distribution.

        Raises
        ------
        ValueError
            If the parameter is zero.
        """
        interval = params.get('interval', 0.1)
        std_dev = abs(params.get('std_dev', interval * parameter))
        logger.debug("Parameter %d: Creating normal distribution with mean=%.6f and std_dev=%.6f", index, parameter, std_dev)

        logger.info("Parameter %d: Normal distribution with mean=%.6f and std_dev=%.6f", index, parameter, std_dev)
        return cp.Normal(parameter, std_dev)

    def _beta(self, parameter: float, params: Dict[str, Any], index: int) -> cp.Distribution:
        """
        Creates a beta distribution scaled by the parameter value.

        Parameters
        ----------
        parameter : float
            The scaling factor for the beta distribution. Must be between 0 and 1.
        params : dict
            Additional parameters for the beta distribution.
        index : int
            The index of the parameter.

        Returns
        -------
        chaospy.distributions.Beta
            The beta distribution scaled by the parameter.

        Raises
        ------
        ValueError
            If the parameter is not between 0 and 1.
        """
        alpha = params.get('alpha', 2)
        beta_param = params.get('beta_param', 5)
        logger.debug("Parameter %d: Creating beta distribution with alpha=%.6f, beta_param=%.6f, scaled by %.6f", index, alpha, beta_param, parameter)

        logger.info("Parameter %d: Beta distribution with alpha=%.6f, beta_param=%.6f, scaled by %.6f", index, alpha, beta_param, parameter)
        return cp.Beta(alpha, beta_param) * parameter

    def _exponential(self, parameter: float, params: Dict[str, Any], index: int) -> cp.Distribution:
        """
        Creates an exponential distribution scaled by the parameter value.

        Parameters
        ----------
        parameter : float
            The scaling factor for the exponential distribution. Must be positive.
        params : dict
            Additional parameters for the exponential distribution.
        index : int
            The index of the parameter.

        Returns
        -------
        chaospy.distributions.Exponential
            The exponential distribution scaled by the parameter.

        Raises
        ------
        ValueError
            If the parameter is not positive.
        """
        rate = params.get('rate', 1.0)
        logger.debug("Parameter %d: Creating exponential distribution with rate=%.6f, scaled by %.6f", index, rate, parameter)

        logger.info("Parameter %d: Exponential distribution with rate=%.6f, scaled by %.6f", index, rate, parameter)
        return cp.Exponential(rate) * parameter

    def _lognormal(self, parameter: float, params: Dict[str, Any], index: int) -> cp.Distribution:
        """
        Creates a log-normal distribution based on the parameter value.

        Parameters
        ----------
        parameter : float
            The mean of the underlying normal distribution (in log-space). Must be positive.
        params : dict
            Additional parameters for the log-normal distribution.
        index : int
            The index of the parameter.

        Returns
        -------
        chaospy.distributions.Lognormal
            The log-normal distribution.

        Raises
        ------
        ValueError
            If the parameter is not positive.
        """
        sigma = params.get('sigma', 0.5)
        logger.debug("Parameter %d: Creating log-normal distribution with mean=%.6f and sigma=%.6f", index, parameter, sigma)

        logger.info("Parameter %d: Log-normal distribution with mean=%.6f and sigma=%.6f", index, parameter, sigma)
        return cp.Lognormal(parameter, sigma)

    def _generic_distribution(
        self,
        parameter: float,
        dist_type: str,
        params: Dict[str, Any],
        index: int
    ) -> cp.Distribution:
        """
        Creates a generic Chaospy distribution using the distribution type and additional parameters.

        Parameters
        ----------
        parameter : float
            The parameter value to be used in the distribution.
        dist_type : str
            The type of distribution to create.
        params : dict
            Additional parameters for the distribution.
        index : int
            The index of the parameter.

        Returns
        -------
        chaospy.distributions.Distribution
            The generic Chaospy distribution.

        Raises
        ------
        ValueError
            If the distribution type is unsupported or if required parameters are missing.
        """
        logger.debug("Parameter %d: Creating generic distribution of type '%s' with params=%s", index, dist_type, params)

        try:
            # Attempt to retrieve the distribution class from chaospy
            dist_class = getattr(cp, dist_type.capitalize())
            logger.debug("Parameter %d: Found distribution class '%s'.", index, dist_class)
        except AttributeError:
            logger.error("Parameter %d: Unsupported distribution type: '%s'.", index, dist_type)
            raise ValueError(f"Unsupported distribution type: '{dist_type}' for parameter index {index}.")

        # Prepare arguments for the distribution
        try:
            distribution = dist_class(parameter, **params)
            logger.debug("Parameter %d: Created generic distribution: %s", index, distribution)
            logger.info("Parameter %d: Generic distribution '%s' created successfully.", index, dist_type)
            return distribution
        except TypeError as e:
            logger.error("Parameter %d: Error creating distribution '%s': %s", index, dist_type, e)
            raise ValueError(f"Error creating distribution '{dist_type}' for parameter index {index}: {e}")

    def define_distributions(
        self,
        default_params: Dict[str, float],
        param_variation: float = 0.1
    ) -> Tuple[cp.Distribution, List[str]]:
        """
        Define parameter distributions using uniform distributions around default values.

        Parameters
        ----------
        default_params : dict
            A dictionary of parameter names and their default values.
        param_variation : float, optional
            The relative variation around the default value to define the uniform distribution.
            Defaults to 0.1 (i.e., ±10%).

        Returns
        -------
        tuple
            A tuple containing the joint Chaospy distribution and a list of parameter names.

        Raises
        ------
        ValueError
            If any default parameter is invalid for the uniform distribution.
        """
        logger.info("Defining distributions based on default_params with param_variation=%.2f", param_variation)
        distributions = []
        param_names = []

        for key, value in default_params.items():
            logger.info("Defining distribution for parameter '%s' with default value %.6f", key, value)
            if value != 0.0:
                lower = value * (1 - param_variation)
                upper = value * (1 + param_variation)
                logger.debug("Parameter '%s': Calculated uniform bounds: lower=%.6f, upper=%.6f", key, lower, upper)
                # Ensure that lower <= upper
                if lower > upper:
                    lower, upper = upper, lower
                    logger.debug("Parameter '%s': Swapped bounds to maintain lower <= upper.", key)
            else:
                # For zero default value, define an absolute variation
                absolute_variation = 0.1  # Adjust as appropriate for your model
                lower = 0.0  # Assuming negative values are not acceptable
                upper = absolute_variation
                logger.debug("Parameter '%s': Zero default value, setting uniform bounds: lower=%.6f, upper=%.6f", key, lower, upper)

            if lower == upper:
                upper += 1e-6  # Ensure upper > lower
                logger.warning("Parameter '%s': Adjusted upper bound to avoid zero width distribution.", key)

            distributions.append(cp.Uniform(lower, upper))
            param_names.append(key)
            logger.info("Parameter '%s' distribution: Uniform(%.6f, %.6f)", key, lower, upper)

        joint_distribution = cp.J(*distributions)
        logger.info("Joint distribution created: %s", joint_distribution)
        return joint_distribution, param_names


def generate_joint_distribution(default_params: Dict[str, float],
                                constant_params: Dict[str, float],
                                distribution_type: Union[str, List[str]] = 'uniform',
                                param_variation: float = 0.1,
                                additional_params: Dict[str, Any] = None) -> Tuple[cp.Distribution, List[str]]:
    """
    Generate a joint Chaospy distribution for the variable parameters.

    Parameters:
    ----------
    default_params : dict
        Dictionary of default parameter values for variable parameters.
    constant_params : dict
        Dictionary of constant parameter values that should not be sampled.
    distribution_type : str or list of str
        Distribution type(s) for parameters. Can be a single string or a list matching the number of parameters.
    param_variation : float
        Relative variation for parameters (used within the ParametersDist class).
    additional_params : dict
        Additional parameters for defining distributions (e.g., specific distribution parameters).

    Returns:
    -------
    tuple
        - Joint Chaospy distribution for variable parameters.
        - List of variable parameter names.
    """
    logger.debug("Entering generate_joint_distribution with default_params: %s, constant_params: %s, distribution_type: %s, param_variation: %.2f, additional_params: %s",
                 default_params, constant_params, distribution_type, param_variation, additional_params)

    if additional_params is None:
        additional_params = {}

    # Filter out constant parameters to get only variable ones
    variable_params = {k: v for k, v in default_params.items() if k not in constant_params}
    logger.debug("Filtered variable_params: %s", variable_params)

    # Initialize ParametersDist with distribution types and additional distribution-specific parameters
    try:
        param_dist = ParametersDist(distribution_type=distribution_type, **additional_params)
        logger.debug("ParametersDist instance created successfully.")
    except ValueError as e:
        logger.error("Failed to initialize ParametersDist: %s", e)
        raise

    # Define distributions based on variable_params and param_variation
    try:
        joint_distribution, param_names = param_dist.define_distributions(variable_params, param_variation)
        logger.debug("Joint distribution and parameter names generated successfully.")
    except ValueError as e:
        logger.error("Failed to define distributions: %s", e)
        raise

    logger.debug("Exiting generate_joint_distribution with joint_distribution: %s, param_names: %s", joint_distribution, param_names)
    return joint_distribution, param_names

def sample_parameters(joint_distribution: cp.Distribution,
                      param_names: List[str],
                      num_samples: int = 100,
                      seed: int = None,
                      constant_params: Dict[str, float] = None) -> List[Dict[str, float]]:
    """
    Sample parameter sets from the joint distribution and add constant parameters.

    Parameters:
    ----------
    joint_distribution : cp.Distribution
        The joint distribution of variable parameters.
    param_names : list of str
        List of variable parameter names corresponding to the joint distribution.
    num_samples : int
        Number of parameter sets to sample.
    seed : int, optional
        Random seed for reproducibility.
    constant_params : dict, optional
        Dictionary of constant parameter values to include in each sample.

    Returns:
    -------
    list of dict
        List containing dictionaries of sampled parameter values combined with constant parameters.
    """
    logger.debug("Entering sample_parameters with joint_distribution: %s, param_names: %s, num_samples: %d, seed: %s, constant_params: %s",
                 joint_distribution, param_names, num_samples, seed, constant_params)

    if seed is not None:
        np.random.seed(seed)
        logger.debug("Random seed set to %d.", seed)

    try:
        # Generate samples based on the defined rule
        samples = joint_distribution.sample(num_samples, rule="sobol")
        logger.debug("Samples generated successfully.")
    except Exception as e:
        logger.error("Failed to generate samples from joint_distribution: %s", e)
        raise

    # Convert samples to list of dictionaries and add constant parameters
    parameter_grid = []
    for i in range(num_samples):
        try:
            # Create a parameter set from sampled values
            param_set = {name: samples[j, i] for j, name in enumerate(param_names)}
            logger.debug("Sample %d: Generated param_set: %s", i, param_set)

            # Add constant parameters to each sample
            if constant_params:
                param_set.update(constant_params)
                logger.debug("Sample %d: Updated param_set with constant_params: %s", i, constant_params)

            parameter_grid.append(param_set)
        except Exception as e:
            logger.error("Failed to create parameter set for sample %d: %s", i, e)
            raise

    logger.debug("Exiting sample_parameters with %d samples generated.", num_samples)
    return parameter_grid, samples.T
