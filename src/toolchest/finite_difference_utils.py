import inspect
import numpy as np


def finite_difference(signal: np.ndarray, timesteps: np.ndarray, method: str = 'central', **kwargs) -> np.ndarray:
    """
    Compute the finite difference of a signal using specified methods (central, forward, polyfit).

    Parameters:
    - signal: The signal to differentiate.
    - timesteps: The time points corresponding to each signal value.
    - method: The differentiation method ('central', 'forward', or 'polyfit').

    Kwargs for specific methods:
    - edges: Edge handling method for 'central' and 'forward' methods ('extend' or 'zero').
    - order: Polynomial order for the 'polyfit' method.
    - window_size: Window size for 'polyfit' method.
    - derivative_order: Derivative order for 'polyfit' method.

    Returns:
    - np.ndarray: The differentiated signal.
    """

    # Dictionary mapping methods to their specific function implementations
    method_funcs = {
        'central': central_difference,
        'forward': forward_difference,
        'polyfit': polynomial_fit_derivative
    }

    # Validate method argument
    if method not in method_funcs:
        raise ValueError(f"Invalid method '{method}'. Choose from {list(method_funcs.keys())}.")

    # Get the specific function for the chosen method
    selected_func = method_funcs[method]

    # Retrieve valid parameters for the selected function using inspect.signature
    valid_params = set(inspect.signature(selected_func).parameters.keys())

    # Check if provided kwargs are valid
    provided_keys = set(kwargs.keys())
    if not provided_keys <= valid_params:
        invalid_keys = provided_keys - valid_params
        raise ValueError(
            f"Invalid arguments for method '{method}': {invalid_keys}. Valid arguments are {valid_params}.")

    # Call the specific function with validated kwargs
    return selected_func(signal, timesteps, **kwargs)


def central_difference(signal: np.ndarray, timesteps: np.ndarray, edges='extend') -> np.ndarray:
    """
    This function computes the central difference of a signal given the signal and the corresponding timesteps.
    """
    # Compute the central difference
    gradient = np.zeros_like(signal, dtype=np.float64)
    for i in range(1, len(signal) - 1):
        gradient[i] = (signal[i + 1] - signal[i - 1]) / (timesteps[i + 1] - timesteps[i - 1])
    if edges == 'extend':
        gradient[0] = gradient[1]
        gradient[-1] = gradient[-2]
    elif edges == 'zero':
        gradient[0] = 0
        gradient[-1] = 0
    else:
        raise ValueError(f"Invalid edges argument: {edges}")
    return gradient


def forward_difference(signal: np.ndarray, timesteps: np.ndarray, edges='extend') -> np.ndarray:
    """
    This function computes the forward difference of a signal given the signal and the corresponding timesteps.
    """
    # Compute the forward difference
    gradient = np.zeros_like(signal, dtype=np.float64)
    for i in range(len(signal) - 1):
        gradient[i] = (signal[i + 1] - signal[i]) / (timesteps[i + 1] - timesteps[i])
    if edges == 'extend':
        gradient[-1] = gradient[-2]
    elif edges == 'zero':
        gradient[-1] = 0
    else:
        raise ValueError(f"Invalid edges argument: {edges}")
    return gradient


def polynomial_fit_derivative(signal: np.ndarray, timesteps: np.ndarray, order: int = 3, window_size: int = 10,
                              derivative_order: int = 1) -> np.ndarray:
    """
    This function computes the sliding window polynomial fit derivative of a signal.
    """
    if len(signal) < window_size:
        raise ValueError("window_size (" + str(window_size) + ") must be less than the length of the signal (" + str(
            len(signal)) + ").")

    gradient = np.zeros_like(signal, dtype=np.float64)
    gradient_counts = np.zeros_like(signal)
    for i in range(len(signal) - window_size + 1):
        window_signal = signal[i:i + window_size]
        window_timesteps = timesteps[i:i + window_size]
        p = np.polyfit(window_timesteps, window_signal, order)
        derivative = np.polyval(np.polyder(p, derivative_order), window_timesteps)
        gradient[i:i + window_size] += derivative
        gradient_counts[i:i + window_size] += 1
    for i in range(len(signal)):
        if gradient_counts[i] > 0:
            gradient[i] /= gradient_counts[i]
    return gradient
