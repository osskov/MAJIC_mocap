import inspect
import numpy as np


def finite_difference(signal: np.ndarray, timesteps: np.ndarray, method: str = 'central', **kwargs) -> np.ndarray:
    """
    Compute the finite difference of a signal using specified methods (central, forward, polyfit).

    Parameters:
    - signal: The signal to differentiate.
    - timesteps: The time points corresponding to each signal value.
    - method: The differentiation method ('central', 'forward', 'backward', or 'polyfit').

    Kwargs for specific methods:
    - edges: Edge handling method for 'central', 'forward' and 'backward' ('extend' or 'zero').
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
        'backward': backward_difference,
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


def _as_columns(signal):
    """(N,) -> (N, 1) plus a flag to undo it. (N, k) passes through."""
    sig = np.asarray(signal, dtype=np.float64)
    if sig.ndim == 1:
        return sig[:, None], True
    return sig, False


def central_difference(signal: np.ndarray, timesteps: np.ndarray, edges='extend') -> np.ndarray:
    """
    This function computes the central difference of a signal given the signal and the corresponding timesteps.

    Accepts (N,) or (N, k); every column is differentiated independently.
    """
    sig, squeeze = _as_columns(signal)
    t = np.asarray(timesteps, dtype=np.float64)

    gradient = np.zeros_like(sig)
    gradient[1:-1] = (sig[2:] - sig[:-2]) / (t[2:] - t[:-2])[:, None]
    if edges == 'extend':
        gradient[0] = gradient[1]
        gradient[-1] = gradient[-2]
    elif edges == 'zero':
        gradient[0] = 0
        gradient[-1] = 0
    else:
        raise ValueError(f"Invalid edges argument: {edges}")
    return gradient[:, 0] if squeeze else gradient


def forward_difference(signal: np.ndarray, timesteps: np.ndarray, edges='extend') -> np.ndarray:
    """
    This function computes the forward difference of a signal given the signal and the corresponding timesteps.

    Accepts (N,) or (N, k); every column is differentiated independently.
    """
    sig, squeeze = _as_columns(signal)
    t = np.asarray(timesteps, dtype=np.float64)

    gradient = np.zeros_like(sig)
    gradient[:-1] = (sig[1:] - sig[:-1]) / (t[1:] - t[:-1])[:, None]
    if edges == 'extend':
        gradient[-1] = gradient[-2]
    elif edges == 'zero':
        gradient[-1] = 0
    else:
        raise ValueError(f"Invalid edges argument: {edges}")
    return gradient[:, 0] if squeeze else gradient


def backward_difference(signal: np.ndarray, timesteps: np.ndarray, edges='extend') -> np.ndarray:
    """
    This function computes the backward difference of a signal given the signal and the
    corresponding timesteps.

    (s[t] - s[t-1]) / dt. The only differentiator here that reads NO future sample, which
    is why it is project_acc's default: the projected acceleration then depends only on
    data up to t, so the pipeline can be run online. The others all look ahead --
    `central` and `forward` by one sample, `polyfit` by window_size - 1 (90 ms with the
    default window at 100 Hz).

    It pays for that with noise: as an FIR differentiator its white-noise gain is 10x
    polyfit's and 2x central's, and it carries half a sample of lag. Both are the price
    of causality, and the intended remedy is an explicit low-pass on the projected
    acceleration rather than a smoother derivative -- smoothing inside the differentiator
    costs lag and a gain error that no downstream filter can undo.

    Note `edges='extend'` fills the first sample from the second, which does read one
    sample ahead. It is one sample out of a trial, and it matches how the other
    differentiators handle their own edges; use edges='zero' if that matters.

    Accepts (N,) or (N, k); every column is differentiated independently.
    """
    sig, squeeze = _as_columns(signal)
    t = np.asarray(timesteps, dtype=np.float64)

    gradient = np.zeros_like(sig)
    gradient[1:] = (sig[1:] - sig[:-1]) / (t[1:] - t[:-1])[:, None]
    if edges == 'extend':
        gradient[0] = gradient[1]
    elif edges == 'zero':
        gradient[0] = 0
    else:
        raise ValueError(f"Invalid edges argument: {edges}")
    return gradient[:, 0] if squeeze else gradient


def _coeff_derivative_matrix(order: int, derivative_order: int) -> np.ndarray:
    """Differentiation in the np.vander basis [u^order, ..., u, 1].

    d/du sends coefficient j to coefficient j+1 scaled by its power, so the operator
    is the sub-diagonal matrix Sd[j, j-1] = (order - (j-1)), applied `derivative_order`
    times.
    """
    powers = np.arange(order, -1, -1, dtype=np.float64)
    Sd = np.zeros((order + 1, order + 1))
    for j in range(1, order + 1):
        Sd[j, j - 1] = powers[j - 1]
    Dc = np.eye(order + 1)
    for _ in range(derivative_order):
        Dc = Sd @ Dc
    return Dc


def _window_operator(t_win: np.ndarray, order: int, derivative_order: int) -> np.ndarray:
    """w x w matrix D with  d^n/dt^n (fit)(t_win) = D @ y_win.

    Built in centred, scaled local time u = (t - mid) / half so the Vandermonde stays
    well conditioned. Fitting on ABSOLUTE timestamps, as the previous per-window
    np.polyfit did, reaches cond(V) ~ 1e21 late in a 600 s trial; numpy's internal
    column scaling keeps that usable but not free, and there is no reason to pay it.
    """
    mid = 0.5 * (t_win[0] + t_win[-1])
    half = 0.5 * (t_win[-1] - t_win[0])
    u = (t_win - mid) / half

    V = np.vander(u, order + 1)
    Dc = _coeff_derivative_matrix(order, derivative_order)
    return (V @ Dc @ np.linalg.pinv(V)) / half ** derivative_order


def _overlap_average(est: np.ndarray, N: int, w: int) -> np.ndarray:
    """Average the per-window estimates back onto the sample grid.

    est is (M, w, k): est[m, j] is window m's estimate for sample m + j. Samples near
    either end are covered by fewer than w windows, and are averaged over however many
    cover them -- matching the partial-window behaviour of the original loop, which
    experiments/acceleration_projection.py trims for.
    """
    M, _, k = est.shape
    gradient = np.zeros((N, k))
    counts = np.zeros(N)
    for j in range(w):
        gradient[j:j + M] += est[:, j, :]
        counts[j:j + M] += 1
    covered = counts > 0
    gradient[covered] /= counts[covered, None]
    return gradient


def polynomial_fit_derivative(signal: np.ndarray, timesteps: np.ndarray, order: int = 3, window_size: int = 10,
                              derivative_order: int = 1) -> np.ndarray:
    """
    This function computes the sliding window polynomial fit derivative of a signal.

    A length-`window_size` window slides over the signal; each position is fitted with a
    polynomial of `order`, differentiated analytically, evaluated at every sample in the
    window, and the overlapping estimates are averaged. That is a smoothed
    Savitzky-Golay derivative.

    NOT CAUSAL. The window extends forwards, so sample t draws on samples up to
    t + window_size - 1 -- 90 ms of lookahead at 100 Hz with the default window. Fine
    for offline analysis; see experiments/ for the causal alternatives.

    Implementation note: for a given set of window timestamps the map
    (window samples) -> (derivative at those samples) is a fixed `window_size` square
    matrix, and when the sampling is uniform it is the SAME matrix for every window. So
    the whole routine is one small matmul over a strided view plus an overlap-add,
    rather than a np.polyfit per sample. Non-uniform sampling falls back to a batched
    least-squares solve, which is slower but still vectorised.

    Accepts (N,) or (N, k); every column is differentiated independently.
    """
    if len(signal) < window_size:
        raise ValueError("window_size (" + str(window_size) + ") must be less than the length of the signal (" + str(
            len(signal)) + ").")

    sig, squeeze = _as_columns(signal)
    t = np.asarray(timesteps, dtype=np.float64)
    N, k = sig.shape
    w = window_size
    M = N - w + 1

    # (M, k, w) view of every window position, no copy
    windows = np.lib.stride_tricks.sliding_window_view(sig, w, axis=0)

    spacing = np.diff(t)
    uniform = np.ptp(spacing) <= 1e-9 * max(abs(float(np.mean(spacing))), 1e-30)

    if uniform:
        D = _window_operator(t[:w], order, derivative_order)
        est = np.einsum('ij,mcj->mic', D, windows)
    else:
        t_win = np.lib.stride_tricks.sliding_window_view(t, w)          # (M, w)
        mid = 0.5 * (t_win[:, :1] + t_win[:, -1:])
        half = 0.5 * (t_win[:, -1:] - t_win[:, :1])
        u = (t_win - mid) / half
        V = np.stack([u ** p for p in range(order, -1, -1)], axis=-1)   # (M, w, order+1)
        Y = np.moveaxis(windows, 1, 2)                                  # (M, w, k)
        coef = np.linalg.solve(np.einsum('mwi,mwj->mij', V, V),
                               np.einsum('mwi,mwk->mik', V, Y))
        Dc = _coeff_derivative_matrix(order, derivative_order)
        est = (np.einsum('mwi,ij,mjk->mwk', V, Dc, coef)
               / half[:, :, None] ** derivative_order)

    gradient = _overlap_average(est, N, w)
    return gradient[:, 0] if squeeze else gradient
