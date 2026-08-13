"""
Band-limited resampling onto an arbitrary time grid, shared by IMUTrace and WorldTrace.

Resampling in this repo used to mean `interp1d(kind='linear')` on the IMU only. That is not
band-limited: its response is sinc^2, so upsampling 40 -> 100 Hz cost -1.2 dB at 8 Hz and
-7.8 dB at 20 Hz, and it fabricated spectral images at (source_rate +/- f). Because the mocap
was never resampled to match, the two sides of every IMU-vs-mocap comparison sat in different
bands. Measured cost: the IMoVE cluster-to-IMU offset came out 7-9 mm too large on the shank.

Two rules follow, and this module exists to make them the default.

  DOWNSAMPLING MUST ANTI-ALIAS. Subsampling without a filter folds everything above the new
  Nyquist back into the signal band. That is at its worst here, because mocap acceleration
  comes from twice-differentiated marker positions and so has f^2-amplified noise -- the
  loudest content sits exactly at the top of the band, and is the worst thing to fold down.
  Done properly, decimation is not lossy averaging but a noise REDUCTION: each output sample
  is a weighted sum over the filter's support, and broadband noise drops by the bandwidth
  ratio (100 -> 40 Hz measures 1.62x quieter, against sqrt(40/100) = 1.58 in theory).

  THE OUTPUT GRID IS ARBITRARY. Alignment wants a FRACTIONAL sample shift -- at 40 Hz,
  integer-sample sync leaves up to +/-12.5 ms of residual error -- so resampling takes
  explicit output timestamps rather than only a rate.

The method is: anti-alias at the source rate if downsampling, then evaluate a cubic spline at
the requested times. After filtering, the signal is heavily oversampled relative to its own
bandwidth, which is the regime where spline interpolation is accurate. `resample_poly` cannot
be used instead because it only produces rational rate changes on a grid starting at sample
zero, and cannot express a fractional offset.
"""
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import binary_dilation
from scipy.signal import butter, resample_poly, sosfiltfilt
from scipy.spatial.transform import Rotation

# Anti-alias cutoff as a fraction of the TARGET rate. 0.45 leaves a little transition band
# below the new Nyquist (0.5) rather than sitting on it, which an order-8 Butterworth needs
# to actually be down by the time it matters.
ANTIALIAS_FRACTION = 0.45
ANTIALIAS_ORDER = 8

# Integer upsampling applied before the spline so it only ever interpolates a densely
# oversampled signal. 8x puts spline error below a part in 10^4 for anything in band.
OVERSAMPLE = 8

# Window for the polyphase FIR. resample_poly defaults to ('kaiser', 5.0), whose passband
# ripple is a systematic GAIN ERROR of about 0.07% -- visible as an error growing linearly
# along a position ramp, and as a 5 Hz tone coming back at 1.000666. Since the 8x upsample
# leaves an enormous transition band, a much tighter window costs nothing: beta 14 takes the
# ramp error from 6.7e-4 to 1.8e-7 and the tone gain to 1.000000.
OVERSAMPLE_WINDOW = ('kaiser', 14.0)

# How far a gap contaminates its neighbours, in periods of the anti-alias cutoff. filtfilt
# runs the filter both ways, so ringing spreads either side; three periods is where an
# order-8 Butterworth impulse response has decayed into the noise. Used to widen the invalid
# mask, because filtering across a dropout silently poisons the samples around it and the
# mask is the only thing that can say so.
GAP_CONTAMINATION_PERIODS = 3.0


def antialias(values: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
    """Low-passes `values` (N, ...) along axis 0 ahead of a rate reduction.

    A no-op when `target_rate >= source_rate`: upsampling cannot alias, and filtering it
    would throw away signal for nothing.
    """
    if target_rate >= source_rate:
        return values
    cutoff = ANTIALIAS_FRACTION * target_rate

    # sosfiltfilt needs the signal to be longer than its padding, which for an order-8
    # Butterworth is 26 samples. Short traces are legitimate here -- test fixtures, and
    # single segments carved out of a trial -- so drop the order until the filter fits
    # rather than refusing. Below order 2 there is nothing left to drop; a trace that short
    # carries no resolvable frequency content anyway, so it passes through unfiltered.
    order = ANTIALIAS_ORDER
    while order >= 2 and 3 * (2 * (order // 2) + 1) >= len(values):
        order -= 2
    if order < 2:
        return values

    sos = butter(order, cutoff, btype='low', fs=source_rate, output='sos')
    flat = sosfiltfilt(sos, values.reshape(len(values), -1), axis=0)
    return flat.reshape(values.shape)


def contamination_width(source_rate: float, target_rate: float) -> int:
    """How many source samples either side of a gap the anti-alias filter corrupts."""
    if target_rate >= source_rate:
        return 0
    cutoff = ANTIALIAS_FRACTION * target_rate
    return int(np.ceil(GAP_CONTAMINATION_PERIODS * source_rate / cutoff))


def resample_values(values: np.ndarray, timestamps: np.ndarray,
                    new_timestamps: np.ndarray, target_rate: Optional[float] = None,
                    source_rate: Optional[float] = None) -> np.ndarray:
    """Band-limited resample of (N, ...) `values` onto `new_timestamps`.

    `target_rate` defaults to the new grid's own rate. Pass it explicitly when the new grid
    is shorter than the signal (so its rate cannot be inferred reliably) or when a stricter
    band limit than the output rate is wanted.
    """
    values = np.asarray(values, dtype=np.float64)
    if source_rate is None:
        source_rate = 1.0 / np.mean(np.diff(timestamps))
    if target_rate is None:
        target_rate = (1.0 / np.mean(np.diff(new_timestamps))
                       if len(new_timestamps) > 1 else source_rate)

    filtered = antialias(values, source_rate, target_rate)

    # A cubic spline is only as good as its oversampling. Interpolating an 8 Hz component
    # sampled at 40 Hz -- 2.5x its own Nyquist -- loses 4.6% of amplitude, which is better
    # than linear interpolation's 12.5% but still a real error. resample_poly IS exact for
    # a rational rate change, so use it to get onto a densely oversampled grid first and let
    # the spline handle only the last fractional step, where it is essentially perfect.
    # Measured: 0.954 -> 1.000 gain on that 8 Hz case.
    #
    # This cannot simply BE resample_poly: it only lands on grids starting at sample zero
    # and cannot express the fractional time shift that alignment needs.
    dense = resample_poly(filtered, OVERSAMPLE, 1, axis=0, padtype='line',
                          window=OVERSAMPLE_WINDOW)
    dense_timestamps = timestamps[0] + np.arange(len(dense)) / (source_rate * OVERSAMPLE)

    # HOLD outside the source span rather than extrapolating. A cubic run past the end of its
    # data diverges as t^3: asked for the 430 s of Subject01's walking trial that precede the
    # mocap, it returned marker positions of 1.3e10 m. Those frames are marked invalid either
    # way, so no statistic was wrong -- but they sit in the same arrays that filters integrate
    # through and that finite differences run over, and a derivative of 1e10 is not inert.
    # Clamping reproduces what the old index-clipping did: the nearest real pose, held, which
    # has zero angular velocity and disturbs nothing.
    clamped = np.clip(new_timestamps, dense_timestamps[0], dense_timestamps[-1])
    return CubicSpline(dense_timestamps, dense, axis=0, extrapolate=False)(clamped)


def resample_mask(valid: np.ndarray, timestamps: np.ndarray, new_timestamps: np.ndarray,
                  source_rate: Optional[float] = None,
                  target_rate: Optional[float] = None) -> np.ndarray:
    """Carries a boolean validity mask through the same resample, conservatively.

    Widened by `contamination_width` BEFORE being sampled, because the anti-alias filter
    mixes each invalid frame into its neighbours. Without this a single dropped frame stays
    marked as one bad sample while having quietly corrupted the dozen around it.

    Sampled by nearest neighbour rather than interpolated: a mask is not a signal, and any
    smoothing of it would invent partially-valid frames.
    """
    valid = np.asarray(valid, dtype=bool)
    if source_rate is None:
        source_rate = 1.0 / np.mean(np.diff(timestamps))
    if target_rate is None:
        target_rate = (1.0 / np.mean(np.diff(new_timestamps))
                       if len(new_timestamps) > 1 else source_rate)

    width = contamination_width(source_rate, target_rate)
    if width:
        invalid = binary_dilation(~valid, structure=np.ones(2 * width + 1, dtype=bool))
        valid = ~invalid

    nearest = np.clip(np.searchsorted(timestamps, new_timestamps), 0, len(valid) - 1)
    # searchsorted rounds up; step back where the previous sample is genuinely closer.
    previous = np.clip(nearest - 1, 0, len(valid) - 1)
    take_previous = (np.abs(timestamps[previous] - new_timestamps)
                     < np.abs(timestamps[nearest] - new_timestamps))
    nearest = np.where(take_previous, previous, nearest)

    # Anything outside the source's span was never observed at all.
    inside = ((new_timestamps >= timestamps[0] - 1e-9)
              & (new_timestamps <= timestamps[-1] + 1e-9))
    return valid[nearest] & inside


def resample_rotations(rotations: np.ndarray, timestamps: np.ndarray,
                       new_timestamps: np.ndarray, source_rate: Optional[float] = None,
                       target_rate: Optional[float] = None) -> np.ndarray:
    """Band-limited resample of (N, 3, 3) rotations, via sign-continuous quaternions.

    Filtering an orientation is not a well-defined operation -- SO(3) is not a vector space,
    so every method is some linear operation in a chosen chart plus a projection back. The
    chart matters, and it was measured (scratch/compare_rotation_resampling.py, 100 -> 40 Hz
    against analytic ground truth, worst-case RMS over four trajectories):

        quaternion components + renormalize   0.19 deg   <- this
        matrix components + SVD projection    0.60 deg
        absolute axis-angle vector            6.57 deg
        SLERP with no anti-alias filter       1.71 deg
        filter increments and re-integrate   77.9  deg

    Quaternions beat matrices for a specific reason: a rotation of angle theta moves its
    quaternion by theta/2 on the unit 3-sphere, so the chord that linear filtering cuts is
    half as long and the error, being quadratic in the arc, is about four times smaller.
    Measured 3.2x on a 2985 deg/s trajectory; the two are indistinguishable below ~150 deg/s.

    The two rejected options are rejected on correctness, not accuracy:

      ABSOLUTE AXIS-ANGLE wraps. Past +/-pi the vector jumps the long way round and the
      filter smears the discontinuity across its whole support. Harmless on a bounded joint
      angle, fatal on world-frame yaw over a 632 s walk with turns -- 57 deg peak error here.
      WorldTrace.lowpass_filter still does this.

      FILTERING INCREMENTS AND RE-INTEGRATING drifts without bound. The increments carry a
      second-order BCH (coning) DC term generated by the high-frequency content -- 4.24e-4
      rad/sample for a 30 Hz wobble, integrating to 145.6 deg over 60 s against a true drift
      of 3.2 deg. Unfiltered, that bias is cancelled by the AC terms it composes against,
      since rotations do not commute. A zero-phase filter preserves DC and removes the AC,
      so it destroys the cancellation and keeps the bias. This is the standard Local Tangent
      Space Filtering recipe, and it is unsafe here for exactly the signals a resampler
      exists to remove.

    Sign continuity is applied first: q and -q are the same rotation, and an unforced sign
    flip is a step discontinuity that the filter would smear just like a wrap.
    """
    quaternions = _sign_continuous(Rotation.from_matrix(rotations).as_quat())
    resampled = resample_values(quaternions, timestamps, new_timestamps,
                                source_rate=source_rate, target_rate=target_rate)
    norms = np.linalg.norm(resampled, axis=1, keepdims=True)
    return Rotation.from_quat(resampled / norms).as_matrix()


def _sign_continuous(quaternions: np.ndarray) -> np.ndarray:
    """Puts a quaternion sequence on one continuous branch of the double cover."""
    dots = np.sum(quaternions[1:] * quaternions[:-1], axis=1)
    flip = np.cumprod(np.where(dots < 0, -1.0, 1.0))
    return np.vstack([quaternions[:1], quaternions[1:] * flip[:, None]])


def orthonormalize(rotations: np.ndarray) -> np.ndarray:
    """Projects (N, 3, 3) back onto SO(3) after component-wise interpolation.

    Filtering and interpolating a rotation matrix entry by entry does not preserve
    orthonormality -- the result is close to a rotation but not one, and anything that later
    inverts it by transposing would be subtly wrong. The nearest true rotation in the
    Frobenius sense is U V^T from the SVD, with the determinant forced positive so a
    near-degenerate frame cannot flip handedness.
    """
    u, _, vt = np.linalg.svd(rotations)
    determinant = np.linalg.det(np.matmul(u, vt))
    u[:, :, -1] *= np.sign(determinant)[:, None]
    return np.matmul(u, vt)
