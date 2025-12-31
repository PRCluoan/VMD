"""VMD example for a synthetic vibration signal.

This module provides a lightweight Python implementation of the Variational
Mode Decomposition (VMD) algorithm along with a demonstration tailored to
rotating machinery vibration analysis. The example synthesises a vibration
signal containing a rotating component, a modulated resonance and broadband
noise, then decomposes it into intrinsic mode functions (IMFs).
"""

import numpy as np
import matplotlib.pyplot as plt


def vmd(signal, alpha=2000, tau=0, k=3, dc=0, init=1, tol=1e-6, n_iters=500):
    """Decompose *signal* into *k* intrinsic mode functions using VMD.

    Parameters
    ----------
    signal : array-like
        Real-valued 1D input signal.
    alpha : float, optional
        Quadratic penalty for the bandwidth of each mode.
    tau : float, optional
        Time-step of the dual ascent; set to ``0`` for noise slack.
    k : int, optional
        Number of modes to extract.
    dc : int, optional
        If ``1``, the first mode is forced to remain at 0 Hz (DC component).
    init : int, optional
        Initialization strategy for center frequencies: ``0`` = zeros,
        ``1`` = uniform spread, ``2`` = random distribution.
    tol : float, optional
        Tolerance for convergence. Typical values are around ``1e-6``.
    n_iters : int, optional
        Maximum number of iterations of the optimization loop.

    Returns
    -------
    imfs : ndarray
        Array of shape ``(k, len(signal))`` containing the decomposed modes.
    center_freqs : ndarray
        Column vector with the estimated center frequencies of each mode
        (normalized to the Nyquist frequency).
    """

    x = np.array(signal, dtype=float)
    length = x.shape[0]

    # Mirror extension to mitigate boundary effects.
    mirrored = np.concatenate([x[::-1], x, x[::-1]])
    t = mirrored.shape[0]

    # Frequency grid.
    freq_grid = (np.arange(t) - t / 2) / t

    # Fourier transform of the mirrored signal and analytic (positive) part.
    f_hat = np.fft.fft(mirrored)
    f_hat_plus = np.copy(f_hat)
    f_hat_plus[: t // 2] = 0

    # Initialization of variables.
    u_hat_plus = np.zeros((k, t), dtype=complex)
    omega = np.zeros(k)
    if init == 1:
        omega = np.linspace(0, 0.5, k, endpoint=False)
    elif init == 2:
        omega = np.sort(np.abs(np.random.randn(k)) / 2)
    if dc:
        omega[0] = 0

    lambda_hat = np.zeros(t, dtype=complex)
    u_hat_prev = np.zeros_like(u_hat_plus)

    # Main loop.
    for _ in range(n_iters):
        for i in range(k):
            # Accumulate all other modes.
            sum_others = np.sum(u_hat_plus, axis=0) - u_hat_plus[i]
            residual = f_hat_plus - sum_others - lambda_hat / 2

            # Frequency-shifted Wiener filter for mode i.
            denominator = 1 + 2 * alpha * (freq_grid - omega[i]) ** 2
            u_hat_plus[i] = residual / denominator

            # Update center frequency.
            power = np.abs(u_hat_plus[i]) ** 2
            omega[i] = np.sum(freq_grid * power) / (np.sum(power) + np.finfo(float).eps)
            if dc and i == 0:
                omega[i] = 0

        # Dual ascent.
        lambda_hat = lambda_hat + tau * (np.sum(u_hat_plus, axis=0) - f_hat_plus)

        # Convergence check.
        diff = np.linalg.norm(u_hat_plus - u_hat_prev) / (np.linalg.norm(u_hat_prev) + np.finfo(float).eps)
        if diff < tol:
            break
        u_hat_prev = np.copy(u_hat_plus)

    # Reconstruct the modes and trim the mirrored parts.
    imfs_full = np.real(np.fft.ifft(u_hat_plus, axis=1))
    start = length
    end = 2 * length
    imfs = imfs_full[:, start:end]
    return imfs, omega.reshape(-1, 1)


def generate_vibration_signal(fs=6400, duration=1.0, noise_std=0.05):
    """Create a synthetic vibration waveform with multiple fault-like tones."""
    t = np.arange(0, duration, 1 / fs)

    # Fundamental shaft rotation and its harmonic.
    shaft = 0.8 * np.sin(2 * np.pi * 40 * t)
    harmonic = 0.4 * np.sin(2 * np.pi * 80 * t)

    # Resonant response excited by periodic impacts (bearing fault surrogate).
    impact_rate = 24  # Hz
    envelope = np.zeros_like(t)
    for idx in range(int(duration * impact_rate)):
        impact_time = idx / impact_rate
        envelope += np.exp(-600 * (t - impact_time) ** 2)
    resonance = 0.6 * envelope * np.sin(2 * np.pi * 300 * t)

    noise = noise_std * np.random.randn(t.size)
    signal = shaft + harmonic + resonance + noise
    return t, signal


def plot_results(t, signal, imfs, omega, save_path="vibration_vmd.png"):
    """Visualize the original vibration signal and its VMD components."""
    fig, axes = plt.subplots(imfs.shape[0] + 1, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, signal, color="black", linewidth=0.9)
    axes[0].set_title("Synthetic vibration signal")
    axes[0].set_ylabel("Amplitude")

    for i, ax in enumerate(axes[1:], start=0):
        ax.plot(t, imfs[i], linewidth=0.9)
        ax.set_ylabel(f"IMF {i + 1}\n$\\omega_c={omega[i,0]*2:.3f}$")

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)


if __name__ == "__main__":
    np.random.seed(0)
    time, vibration = generate_vibration_signal()
    modes, center_freqs = vmd(vibration, alpha=500, tau=0, k=4, dc=0, init=1, tol=1e-5, n_iters=800)
    plot_results(time, vibration, modes, center_freqs)
    print("Center frequencies (normalized to Nyquist):")
    for idx, freq in enumerate(center_freqs.flatten(), start=1):
        print(f"IMF {idx}: {freq:.4f}")
    print("Figure saved to vibration_vmd.png")
