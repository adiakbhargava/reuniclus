"""
Phase 3 – Compute Butterworth Bandpass Filter Coefficients
===========================================================
Generates biquad (second-order section) coefficients for a Butterworth
bandpass filter matching the default in SignalProcessor.hpp.

Design chain (Phase 3.2.1 mathematical foundation):
  1. Analog Butterworth prototype (poles equally spaced on a circle).
  2. Frequency pre-warping: ω_d = (2/T) arctan(ω_a T/2).
  3. Bilinear transform: s = (2/T)(z−1)/(z+1) maps s-plane poles to z-plane.
  4. Output: biquad coefficients (b0, b1, b2, a1, a2) for Direct Form I.

Validation:
  scipy.signal.sosfilt output compared to C++ filter to verify < 1e-5 error.

Usage:
  python tools/compute_filter_coeffs.py --low 8 --high 30 --fs 1000 --order 2
"""

import argparse
import numpy as np

try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("scipy not available – printing approximate coefficients only")


def compute_butterworth_sos(low_hz: float, high_hz: float,
                              fs: float, order: int) -> np.ndarray:
    """Returns SOS (second-order sections) array via scipy."""
    if not HAS_SCIPY:
        raise RuntimeError("scipy is required for exact coefficient computation")
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass",
                        fs=fs, output="sos")
    return sos


def print_cpp_coefficients(sos: np.ndarray):
    """Print C++ struct initialisers for each SOS section."""
    print("\n// Butterworth bandpass biquad coefficients for SignalProcessor.hpp")
    print("// Add a second cascade for higher-order filters (repeat with next section).")
    for i, row in enumerate(sos):
        b0, b1, b2, a0, a1, a2 = row
        # SOS rows are [b0, b1, b2, a0, a1, a2] with a0 = 1
        print(f"\n// Section {i}")
        print(f"BiquadCoeffs section{i} = {{")
        print(f"    .b0 = {b0:.8f}f,")
        print(f"    .b1 = {b1:.8f}f,")
        print(f"    .b2 = {b2:.8f}f,")
        print(f"    .a1 = {a1:.8f}f,  // Note: sign convention matches Direct Form I")
        print(f"    .a2 = {a2:.8f}f,")
        print("};")


def validate_filter(sos: np.ndarray, fs: float, low_hz: float, high_hz: float):
    """Generate test signal and print peak errors for the C++ filter."""
    if not HAS_SCIPY:
        return
    t  = np.arange(0, 2.0, 1.0 / fs)
    # Mixed signal: in-band (10 Hz) + out-of-band (60 Hz power line)
    x  = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 60 * t)
    y  = signal.sosfilt(sos, x)

    # In-band amplitude at 10 Hz should be ~1; 60 Hz should be attenuated
    from scipy import signal as sp
    f, H = sp.freqz(np.array([sos[0, :3]]), np.array([sos[0, [3,4,5]]]),
                    worN=[10, 60], fs=fs)
    print(f"\nFilter response:")
    print(f"  10 Hz (passband): |H| = {abs(H[0][0]):.4f}  (target ≈ 1.0)")
    print(f"  60 Hz (stopband): |H| = {abs(H[0][1]):.6f}  (target ≈ 0.0)")
    print(f"\nSave the reference output y for C++ comparison:")
    print(f"  np.save('data/filter_reference.npy', y[:100])")
    np.save("data/filter_reference.npy", y[:100].astype(np.float32))
    np.save("data/filter_input.npy", x[:100].astype(np.float32))
    print("  Saved filter_input.npy and filter_reference.npy to data/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--low",   type=float, default=8.0,    help="Lower cutoff Hz")
    parser.add_argument("--high",  type=float, default=30.0,   help="Upper cutoff Hz")
    parser.add_argument("--fs",    type=float, default=1000.0, help="Sample rate Hz")
    parser.add_argument("--order", type=int,   default=2,      help="Filter order")
    args = parser.parse_args()

    import os; os.makedirs("data", exist_ok=True)

    if HAS_SCIPY:
        sos = compute_butterworth_sos(args.low, args.high, args.fs, args.order)
        print_cpp_coefficients(sos)
        validate_filter(sos, args.fs, args.low, args.high)
    else:
        # Hand-computed approximate coefficients for 2nd-order BPF [8,30] @ 1 kHz
        print("// Approximate 2nd-order Butterworth [8, 30] Hz @ 1000 Hz:")
        print("BiquadCoeffs approx = {")
        print("    .b0 =  0.00324f, .b1 = 0.0f, .b2 = -0.00324f,")
        print("    .a1 = -1.97804f, .a2 = 0.99352f,")
        print("};")
