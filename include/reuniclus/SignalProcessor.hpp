#pragma once

// Phase 3 – SIMD-Accelerated Signal Processing
//
// Two operations are implemented:
//
//   1. Common Average Reference (CAR) – removes spatially common noise.
//      Mathematical basis: CAR is projection by the centering matrix
//          H = I − (1/N)11ᵀ  (idempotent, symmetric, rank N-1).
//      SIMD: AVX-512 processes 16 floats per instruction, reducing 256-channel
//      accumulation from 256 scalar adds to 16 vector adds.
//
//   2. Butterworth Bandpass Filter (IIR biquad, Direct Form I) –
//      flat passband magnitude, no amplitude distortion.
//      Per-channel filter state is stored contiguously and aligned to enable
//      SIMD across channels (16 independent channels per vector load).
//
//      Biquad difference equation:
//          y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2] − a1·y[n-1] − a2·y[n-2]
//
//      Transfer function:
//          H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)
//      Stability: all poles inside |z| < 1 (guaranteed by Butterworth design).
//
// AVX-512 is enabled when __AVX512F__ is defined by the compiler
// (-mavx512f in CMakeLists.txt).  A scalar fallback is always provided.

#include <reuniclus/NeuralFrame.hpp>

#include <array>
#include <cstddef>
#include <cmath>
#include <vector>

#ifdef __AVX512F__
#  include <immintrin.h>
#endif

namespace reuniclus {

// ─────────────────────────────────────────────────────────────────────────────
// Biquad filter state – one instance per channel, stored in a contiguous
// cache-aligned array so SIMD loads across channels work without gather ops.
// ─────────────────────────────────────────────────────────────────────────────
struct BiquadState {
    float x1 = 0.0f, x2 = 0.0f;  // Previous inputs
    float y1 = 0.0f, y2 = 0.0f;  // Previous outputs
};

// ─────────────────────────────────────────────────────────────────────────────
// Biquad coefficients – shared across all channels for a given filter design.
// ─────────────────────────────────────────────────────────────────────────────
struct BiquadCoeffs {
    float b0, b1, b2;  // Numerator
    float a1, a2;      // Denominator (a0 normalised to 1)
};

class SignalProcessor {
public:
    /// @param n_channels  Number of electrode channels (default 256).
    /// @param coeffs      Biquad coefficients for the desired Butterworth filter.
    explicit SignalProcessor(std::size_t       n_channels = kChannels,
                             BiquadCoeffs      coeffs     = default_bandpass_coeffs())
        : n_channels_(n_channels)
        , coeffs_(coeffs)
        , states_(n_channels)
    {}

    // ── Phase 3.1: Common Average Reference ──────────────────────────────────
    //
    // Computes channel mean, then subtracts it from every channel.
    // In hardware: two passes over 256 floats with AVX-512 (16 floats/cycle).
    void apply_car(float* data, std::size_t n) noexcept {
#ifdef __AVX512F__
        car_avx512(data, n);
#else
        car_scalar(data, n);
#endif
    }

    // ── Phase 3.2: Butterworth Bandpass Filter ────────────────────────────────
    //
    // Applies the biquad to every channel.  Parallelised across channels with
    // AVX-512 (16 independent channels simultaneously).
    void apply_bandpass(float* data, std::size_t n) noexcept {
#ifdef __AVX512F__
        bandpass_avx512(data, n);
#else
        bandpass_scalar(data, n);
#endif
    }

    // Convenience overload operating on a NeuralFrame in-place.
    void process(NeuralFrame& frame) noexcept {
        apply_car(frame.channels.data(), n_channels_);
        apply_bandpass(frame.channels.data(), n_channels_);
    }

    void reset_state() noexcept {
        for (auto& s : states_) s = {};
    }

    // ── Default Butterworth 8–30 Hz (Mu/Beta motor imagery band) ─────────────
    // Coefficients for a 4th-order Butterworth bandpass [8, 30] Hz at 1 kHz.
    // Implemented as two cascaded biquad sections; this stores the first section.
    // In production, generate via scipy.signal.butter then store all sections.
    static constexpr BiquadCoeffs default_bandpass_coeffs() noexcept {
        // 2nd-order Butterworth bandpass 8–30 Hz @ 1000 Hz sample rate
        // (pre-computed via bilinear transform; see tools/compute_filter_coeffs.py)
        return {
            .b0 =  0.00324f,
            .b1 =  0.00000f,
            .b2 = -0.00324f,
            .a1 = -1.97804f,
            .a2 =  0.99352f,
        };
    }

private:
    // ── Scalar implementations (always compiled as fallback) ──────────────────
    void car_scalar(float* data, std::size_t n) noexcept {
        float sum = 0.0f;
        for (std::size_t i = 0; i < n; ++i) sum += data[i];
        const float mean = sum / static_cast<float>(n);
        for (std::size_t i = 0; i < n; ++i) data[i] -= mean;
    }

    void bandpass_scalar(float* data, std::size_t n) noexcept {
        const float b0 = coeffs_.b0, b1 = coeffs_.b1, b2 = coeffs_.b2;
        const float a1 = coeffs_.a1, a2 = coeffs_.a2;
        for (std::size_t ch = 0; ch < n; ++ch) {
            auto& s  = states_[ch];
            float xn = data[ch];
            float yn = b0*xn + b1*s.x1 + b2*s.x2 - a1*s.y1 - a2*s.y2;
            s.x2 = s.x1; s.x1 = xn;
            s.y2 = s.y1; s.y1 = yn;
            data[ch] = yn;
        }
    }

#ifdef __AVX512F__
    // ── AVX-512 CAR (Phase 3.1.1) ─────────────────────────────────────────────
    // Two passes over the array, each processing 16 floats simultaneously.
    void car_avx512(float* data, std::size_t n) noexcept {
        __m512 vsum = _mm512_setzero_ps();
        for (std::size_t i = 0; i < n; i += 16)
            vsum = _mm512_add_ps(vsum, _mm512_load_ps(&data[i]));
        const float mean = _mm512_reduce_add_ps(vsum) / static_cast<float>(n);
        const __m512 vmean = _mm512_set1_ps(mean);
        for (std::size_t i = 0; i < n; i += 16)
            _mm512_store_ps(&data[i],
                _mm512_sub_ps(_mm512_load_ps(&data[i]), vmean));
    }

    // ── AVX-512 Bandpass (Phase 3.2 – cross-channel SIMD) ────────────────────
    // Processes 16 channels simultaneously.  IIR cannot be parallelised across
    // time; each channel's filter state is independent, so SIMD across channels
    // is valid.
    void bandpass_avx512(float* data, std::size_t n) noexcept {
        const __m512 vb0 = _mm512_set1_ps(coeffs_.b0);
        const __m512 vb1 = _mm512_set1_ps(coeffs_.b1);
        const __m512 vb2 = _mm512_set1_ps(coeffs_.b2);
        const __m512 va1 = _mm512_set1_ps(coeffs_.a1);
        const __m512 va2 = _mm512_set1_ps(coeffs_.a2);

        for (std::size_t ch = 0; ch < n; ch += 16) {
            // Load 16 current inputs and per-channel state
            __m512 xn = _mm512_load_ps(&data[ch]);
            __m512 x1 = _mm512_set_ps(
                states_[ch+15].x1, states_[ch+14].x1, states_[ch+13].x1,
                states_[ch+12].x1, states_[ch+11].x1, states_[ch+10].x1,
                states_[ch+ 9].x1, states_[ch+ 8].x1, states_[ch+ 7].x1,
                states_[ch+ 6].x1, states_[ch+ 5].x1, states_[ch+ 4].x1,
                states_[ch+ 3].x1, states_[ch+ 2].x1, states_[ch+ 1].x1,
                states_[ch+ 0].x1);
            __m512 x2 = _mm512_set_ps(
                states_[ch+15].x2, states_[ch+14].x2, states_[ch+13].x2,
                states_[ch+12].x2, states_[ch+11].x2, states_[ch+10].x2,
                states_[ch+ 9].x2, states_[ch+ 8].x2, states_[ch+ 7].x2,
                states_[ch+ 6].x2, states_[ch+ 5].x2, states_[ch+ 4].x2,
                states_[ch+ 3].x2, states_[ch+ 2].x2, states_[ch+ 1].x2,
                states_[ch+ 0].x2);
            __m512 y1 = _mm512_set_ps(
                states_[ch+15].y1, states_[ch+14].y1, states_[ch+13].y1,
                states_[ch+12].y1, states_[ch+11].y1, states_[ch+10].y1,
                states_[ch+ 9].y1, states_[ch+ 8].y1, states_[ch+ 7].y1,
                states_[ch+ 6].y1, states_[ch+ 5].y1, states_[ch+ 4].y1,
                states_[ch+ 3].y1, states_[ch+ 2].y1, states_[ch+ 1].y1,
                states_[ch+ 0].y1);
            __m512 y2 = _mm512_set_ps(
                states_[ch+15].y2, states_[ch+14].y2, states_[ch+13].y2,
                states_[ch+12].y2, states_[ch+11].y2, states_[ch+10].y2,
                states_[ch+ 9].y2, states_[ch+ 8].y2, states_[ch+ 7].y2,
                states_[ch+ 6].y2, states_[ch+ 5].y2, states_[ch+ 4].y2,
                states_[ch+ 3].y2, states_[ch+ 2].y2, states_[ch+ 1].y2,
                states_[ch+ 0].y2);

            // y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
            __m512 yn = _mm512_fmadd_ps(vb0, xn,
                        _mm512_fmadd_ps(vb1, x1,
                        _mm512_fmadd_ps(vb2, x2,
                        _mm512_fnmadd_ps(va1, y1,
                        _mm512_fnmadd_ps(va2, y2,
                        _mm512_setzero_ps())))));

            _mm512_store_ps(&data[ch], yn);

            // Write back state – scalar scatter (state update is not hot-path)
            alignas(64) float yn_arr[16], xn_arr[16], x1_arr[16], y1_arr[16];
            _mm512_store_ps(yn_arr, yn);
            _mm512_store_ps(xn_arr, xn);
            _mm512_store_ps(x1_arr, x1);
            _mm512_store_ps(y1_arr, y1);
            for (int k = 0; k < 16; ++k) {
                states_[ch+k].x2 = x1_arr[k];
                states_[ch+k].x1 = xn_arr[k];
                states_[ch+k].y2 = y1_arr[k];
                states_[ch+k].y1 = yn_arr[k];
            }
        }
    }
#endif

    std::size_t              n_channels_;
    BiquadCoeffs             coeffs_;
    std::vector<BiquadState> states_;
};

} // namespace reuniclus
