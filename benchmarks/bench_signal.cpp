// Phase 3 Benchmark – SIMD-Accelerated Signal Processing
//
// Measures throughput of:
//   1. CAR (Common Average Reference) – AVX-512 vs scalar
//   2. Butterworth bandpass filter   – AVX-512 cross-channel vs scalar
//   3. Full per-frame pipeline       – CAR + bandpass on a NeuralFrame

#include <reuniclus/SignalProcessor.hpp>
#include <reuniclus/NeuralFrame.hpp>

#include <benchmark/benchmark.h>
#include <array>
#include <cmath>
#include <numbers>

using namespace reuniclus;

// ── CAR on 256 channels ───────────────────────────────────────────────────────
static void BM_CAR_256ch(benchmark::State& state) {
    SignalProcessor dsp;
    alignas(64) std::array<float, kChannels> data;
    for (std::size_t i = 0; i < kChannels; ++i)
        data[i] = static_cast<float>(std::sin(i * 0.1));

    for (auto _ : state) {
        dsp.apply_car(data.data(), kChannels);
        benchmark::DoNotOptimize(data[0]);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kChannels));
    state.SetBytesProcessed(
        state.iterations() * static_cast<int64_t>(sizeof(float) * kChannels));
}
BENCHMARK(BM_CAR_256ch)->Iterations(1'000'000);

// ── Bandpass filter on 256 channels ──────────────────────────────────────────
static void BM_Bandpass_256ch(benchmark::State& state) {
    SignalProcessor dsp;
    alignas(64) std::array<float, kChannels> data;
    for (std::size_t i = 0; i < kChannels; ++i)
        data[i] = static_cast<float>(std::sin(i * 0.1));

    for (auto _ : state) {
        dsp.apply_bandpass(data.data(), kChannels);
        benchmark::DoNotOptimize(data[0]);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kChannels));
}
BENCHMARK(BM_Bandpass_256ch)->Iterations(1'000'000);

// ── Full frame processing (CAR + bandpass) ────────────────────────────────────
static void BM_ProcessFrame(benchmark::State& state) {
    SignalProcessor dsp;
    NeuralFrame frame;
    for (std::size_t i = 0; i < kChannels; ++i)
        frame.channels[i] = static_cast<float>(std::sin(i * 0.05));

    for (auto _ : state) {
        dsp.process(frame);
        benchmark::DoNotOptimize(frame.channels[0]);
    }
    // At 1 kHz, one frame = 1 ms budget; target << 1 ms
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(
        state.iterations() * static_cast<int64_t>(sizeof(NeuralFrame)));
}
BENCHMARK(BM_ProcessFrame)->Iterations(1'000'000);

// ── 1 kHz sustained throughput simulation ────────────────────────────────────
// Simulates 1 second of continuous 1 kHz data to verify no throughput cliff.
static void BM_ProcessFrame_1kHz_Simulation(benchmark::State& state) {
    SignalProcessor dsp;
    NeuralFrame frame;
    frame.channels.fill(0.5f);

    for (auto _ : state) {
        for (int t = 0; t < 1000; ++t) {
            // Vary signal so the compiler doesn't optimise the loop away
            frame.channels[0] = static_cast<float>(t) * 0.001f;
            dsp.process(frame);
            benchmark::DoNotOptimize(frame.channels[255]);
        }
    }
    state.SetItemsProcessed(state.iterations() * 1000LL);
}
BENCHMARK(BM_ProcessFrame_1kHz_Simulation);

BENCHMARK_MAIN();
