// Phase 1 Benchmark – SPSC Ring Buffer Round-Trip Latency
//
// Validation checkpoint (Phase 1):
//   "Measure round-trip latency: producer pushes a timestamped frame,
//    consumer pops it, measures the delta.
//    Target: < 100 nanoseconds per operation.
//    Run with 1 million iterations to check for outliers."
//
// Queuing theory check (Phase 1.2):
//   With λ/µ = 0.5 and N = 1024, overflow probability < 10^-308.
//   The empirical service time distribution from this benchmark informs
//   the Kingman formula for buffer sizing.

#include <reuniclus/SPSCRingBuffer.hpp>
#include <reuniclus/NeuralFrame.hpp>

#include <benchmark/benchmark.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

using namespace reuniclus;
using Clock = std::chrono::high_resolution_clock;

// ── Single-threaded push + pop latency ───────────────────────────────────────
static void BM_SPSC_PushPop_Scalar(benchmark::State& state) {
    SPSCRingBuffer<int, 1024> buf;
    for (auto _ : state) {
        buf.try_push(42);
        int v;
        buf.try_pop(v);
        benchmark::DoNotOptimize(v);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_SPSC_PushPop_Scalar)->Iterations(1'000'000);

// ── NeuralFrame push + pop (full struct) ─────────────────────────────────────
static void BM_SPSC_NeuralFrame_PushPop(benchmark::State& state) {
    SPSCRingBuffer<NeuralFrame, 1024> buf;
    NeuralFrame frame;
    frame.channels.fill(1.0f);

    for (auto _ : state) {
        buf.try_push(frame);
        NeuralFrame out;
        buf.try_pop(out);
        benchmark::DoNotOptimize(out.channels[0]);
    }
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(
        state.iterations() * static_cast<int64_t>(sizeof(NeuralFrame)));
}
BENCHMARK(BM_SPSC_NeuralFrame_PushPop)->Iterations(1'000'000);

// ── Producer–Consumer cross-thread round-trip (timestamped) ──────────────────
// Measures the delta between push and pop timestamps: the actual inter-thread
// visibility latency including cache coherency effects.
static void BM_SPSC_CrossThread_RoundTrip(benchmark::State& state) {
    struct TimestampedInt {
        int           value;
        long long     push_ts_ns;
    };

    SPSCRingBuffer<TimestampedInt, 1024> buf;
    std::atomic<bool> go{false};
    std::atomic<bool> stop{false};

    // Results vector (pre-allocated, no allocation on hot path)
    std::vector<long long> latencies;
    latencies.reserve(1'000'000);

    std::thread consumer([&] {
        while (!go.load(std::memory_order_acquire)) {}
        TimestampedInt item;
        while (!stop.load(std::memory_order_relaxed)) {
            if (buf.try_pop(item)) {
                long long pop_ts = Clock::now().time_since_epoch().count();
                latencies.push_back(pop_ts - item.push_ts_ns);
            }
        }
    });

    go.store(true, std::memory_order_release);

    for (auto _ : state) {
        TimestampedInt item{42, Clock::now().time_since_epoch().count()};
        while (!buf.try_push(item)) {}
    }

    stop.store(true);
    consumer.join();

    if (!latencies.empty()) {
        std::sort(latencies.begin(), latencies.end());
        auto n    = latencies.size();
        double mean = 0;
        for (auto v : latencies) mean += v;
        mean /= n;

        state.counters["mean_ns"]   = mean;
        state.counters["p50_ns"]    = latencies[n / 2];
        state.counters["p99_ns"]    = latencies[static_cast<std::size_t>(n * 0.99)];
        state.counters["max_ns"]    = latencies.back();
        state.counters["n_samples"] = static_cast<double>(n);

        // Validation: target < 100 ns mean
        if (mean > 100.0) {
            state.SkipWithError("Round-trip mean exceeds 100 ns target");
        }
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_SPSC_CrossThread_RoundTrip)->Iterations(1'000'000)->UseRealTime();

// ── Throughput: sustained 1 MHz push rate (10× above 100 kHz BCI) ─────────────
static void BM_SPSC_ThroughputSustained(benchmark::State& state) {
    SPSCRingBuffer<int, 1024> buf;
    std::atomic<bool> stop{false};

    std::thread consumer([&] {
        int v;
        while (!stop.load(std::memory_order_relaxed)) buf.try_pop(v);
    });

    for (auto _ : state) {
        while (!buf.try_push(1)) {}
    }

    stop.store(true);
    consumer.join();

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_SPSC_ThroughputSustained)->Iterations(1'000'000)->UseRealTime();

BENCHMARK_MAIN();
