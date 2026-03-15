#pragma once

// Phase 8.4 – Telemetry System
//
// Records per-frame latency, decoder confidence, Kalman innovation statistics,
// and posterior entropy.  All writes go to a pre-allocated ring buffer (no
// heap allocation on the hot path).
//
// Analysis outputs (Phase 8.4):
//   • Latency histogram: distribution of per-frame processing times.
//   • Summary statistics: mean, median, p95, p99, max.
//     Targets: mean < 500 µs, p99 < 1 ms.
//   • Innovation whiteness: autocorrelation check (Ljung-Box test).
//   • Calibration plot: predicted vs. observed confidence (should be 1:1).

#include <reuniclus/SPSCRingBuffer.hpp>

#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <vector>
#include <string>

namespace reuniclus {

struct TelemetryRecord {
    long long latency_ns;    // End-to-end processing latency (nanoseconds)
    float     confidence;    // Kalman decoder confidence [0, 1]
    float     entropy;       // Posterior entropy H(z|x)
    float     maha_dist;     // Mahalanobis distance of Kalman innovation
    long long wall_time_ns;  // Absolute wall-clock timestamp (ns since epoch)
};

class Telemetry {
public:
    static constexpr std::size_t kBufSize = 65536;  // Power-of-2: ~65 s at 1 kHz

    // ── Hot-path record (called from the processing loop) ─────────────────
    // Zero allocation: writes to a pre-allocated ring buffer.
    void record(std::chrono::nanoseconds latency,
                float                   confidence,
                float                   entropy    = 0.0f,
                float                   maha_dist  = 0.0f) noexcept
    {
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
        TelemetryRecord rec{
            .latency_ns   = latency.count(),
            .confidence   = confidence,
            .entropy      = entropy,
            .maha_dist    = maha_dist,
            .wall_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count()
        };
        (void)buf_.try_push(rec);  // drop silently if full — telemetry is best-effort
        total_frames_.fetch_add(1, std::memory_order_relaxed);
    }

    // ── Drain buffer into a local vector (called from Telemetry thread) ───
    void flush(std::vector<TelemetryRecord>& out) {
        TelemetryRecord rec;
        while (buf_.try_pop(rec)) out.push_back(rec);
    }

    // ── Summary statistics (Phase 8.4) ────────────────────────────────────
    struct Stats {
        double mean_ns, median_ns, p95_ns, p99_ns, max_ns;
        double mean_confidence;
        double mean_entropy;
        std::size_t n_frames;
    };

    static Stats compute_stats(const std::vector<TelemetryRecord>& records) {
        if (records.empty()) return {};
        std::vector<double> lat;
        lat.reserve(records.size());
        double sum_conf = 0, sum_ent = 0;
        for (auto& r : records) {
            lat.push_back(static_cast<double>(r.latency_ns));
            sum_conf += r.confidence;
            sum_ent  += r.entropy;
        }
        std::sort(lat.begin(), lat.end());
        auto n = lat.size();
        Stats s{};
        s.n_frames        = n;
        s.mean_ns         = std::accumulate(lat.begin(), lat.end(), 0.0) / n;
        s.median_ns       = lat[n / 2];
        s.p95_ns          = lat[static_cast<std::size_t>(n * 0.95)];
        s.p99_ns          = lat[static_cast<std::size_t>(n * 0.99)];
        s.max_ns          = lat.back();
        s.mean_confidence = sum_conf / n;
        s.mean_entropy    = sum_ent  / n;
        return s;
    }

    // ── Ljung-Box whiteness test on innovation sequence ───────────────────
    // Returns test statistic Q.  Q ~ χ²(lags) under H0 (white noise).
    // p-value > 0.05 confirms white innovations (model is well-specified).
    static double ljung_box(const std::vector<float>& series, int lags = 20) {
        if (series.size() < static_cast<std::size_t>(lags + 2)) return 0.0;
        const int n = static_cast<int>(series.size());
        // Sample autocorrelations at lags 1..lags
        double mean = 0;
        for (float v : series) mean += v;
        mean /= n;
        double var = 0;
        for (float v : series) var += (v - mean) * (v - mean);
        var /= n;
        if (var < 1e-12) return 0.0;

        double Q = 0.0;
        for (int k = 1; k <= lags; ++k) {
            double rk = 0;
            for (int t = k; t < n; ++t)
                rk += (series[t] - mean) * (series[t-k] - mean);
            rk /= (n * var);
            Q += (rk * rk) / (n - k);
        }
        Q *= static_cast<double>(n * (n + 2));
        return Q;
    }

    // ── Write CSV report for offline analysis ────────────────────────────
    static void write_csv(const std::vector<TelemetryRecord>& records,
                           const std::string& path) {
        std::ofstream f(path);
        f << "wall_time_ns,latency_ns,confidence,entropy,maha_dist\n";
        for (auto& r : records) {
            f << r.wall_time_ns << ','
              << r.latency_ns   << ','
              << r.confidence   << ','
              << r.entropy      << ','
              << r.maha_dist    << '\n';
        }
    }

    [[nodiscard]] std::size_t total_frames() const noexcept {
        return total_frames_.load(std::memory_order_relaxed);
    }

private:
    SPSCRingBuffer<TelemetryRecord, kBufSize> buf_;
    std::atomic<std::size_t>                  total_frames_{0};
};

} // namespace reuniclus
