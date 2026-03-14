#pragma once

// Phase 8.5 – Rehabilitation-Specific Outputs
//
// 8.5.1 Plasticity Tracking Dashboard
//   Real-time metrics comparing observed neural dynamics against connectome
//   baseline: channel-by-channel correlation deviation, temporal trend of
//   manifold geometry, circuit-level activity summary.
//
// 8.5.2 Assisted Decoding for Impaired Signals
//   When channels drop out, the graph decoder imputes missing activity from
//   connectome-predicted contributions of neighbouring channels.
//
// 8.5.3 Clinician Report Generation (Phase 8.5.3)
//   Session-end summary: duration, task performance, decoder confidence with
//   flagged low-confidence epochs, plasticity indicators with anatomical
//   context, difficulty progression history, next-session recommendations.

#include <reuniclus/Telemetry.hpp>
#include <reuniclus/DifficultyController.hpp>

#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <format>
#include <numeric>
#include <algorithm>

namespace reuniclus {

// ── Plasticity Indicator (Phase 8.5.1) ───────────────────────────────────────
// Computed from the ConnectomeDecoder's drift_matrix after each session.
// Distinguishes electrode drift (inconsistent with connectivity) from
// neuroplasticity (consistent with connectivity but altered firing patterns).
struct PlasticityMetrics {
    double drift_norm;           // Frobenius norm of electrode drift component
    double plasticity_score;     // Norm of topology-consistent changes (0–1)
    std::vector<int> top_channels; // Channels showing the most change
};

// ── Trial Record (per-trial performance for DifficultyController) ────────────
struct TrialRecord {
    int    trial_index;
    double accuracy;
    double engagement;    // Neural engagement proxy (beta power)
    TaskParams params;    // What difficulty was used
    double timestamp_s;
};

// ── Assisted Decoding (Phase 8.5.2) ──────────────────────────────────────────
// Imputes dropped channels from connectome-predicted neighbour contributions.
// @param data         Full channel array (dropped channels are NaN or zero).
// @param is_dropped   Boolean mask of dropped channels.
// @param adj          Connectome adjacency matrix [N×N].
// Returns a corrected channel array with imputed values.
inline std::vector<float> impute_dropped_channels(
    const std::vector<float>& data,
    const std::vector<bool>&  is_dropped,
    const Eigen::MatrixXd&    adj)
{
    std::vector<float> out = data;
    int N = static_cast<int>(data.size());
    for (int i = 0; i < N; ++i) {
        if (!is_dropped[i]) continue;
        // Weighted average of non-dropped neighbours
        double weighted_sum = 0.0, weight_sum = 0.0;
        for (int j = 0; j < N; ++j) {
            if (!is_dropped[j] && adj(i, j) > 0.0) {
                weighted_sum += adj(i, j) * data[j];
                weight_sum   += adj(i, j);
            }
        }
        out[i] = weight_sum > 0.0
            ? static_cast<float>(weighted_sum / weight_sum)
            : 0.0f;
    }
    return out;
}

// ── Session Report (Phase 8.5.3) ─────────────────────────────────────────────
class ClinicalReport {
public:
    struct SessionSummary {
        double   duration_s;
        int      total_trials;
        double   mean_accuracy;
        double   mean_confidence;
        int      low_confidence_epochs;  // Frames with confidence < 0.1
        double   plasticity_score;
        double   mean_latency_us;
        double   p99_latency_us;
        std::string recommended_next_params;
    };

    static SessionSummary summarise(
        const std::vector<TelemetryRecord>& telemetry,
        const std::vector<TrialRecord>&     trials,
        double                              plasticity_score = 0.0)
    {
        SessionSummary s{};
        if (telemetry.empty()) return s;

        // Duration from first to last telemetry record
        s.duration_s = static_cast<double>(
            telemetry.back().wall_time_ns - telemetry.front().wall_time_ns) / 1e9;

        auto stats = Telemetry::compute_stats(telemetry);
        s.mean_latency_us = stats.mean_ns   / 1000.0;
        s.p99_latency_us  = stats.p99_ns    / 1000.0;
        s.mean_confidence = stats.mean_confidence;

        // Low-confidence epoch count
        s.low_confidence_epochs = static_cast<int>(
            std::count_if(telemetry.begin(), telemetry.end(),
                [](const TelemetryRecord& r){ return r.confidence < 0.1f; }));

        if (!trials.empty()) {
            s.total_trials = static_cast<int>(trials.size());
            double acc_sum = 0;
            for (auto& t : trials) acc_sum += t.accuracy;
            s.mean_accuracy = acc_sum / trials.size();
        }

        s.plasticity_score = plasticity_score;

        // Next-session recommendation: based on last trial performance
        if (!trials.empty()) {
            auto& last = trials.back();
            if (last.accuracy > 0.80)
                s.recommended_next_params = "Increase difficulty: reduce target size by 20%";
            else if (last.accuracy < 0.55)
                s.recommended_next_params = "Reduce difficulty: increase target size by 20%";
            else
                s.recommended_next_params = "Maintain current difficulty level";
        }

        return s;
    }

    // ── Write report to file (Phase 8.5.3) ───────────────────────────────────
    static void write(const SessionSummary& s,
                      const std::vector<TrialRecord>& trials,
                      const std::string& path)
    {
        std::ofstream f(path);
        f << "=== Reuniclus Clinical Session Report ===\n\n";
        f << std::format("Session duration        : {:.1f} s\n",  s.duration_s);
        f << std::format("Total trials            : {}\n",        s.total_trials);
        f << std::format("Mean task accuracy      : {:.1f}%\n",   s.mean_accuracy * 100);
        f << std::format("Mean decoder confidence : {:.3f}\n",    s.mean_confidence);
        f << std::format("Low-confidence epochs   : {}\n",        s.low_confidence_epochs);
        f << std::format("Plasticity score        : {:.4f}\n",    s.plasticity_score);
        f << std::format("Mean latency            : {:.1f} µs\n", s.mean_latency_us);
        f << std::format("p99 latency             : {:.1f} µs\n", s.p99_latency_us);
        f << "\nRecommendation:\n  " << s.recommended_next_params << "\n";

        if (!trials.empty()) {
            f << "\nDifficulty Progression:\n";
            f << "trial,accuracy,engagement,target_size,assistance\n";
            for (auto& t : trials) {
                f << std::format("{},{:.3f},{:.3f},{:.2f},{:.2f}\n",
                                 t.trial_index, t.accuracy, t.engagement,
                                 t.params.target_size, t.params.assistance_level);
            }
        }
    }
};

} // namespace reuniclus
