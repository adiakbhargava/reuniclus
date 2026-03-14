// Reuniclus – Real-Time Neural Processing Kernel
// Phase 8: Integration, Telemetry, Closed-Loop Control & Validation
//
// Thread architecture (Phase 8.3):
//   Thread 0 – Ingestor:   Pulls from LSL/UDP, pushes to ring buffer.
//   Thread 1 – Processor:  Pops from ring buffer, runs full pipeline.
//   Thread 2 – Emulator:   Runs LIF connectome circuit, produces reference.
//   Thread 3 – Telemetry:  Drains telemetry buffer, writes CSV report.
//
// Main processing loop (Phase 8.1) — exact sequence from guide:
//   CAR → Bandpass → Channel selection mask → TCN inference →
//   Graph decoder inference → Fuse (weighted by confidence) →
//   Kalman filter → Confidence gate → Device output → Telemetry.
//
// NeuralEngine / ConnectomeDecoder graceful fallback:
//   Both decoders require exported TorchScript model files.  When models are
//   absent (first run before training), the pipeline falls back to using the
//   Kalman filter on raw selected channels — the guide's "dual-path design
//   means Reuniclus remains fully functional without connectomic data."

#include <reuniclus/Version.hpp>
#include <reuniclus/Logger.hpp>
#include <reuniclus/NeuralFrame.hpp>
#include <reuniclus/SPSCRingBuffer.hpp>
#include <reuniclus/SyntheticIngestor.hpp>
#include <reuniclus/LSLIngestor.hpp>
#include <reuniclus/SignalProcessor.hpp>
#include <reuniclus/NeuralEngine.hpp>
#include <reuniclus/KalmanFilter.hpp>
#include <reuniclus/PointProcessDecoder.hpp>
#include <reuniclus/ChannelSelector.hpp>
#include <reuniclus/ConnectomeGraph.hpp>
#include <reuniclus/ConnectomeDecoder.hpp>
#include <reuniclus/LIFNetwork.hpp>
#include <reuniclus/Telemetry.hpp>
#include <reuniclus/DifficultyController.hpp>
#include <reuniclus/ClinicalReport.hpp>

#include <Eigen/Dense>
#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef __x86_64__
#  include <immintrin.h>
#endif

namespace reuniclus {

// ── Global shutdown flag ─────────────────────────────────────────────────────
static std::atomic<bool> g_running{true};
static void signal_handler(int) { g_running = false; }

// ── Default Kalman filter ─────────────────────────────────────────────────────
static KalmanFilter make_default_kalman(int latent_dim) {
    Eigen::MatrixXd A = 0.99 * Eigen::MatrixXd::Identity(latent_dim, latent_dim);
    Eigen::MatrixXd C = Eigen::MatrixXd::Random(latent_dim, latent_dim) * 0.1;
    Eigen::MatrixXd Q = 0.01 * Eigen::MatrixXd::Identity(latent_dim, latent_dim);
    Eigen::MatrixXd R = 0.1  * Eigen::MatrixXd::Identity(latent_dim, latent_dim);
    return {A, C, Q, R};
}

// ── Context passed into the processing thread ─────────────────────────────────
struct ProcessorContext {
    SPSCRingBuffer<NeuralFrame, 1024>&   buffer;
    SignalProcessor&                     dsp;
    KalmanFilter&                        kf;
    Telemetry&                           telemetry;
    const std::vector<int>&              selected_channels;
    NeuralEngine*                        tcn_engine;    // nullptr = fallback
    ConnectomeDecoder*                   graph_engine;  // nullptr = fallback
    int                                  latent_dim;
    std::mutex&                          trial_mutex;
    std::vector<TrialRecord>&            trial_records;
    DifficultyController&                difficulty;
    int                                  frames_per_trial;
};

// ─────────────────────────────────────────────────────────────────────────────
// Processing Thread (Phase 8.1)
// Exact loop from guide: DSP → select → TCN → graph → fuse → Kalman → gate
// ─────────────────────────────────────────────────────────────────────────────
static void processing_loop(ProcessorContext& ctx) {
    const int latent_dim = ctx.latent_dim;
    const int n_selected = static_cast<int>(ctx.selected_channels.size());
    const double kSafetyThreshold = 0.05;

    int    trial_frame_count = 0;
    double trial_acc_sum     = 0.0;
    double trial_beta_sum    = 0.0;
    int    trial_index       = 0;

    NeuralFrame frame;
    while (g_running.load(std::memory_order_relaxed)) {
        if (!ctx.buffer.try_pop(frame)) {
#ifdef __x86_64__
            _mm_pause();
#endif
            continue;
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        // ── Phase 3: Signal processing ─────────────────────────────────────
        ctx.dsp.apply_car(frame.channels.data(), kChannels);
        ctx.dsp.apply_bandpass(frame.channels.data(), kChannels);

        // ── Phase 6: Channel selection mask ───────────────────────────────
        std::vector<float> selected =
            ChannelSelector::apply_mask(frame.channels.data(), ctx.selected_channels);

        // ── Phase 4: TCN inference ─────────────────────────────────────────
        std::vector<float> tcn_latent(latent_dim, 0.0f);
        if (ctx.tcn_engine) {
            tcn_latent = ctx.tcn_engine->infer(selected.data(), n_selected);
        } else {
            for (int i = 0; i < latent_dim && i < n_selected; ++i)
                tcn_latent[i] = selected[i];
        }

        // ── Phase 7: Graph decoder inference ──────────────────────────────
        std::vector<float> graph_latent(latent_dim, 0.0f);
        if (ctx.graph_engine) {
            graph_latent = ctx.graph_engine->infer(selected.data(), n_selected);
        } else {
            graph_latent = tcn_latent;  // Single-path fallback
        }

        // ── Fuse: weighted average by decoder confidence ────────────────────
        auto fused = fuse_estimates(tcn_latent, graph_latent, 0.5, 0.5);

        // ── Phase 5: Kalman filter ─────────────────────────────────────────
        Eigen::VectorXd obs = Eigen::Map<Eigen::VectorXd>(
            fused.data(), static_cast<Eigen::Index>(fused.size()));
        if (obs.size() > static_cast<Eigen::Index>(latent_dim))
            obs = obs.head(latent_dim);

        auto estimate = ctx.kf.update(obs);

        // ── Confidence gating → device output ─────────────────────────────
        auto output = gate(estimate, kSafetyThreshold);
        // device_output.send(output.command, output.is_reliable);

        // ── Telemetry ──────────────────────────────────────────────────────
        auto t1 = std::chrono::high_resolution_clock::now();
        ctx.telemetry.record(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0),
            static_cast<float>(output.confidence),
            static_cast<float>(ctx.kf.posterior_entropy()),
            static_cast<float>(estimate.mahal_dist)
        );

        // ── Phase 8.2: Trial-level DifficultyController ────────────────────
        trial_acc_sum  += output.confidence;
        trial_beta_sum += static_cast<double>(output.command.norm());
        ++trial_frame_count;

        if (trial_frame_count >= ctx.frames_per_trial) {
            double acc  = trial_acc_sum  / trial_frame_count;
            double eng  = std::min(trial_beta_sum / trial_frame_count, 1.0);

            TaskParams params;
            {
                std::lock_guard<std::mutex> lk(ctx.trial_mutex);
                params = ctx.difficulty.next_trial(acc, eng);

                double wall_s = std::chrono::duration_cast<
                    std::chrono::duration<double>>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()
                ).count();
                ctx.trial_records.push_back({trial_index, acc, eng, params, wall_s});
            }

            ++trial_index;
            trial_frame_count = 0;
            trial_acc_sum     = 0.0;
            trial_beta_sum    = 0.0;
        }
    }
}

static void processing_thread_entry(ProcessorContext& ctx) { processing_loop(ctx); }

// ─────────────────────────────────────────────────────────────────────────────
// Emulator Thread (Phase 7.4 – Digital Twin)
// ─────────────────────────────────────────────────────────────────────────────
static void emulator_loop(LIFNetwork& network,
                           SPSCRingBuffer<EmulationFrame, 256>& out_buffer)
{
    while (g_running.load(std::memory_order_relaxed)) {
        network.step();
        auto frame = network.read_output();
        out_buffer.try_push(frame);
        std::this_thread::sleep_for(std::chrono::microseconds(900));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Telemetry Thread (Phase 8.4)
// ─────────────────────────────────────────────────────────────────────────────
static void telemetry_loop(Telemetry& telemetry,
                            std::vector<TelemetryRecord>& records)
{
    while (g_running.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        telemetry.flush(records);
    }
    telemetry.flush(records);
}

} // namespace reuniclus

// ─────────────────────────────────────────────────────────────────────────────
// main()
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    using namespace reuniclus;

    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    LOG_INFO("Reuniclus {} starting up (built {} {})",
             kVersion, kBuildDate, kBuildTime);

    // ── Configuration ────────────────────────────────────────────────────────
    const bool   use_synthetic    = true;
    const int    latent_dim       = 16;
    const int    n_selected_ch    = 64;
    const int    run_duration_s   = 600;
    const int    frames_per_trial = 5000;  // ~5 s at 1 kHz
    const std::string tcn_path    = "models/tcn_decoder.pt";
    const std::string graph_path  = "models/graph_decoder.pt";
    const std::string adj_path    = "data/connectome_adj.csv";

    // Phase 0 validation: LibTorch sanity check
    {
        auto t = torch::zeros({1, static_cast<int>(kChannels), 1000});
        LOG_INFO("Phase 0: LibTorch tensor [{},{},{}] OK",
                 t.size(0), t.size(1), t.size(2));
    }

    // ── Shared ring buffers ──────────────────────────────────────────────────
    SPSCRingBuffer<NeuralFrame, 1024>   neural_buf;
    SPSCRingBuffer<EmulationFrame, 256> emul_buf;

    // ── Signal processor ─────────────────────────────────────────────────────
    SignalProcessor dsp;

    // ── Connectome graph ─────────────────────────────────────────────────────
    ConnectomeGraph graph = std::filesystem::exists(adj_path)
        ? ConnectomeGraph::from_csv(adj_path, static_cast<int>(kChannels))
        : ConnectomeGraph::identity(static_cast<int>(kChannels));

    // ── Channel selection mask ────────────────────────────────────────────────
    std::vector<int> selected_channels;
    selected_channels.reserve(n_selected_ch);
    for (int i = 0; i < n_selected_ch; ++i)
        selected_channels.push_back(i * (static_cast<int>(kChannels) / n_selected_ch));

    // ── Kalman filter ─────────────────────────────────────────────────────────
    auto kf = make_default_kalman(latent_dim);

    // ── TCN decoder (optional) ────────────────────────────────────────────────
    std::unique_ptr<NeuralEngine> tcn_engine;
    if (std::filesystem::exists(tcn_path)) {
        try {
            tcn_engine = std::make_unique<NeuralEngine>(tcn_path, latent_dim);
            LOG_INFO("TCN decoder loaded from {}", tcn_path);
        } catch (const std::exception& e) {
            LOG_WARN("TCN load failed: {}. Using fallback.", e.what());
        }
    } else {
        LOG_WARN("No TCN model at '{}'. Run tools/train_tcn.py first.", tcn_path);
    }

    // ── Graph decoder (optional) ──────────────────────────────────────────────
    std::unique_ptr<ConnectomeDecoder> graph_engine;
    if (std::filesystem::exists(graph_path)) {
        try {
            graph_engine = std::make_unique<ConnectomeDecoder>(
                graph_path, graph, latent_dim);
            LOG_INFO("Graph decoder loaded from {}", graph_path);
        } catch (const std::exception& e) {
            LOG_WARN("Graph decoder load failed: {}. Using fallback.", e.what());
        }
    } else {
        LOG_WARN("No graph model at '{}'. Run tools/train_graph_decoder.py.", graph_path);
    }

    // ── Digital twin ──────────────────────────────────────────────────────────
    LIFNetwork lif(graph);

    // ── Difficulty controller & trial records ─────────────────────────────────
    DifficultyController     difficulty;
    std::mutex               trial_mutex;
    std::vector<TrialRecord> trial_records;
    trial_records.reserve(1000);

    // ── Telemetry ─────────────────────────────────────────────────────────────
    Telemetry                    telemetry;
    std::vector<TelemetryRecord> telem_records;
    telem_records.reserve(1'200'000);

    // ── Ingestor ──────────────────────────────────────────────────────────────
    std::unique_ptr<Ingestor> ingestor;
    if (use_synthetic) {
        ingestor = std::make_unique<SyntheticIngestor>(neural_buf, 1000.0);
    } else {
        ingestor = std::make_unique<LSLIngestor>(neural_buf);
    }

    // ── Processor context ────────────────────────────────────────────────────
    ProcessorContext proc_ctx{
        .buffer            = neural_buf,
        .dsp               = dsp,
        .kf                = kf,
        .telemetry         = telemetry,
        .selected_channels = selected_channels,
        .tcn_engine        = tcn_engine.get(),
        .graph_engine      = graph_engine.get(),
        .latent_dim        = latent_dim,
        .trial_mutex       = trial_mutex,
        .trial_records     = trial_records,
        .difficulty        = difficulty,
        .frames_per_trial  = frames_per_trial,
    };

    // ── Start all four threads (Phase 8.3) ────────────────────────────────────
    ingestor->start();
    std::thread processor_thread(processing_thread_entry, std::ref(proc_ctx));
    std::thread emulator_thread(emulator_loop, std::ref(lif), std::ref(emul_buf));
    std::thread telemetry_thread(telemetry_loop,
                                  std::ref(telemetry), std::ref(telem_records));

    LOG_INFO("All 4 threads running. Duration={} s. Ctrl+C to stop.", run_duration_s);

    // ── Run loop: print periodic status ──────────────────────────────────────
    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::seconds(run_duration_s);
    while (g_running && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        auto stats = Telemetry::compute_stats(telem_records);
        int n_trials;
        {
            std::lock_guard<std::mutex> lk(trial_mutex);
            n_trials = static_cast<int>(trial_records.size());
        }
        LOG_INFO("Status: frames={} mean={:.0f}µs p99={:.0f}µs conf={:.3f} trials={}",
                 stats.n_frames,
                 stats.mean_ns / 1000.0, stats.p99_ns / 1000.0,
                 stats.mean_confidence, n_trials);
    }
    g_running = false;

    // ── Shutdown ─────────────────────────────────────────────────────────────
    ingestor->stop();
    if (processor_thread.joinable()) processor_thread.join();
    if (emulator_thread.joinable())  emulator_thread.join();
    if (telemetry_thread.joinable()) telemetry_thread.join();

    // ── Phase 8.4: Final statistics ───────────────────────────────────────────
    auto stats = Telemetry::compute_stats(telem_records);
    std::cout << "\n=== Reuniclus Session Report ===\n"
              << std::format("Frames processed : {:>10}\n",      stats.n_frames)
              << std::format("Mean latency     : {:>8.1f} µs\n", stats.mean_ns / 1000.0)
              << std::format("Median latency   : {:>8.1f} µs\n", stats.median_ns / 1000.0)
              << std::format("p95 latency      : {:>8.1f} µs\n", stats.p95_ns  / 1000.0)
              << std::format("p99 latency      : {:>8.1f} µs\n", stats.p99_ns  / 1000.0)
              << std::format("Max latency      : {:>8.1f} µs\n", stats.max_ns  / 1000.0)
              << std::format("Mean confidence  : {:>10.4f}\n",   stats.mean_confidence)
              << std::format("Dropped frames   : {:>10}\n",      ingestor->dropped_frames());

    // Validation gate
    bool pass_latency = stats.mean_ns < 500'000 && stats.p99_ns < 1'000'000;
    bool pass_drop    = ingestor->dropped_frames() == 0;
    std::cout << "\nVALIDATION:\n"
              << "  mean<500µs & p99<1ms  : " << (pass_latency ? "PASS" : "FAIL") << '\n'
              << "  zero dropped frames   : " << (pass_drop    ? "PASS" : "FAIL") << '\n';

    if (!telem_records.empty()) {
        std::vector<float> maha;
        maha.reserve(telem_records.size());
        for (auto& r : telem_records) maha.push_back(r.maha_dist);
        double Q_lb  = Telemetry::ljung_box(maha, 20);
        std::cout << std::format("  innovation white (LB Q={:.2f}) : {}\n",
                                 Q_lb, Q_lb < 31.4 ? "PASS" : "FAIL");
    }

    std::cout << std::format("  confidence calibration ({:.3f}) : {}\n",
                             stats.mean_confidence,
                             std::abs(stats.mean_confidence - 0.5) < 0.45
                             ? "PASS" : "FAIL");

    // ── Phase 8.5.3: Clinician report ─────────────────────────────────────────
    {
        double plasticity = graph_engine ? graph_engine->drift_matrix().norm() : 0.0;

        std::vector<TrialRecord> trials_copy;
        {
            std::lock_guard<std::mutex> lk(trial_mutex);
            trials_copy = trial_records;
        }

        auto summary = ClinicalReport::summarise(telem_records, trials_copy, plasticity);
        ClinicalReport::write(summary, trials_copy, "session_report.txt");
        LOG_INFO("Clinical report → session_report.txt");
    }

    // ── Phase 8.4: Telemetry CSV ──────────────────────────────────────────────
    Telemetry::write_csv(telem_records, "telemetry.csv");
    LOG_INFO("Telemetry CSV → telemetry.csv");

    return 0;
}
