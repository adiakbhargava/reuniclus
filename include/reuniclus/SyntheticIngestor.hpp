#pragma once

// Phase 2 – Synthetic Ingestor (for validation and testing)
//
// Generates a 256-channel, 1 kHz stream of synthetic neural data without any
// hardware or network dependency.  Used for:
//   • Validation checkpoint (Phase 2): sustain 1 kHz × 256 channels zero-drop.
//   • Final validation (Phase 8): 10-minute synthetic pipeline run.
//
// Signal model:
//   Each channel is an independent sum of a low-frequency oscillation (Mu/Beta
//   rhythm at 10 Hz) and zero-mean Gaussian noise – a realistic EEG proxy.

#include <reuniclus/Ingestor.hpp>
#include <reuniclus/NeuralFrame.hpp>
#include <reuniclus/SPSCRingBuffer.hpp>
#include <reuniclus/Logger.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <numbers>
#include <random>
#include <thread>

namespace reuniclus {

class SyntheticIngestor final : public Ingestor {
public:
    /// @param buffer     Destination ring buffer.
    /// @param rate_hz    Simulated sampling rate (default 1000 Hz).
    /// @param noise_std  Gaussian noise standard deviation (µV, default 5.0).
    explicit SyntheticIngestor(SPSCRingBuffer<NeuralFrame, 1024>& buffer,
                               double rate_hz   = 1000.0,
                               float  noise_std = 5.0f)
        : buffer_(buffer), rate_hz_(rate_hz), noise_std_(noise_std) {}

    ~SyntheticIngestor() override { stop(); }

    void start() override {
        dropped_ = 0;
        running_ = true;
        worker_  = std::thread([this] { run(); });
        LOG_INFO("SyntheticIngestor started at {} Hz, {} channels",
                 rate_hz_, kChannels);
    }

    void stop() override {
        running_ = false;
        if (worker_.joinable()) worker_.join();
        LOG_INFO("SyntheticIngestor stopped, dropped={}", dropped_.load());
    }

    [[nodiscard]] std::size_t dropped_frames() const noexcept override {
        return dropped_.load(std::memory_order_relaxed);
    }

private:
    void run() {
        std::mt19937                          rng{std::random_device{}()};
        std::normal_distribution<float>       noise{0.0f, noise_std_};
        const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.0 / rate_hz_));
        auto next_fire = std::chrono::high_resolution_clock::now();
        std::uint32_t seq = 0;
        double t = 0.0;

        while (running_) {
            std::this_thread::sleep_until(next_fire);
            next_fire += period;

            NeuralFrame frame;
            frame.timestamp = std::chrono::high_resolution_clock::now();
            frame.sequence  = seq++;

            // 10 Hz Mu rhythm + noise, phase-shifted per channel
            for (std::size_t ch = 0; ch < kChannels; ++ch) {
                const double phase_offset = static_cast<double>(ch) / kChannels
                                            * 2.0 * std::numbers::pi;
                frame.channels[ch] = 20.0f * static_cast<float>(
                    std::sin(2.0 * std::numbers::pi * 10.0 * t + phase_offset))
                    + noise(rng);
            }
            t += 1.0 / rate_hz_;

            if (!buffer_.try_push(frame)) {
                dropped_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    SPSCRingBuffer<NeuralFrame, 1024>& buffer_;
    double                              rate_hz_;
    float                               noise_std_;
    std::thread                         worker_;
    std::atomic<bool>                   running_{false};
    std::atomic<std::size_t>            dropped_{0};
};

} // namespace reuniclus
