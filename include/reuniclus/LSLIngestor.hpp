#pragma once

// Phase 2 – LSL Ingestor
//
// Wraps liblsl to pull samples from a Lab Streaming Layer inlet and push them
// into the SPSC ring buffer.  A std::jthread provides cooperative cancellation
// via std::stop_token so stop() is clean and allocation-free.
//
// Domain knowledge:
//   liblsl resolve_stream() blocks until a matching LSL stream appears on the
//   network.  In practice, start the LSL source before calling start().

#include <reuniclus/Ingestor.hpp>
#include <reuniclus/NeuralFrame.hpp>
#include <reuniclus/SPSCRingBuffer.hpp>
#include <reuniclus/Logger.hpp>

#include <atomic>
#include <chrono>
#include <thread>
#include <string>
#include <cstddef>

// Forward-declare liblsl types so this header compiles even when liblsl is not
// installed (useful for CI builds that stub the library).
namespace lsl {
    class stream_inlet;
    struct stream_info;
    std::vector<stream_info> resolve_stream(const std::string&, const std::string&, int, double);
}

namespace reuniclus {

class LSLIngestor final : public Ingestor {
public:
    explicit LSLIngestor(SPSCRingBuffer<NeuralFrame, 1024>& buffer,
                         std::string stream_type = "EEG")
        : buffer_(buffer), stream_type_(std::move(stream_type)) {}

    void start() override {
        dropped_ = 0;
        worker_ = std::jthread([this](std::stop_token token) { run(token); });
        LOG_INFO("LSLIngestor started, stream_type={}", stream_type_);
    }

    void stop() override {
        worker_.request_stop();
        if (worker_.joinable()) worker_.join();
        LOG_INFO("LSLIngestor stopped, dropped_frames={}", dropped_.load());
    }

    [[nodiscard]] std::size_t dropped_frames() const noexcept override {
        return dropped_.load(std::memory_order_relaxed);
    }

private:
    void run(std::stop_token token) {
        // Resolve stream – blocks until found or timeout (5 s).
        auto streams = lsl::resolve_stream("type", stream_type_, 1, 5.0);
        if (streams.empty()) {
            LOG_ERROR("LSLIngestor: no '{}' stream found within 5 s", stream_type_);
            return;
        }

        lsl::stream_inlet inlet(streams[0]);
        std::uint32_t seq = 0;

        while (!token.stop_requested()) {
            NeuralFrame frame;
            // pull_sample blocks for up to 0.001 s – keeps the loop responsive
            // to stop requests while never busy-spinning on the LSL side.
            double lsl_ts = inlet.pull_sample(
                frame.channels.data(),
                static_cast<int>(kChannels),
                0.001  // timeout_s
            );
            if (lsl_ts == 0.0) continue;  // Timeout, no sample available

            frame.timestamp = std::chrono::high_resolution_clock::now();
            frame.sequence  = seq++;

            if (!buffer_.try_push(frame)) {
                dropped_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    SPSCRingBuffer<NeuralFrame, 1024>& buffer_;
    std::string                         stream_type_;
    std::jthread                        worker_;
    std::atomic<std::size_t>            dropped_{0};
};

} // namespace reuniclus
