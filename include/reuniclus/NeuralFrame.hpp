#pragma once

// Phase 2 – NeuralFrame: fundamental data unit in Reuniclus.
//
// Domain knowledge:
//   One NeuralFrame represents one time sample across all electrode channels.
//   For a 256-channel EEG system at 1 kHz, 1,000 NeuralFrames are produced
//   per second.  The struct is plain-old-data (POD) with NO heap allocations
//   so it can be placed in the SPSC ring buffer and copied safely by memcpy.

#include <array>
#include <chrono>
#include <cstdint>

namespace reuniclus {

inline constexpr std::size_t kChannels = 256;

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

struct alignas(64) NeuralFrame {
    std::array<float, kChannels> channels{};  // Voltage readings (µV or mV)
    TimePoint                    timestamp{};
    std::uint32_t                sequence{0}; // Monotonic counter for gap detection
};

static_assert(sizeof(NeuralFrame) > 0, "NeuralFrame must be non-empty");

} // namespace reuniclus
