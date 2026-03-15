#pragma once

// Phase 2 – Abstract Ingestor Interface
//
// All data sources (LSL, UDP, file replay, synthetic) implement this interface.
// Downstream code never knows which transport is active, making it trivial to
// swap between lab (LSL), clinical (UDP), and test (synthetic) configurations.

namespace reuniclus {

class Ingestor {
public:
    virtual ~Ingestor() = default;

    /// Begin pulling data into the shared ring buffer.
    virtual void start() = 0;

    /// Signal the worker thread to stop; blocks until the thread exits.
    virtual void stop() = 0;

    /// Number of frames dropped since start() due to a full ring buffer.
    [[nodiscard]] virtual std::size_t dropped_frames() const noexcept = 0;
};

} // namespace reuniclus
