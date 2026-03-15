#pragma once

// Phase 1 – Single-Producer Single-Consumer (SPSC) Lock-Free Ring Buffer
//
// Design rationale:
//   Lock-free because mutexes introduce jitter: even a 50 µs OS-level block
//   destroys the latency guarantee of a 1 kHz BCI loop.  The SPSC pattern
//   (exactly one writer, exactly one reader) allows the lock-free proof: the
//   producer owns head_ exclusively and the consumer owns tail_ exclusively,
//   so no compare-and-swap is needed.
//
// Buffer sizing (Phase 1.2 – Queuing Theory):
//   Overflow probability for Poisson arrivals at rate λ and service rate µ:
//       P(overflow) ≈ (λ/µ)^N
//   With λ/µ = 0.5, N = 1024 → P < 10^-308. Generous engineering margin.
//
// False-sharing prevention (Phase 1.1.4):
//   head_ and tail_ are placed on separate 64-byte cache lines via alignas(64).
//   Without this, every producer write invalidates the consumer's cache line
//   for tail_, causing 2–5× throughput regression.

#include <atomic>
#include <array>
#include <cstddef>

namespace reuniclus {

template <typename T, std::size_t N>
class SPSCRingBuffer {
    static_assert((N & (N - 1)) == 0, "N must be a power of 2 (enables O(1) modulo via bitwise AND)");

    // Each atomic placed on its own cache line to prevent false sharing.
    alignas(64) std::atomic<std::size_t> head_{0};  // Written by producer only
    alignas(64) std::atomic<std::size_t> tail_{0};  // Written by consumer only
    alignas(64) std::array<T, N>         buffer_{};

public:
    // ── Producer side ────────────────────────────────────────────────────────
    //
    // Memory ordering rationale:
    //   head_ is loaded relaxed (only the producer reads it here).
    //   tail_ is loaded acquire so any writes made before the consumer updated
    //   tail_ are visible – this prevents reading stale slot data.
    //   head_ is stored release so the written item is visible to the consumer
    //   before head_ advances.
    [[nodiscard]] bool try_push(const T& item) noexcept {
        const std::size_t h = head_.load(std::memory_order_relaxed);
        const std::size_t t = tail_.load(std::memory_order_acquire);
        if (h - t >= N) return false;          // Buffer full
        buffer_[h & (N - 1)] = item;
        head_.store(h + 1, std::memory_order_release);
        return true;
    }

    // ── Consumer side ────────────────────────────────────────────────────────
    [[nodiscard]] bool try_pop(T& item) noexcept {
        const std::size_t t = tail_.load(std::memory_order_relaxed);
        const std::size_t h = head_.load(std::memory_order_acquire);
        if (t >= h) return false;              // Buffer empty
        item = buffer_[t & (N - 1)];
        tail_.store(t + 1, std::memory_order_release);
        return true;
    }

    // ── Observers ────────────────────────────────────────────────────────────
    [[nodiscard]] std::size_t size() const noexcept {
        const std::size_t h = head_.load(std::memory_order_acquire);
        const std::size_t t = tail_.load(std::memory_order_acquire);
        return h - t;
    }

    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] bool full()  const noexcept { return size() >= N; }
    [[nodiscard]] static constexpr std::size_t capacity() noexcept { return N; }
};

} // namespace reuniclus
