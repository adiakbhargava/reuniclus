// Phase 1 Validation – SPSCRingBuffer unit tests
//
// Validation checkpoint (Phase 1):
//   "Write a Google Benchmark test that measures round-trip latency:
//    producer pushes a timestamped frame, consumer pops it, measures the delta.
//    Target: < 100 ns per operation."
//
// These unit tests verify correctness; benchmarks/bench_spsc.cpp verifies perf.

#include <reuniclus/SPSCRingBuffer.hpp>
#include <reuniclus/NeuralFrame.hpp>

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <numeric>
#include <atomic>

using namespace reuniclus;

// ── Basic push / pop ──────────────────────────────────────────────────────────
TEST(SPSCRingBuffer, PushPopSingle) {
    SPSCRingBuffer<int, 8> buf;
    EXPECT_TRUE(buf.empty());
    EXPECT_TRUE(buf.try_push(42));
    EXPECT_FALSE(buf.empty());
    int val = 0;
    EXPECT_TRUE(buf.try_pop(val));
    EXPECT_EQ(val, 42);
    EXPECT_TRUE(buf.empty());
}

// ── Full buffer rejects new push ──────────────────────────────────────────────
TEST(SPSCRingBuffer, FullReturnsFalse) {
    SPSCRingBuffer<int, 4> buf;
    for (int i = 0; i < 4; ++i) EXPECT_TRUE(buf.try_push(i));
    EXPECT_FALSE(buf.try_push(99));  // Buffer full
}

// ── Empty buffer rejects pop ──────────────────────────────────────────────────
TEST(SPSCRingBuffer, EmptyReturnsFalse) {
    SPSCRingBuffer<int, 4> buf;
    int val = -1;
    EXPECT_FALSE(buf.try_pop(val));
    EXPECT_EQ(val, -1);
}

// ── FIFO ordering preserved ───────────────────────────────────────────────────
TEST(SPSCRingBuffer, FIFOOrdering) {
    SPSCRingBuffer<int, 16> buf;
    for (int i = 0; i < 10; ++i) buf.try_push(i);
    for (int i = 0; i < 10; ++i) {
        int val = -1;
        ASSERT_TRUE(buf.try_pop(val));
        EXPECT_EQ(val, i);
    }
}

// ── Wrap-around: more than N total operations ─────────────────────────────────
TEST(SPSCRingBuffer, WrapAround) {
    SPSCRingBuffer<int, 4> buf;
    for (int round = 0; round < 10; ++round) {
        for (int i = 0; i < 3; ++i) ASSERT_TRUE(buf.try_push(round * 10 + i));
        for (int i = 0; i < 3; ++i) {
            int v; ASSERT_TRUE(buf.try_pop(v));
            EXPECT_EQ(v, round * 10 + i);
        }
    }
}

// ── Power-of-two enforcement ──────────────────────────────────────────────────
// This is a compile-time check via static_assert(N & (N-1) == 0).
// We verify sizes that ARE powers of two compile fine.
TEST(SPSCRingBuffer, PowerOfTwoSizes) {
    SPSCRingBuffer<int, 1>    b1;
    SPSCRingBuffer<int, 2>    b2;
    SPSCRingBuffer<int, 1024> b3;
    EXPECT_EQ(b1.capacity(), 1u);
    EXPECT_EQ(b2.capacity(), 2u);
    EXPECT_EQ(b3.capacity(), 1024u);
}

// ── Producer–Consumer concurrency (correctness, not perf) ────────────────────
TEST(SPSCRingBuffer, ProducerConsumerConcurrent) {
    constexpr int kCount = 100'000;
    SPSCRingBuffer<int, 1024> buf;
    std::atomic<int> consumed{0};

    std::thread producer([&]() {
        for (int i = 0; i < kCount; ++i) {
            while (!buf.try_push(i)) { /* spin */ }
        }
    });

    std::thread consumer([&]() {
        int prev = -1;
        int cnt  = 0;
        while (cnt < kCount) {
            int v;
            if (buf.try_pop(v)) {
                EXPECT_EQ(v, prev + 1);  // Verify FIFO
                prev = v;
                ++cnt;
            }
        }
        consumed.store(cnt);
    });

    producer.join();
    consumer.join();
    EXPECT_EQ(consumed.load(), kCount);
}

// ── NeuralFrame round-trip ────────────────────────────────────────────────────
TEST(SPSCRingBuffer, NeuralFrameRoundTrip) {
    SPSCRingBuffer<NeuralFrame, 1024> buf;
    NeuralFrame frame;
    frame.channels[0]   = 1.23f;
    frame.channels[255] = -4.56f;
    frame.sequence      = 99;

    ASSERT_TRUE(buf.try_push(frame));
    NeuralFrame out;
    ASSERT_TRUE(buf.try_pop(out));
    EXPECT_FLOAT_EQ(out.channels[0],   1.23f);
    EXPECT_FLOAT_EQ(out.channels[255], -4.56f);
    EXPECT_EQ(out.sequence, 99u);
}
