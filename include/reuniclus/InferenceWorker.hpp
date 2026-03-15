#pragma once

// Phase 8 – Asynchronous Neural Inference Worker
//
// Wraps any inference callable (NeuralEngine or ConnectomeDecoder) in a
// dedicated background thread, fully decoupling the real-time processing loop
// from model forward-pass latency.
//
// Design: single-slot "latest-wins" queue.
//   - The processing thread submits windows non-blocking; if a newer window
//     arrives before the previous inference finishes, the old job is replaced.
//   - Results are cached under a lightweight mutex; reads from the processing
//     thread never block.
//
// Lifetime:
//   start() must be called before submit()/latest().
//   stop() is called automatically by the destructor.
//   The InferenceWorker must be destroyed before the engine it wraps.

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace reuniclus {

class InferenceWorker {
public:
    using InferFn = std::function<std::vector<float>(const float*, int)>;

    InferenceWorker(InferFn fn, int latent_dim)
        : fn_(std::move(fn))
        , latest_(latent_dim, 0.0f)
    {}

    ~InferenceWorker() { stop(); }

    // Disable copy/move — owns a thread.
    InferenceWorker(const InferenceWorker&)            = delete;
    InferenceWorker& operator=(const InferenceWorker&) = delete;

    void start() {
        stop_.store(false, std::memory_order_relaxed);
        thread_ = std::thread([this] { run(); });
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(job_mtx_);
            stop_.store(true, std::memory_order_relaxed);
        }
        job_cv_.notify_one();
        if (thread_.joinable()) thread_.join();
    }

    // Non-blocking. Overwrites any pending job with the newest window.
    // Called from the processing thread.
    void submit(const std::vector<float>& window, int n_channels) {
        {
            std::lock_guard<std::mutex> lk(job_mtx_);
            pending_data_  = window;        // copy (processing thread keeps its buffer)
            pending_nch_   = n_channels;
            pending_ready_ = true;
        }
        job_cv_.notify_one();
    }

    // Non-blocking. Returns the latest computed latent vector.
    // Returns zeros until the first inference result is available.
    // Called from the processing thread.
    [[nodiscard]] std::vector<float> latest() const {
        std::lock_guard<std::mutex> lk(result_mtx_);
        return latest_;
    }

    // Non-blocking. Returns {latent, is_new} atomically.
    // is_new is true exactly once per inference result — use to gate
    // Kalman updates so innovations stay independent.
    [[nodiscard]] std::pair<std::vector<float>, bool> latest_with_flag() {
        std::lock_guard<std::mutex> lk(result_mtx_);
        bool was_new  = result_new_;
        result_new_   = false;
        return {latest_, was_new};
    }

private:
    void run() {
        while (true) {
            std::vector<float> data;
            int n_ch = 0;

            {
                std::unique_lock<std::mutex> lk(job_mtx_);
                job_cv_.wait(lk, [this] {
                    return stop_.load(std::memory_order_relaxed) || pending_ready_;
                });
                if (stop_.load(std::memory_order_relaxed) && !pending_ready_) return;
                data           = std::move(pending_data_);
                n_ch           = pending_nch_;
                pending_ready_ = false;
            }

            // Run inference outside the lock — may take hundreds of ms on CPU.
            auto result = fn_(data.data(), n_ch);

            {
                std::lock_guard<std::mutex> lk(result_mtx_);
                latest_     = std::move(result);
                result_new_ = true;
            }
        }
    }

    InferFn fn_;

    mutable std::mutex      job_mtx_;
    std::condition_variable job_cv_;
    std::vector<float>      pending_data_;
    int                     pending_nch_{0};
    bool                    pending_ready_{false};
    std::atomic<bool>       stop_{true};

    mutable std::mutex result_mtx_;
    std::vector<float> latest_;
    bool               result_new_{false};

    std::thread thread_;
};

} // namespace reuniclus
