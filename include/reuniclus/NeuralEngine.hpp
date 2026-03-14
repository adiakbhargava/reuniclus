#pragma once

// Phase 4 – Neural Inference Engine (LibTorch / TorchScript)
//
// Architecture: Temporal Convolutional Network (TCN) loaded from a TorchScript
// model file exported via tools/train_tcn.py.
//
// Key design decisions:
//
//   Zero-copy tensor creation (Phase 4.4.1):
//     torch::from_blob wraps existing buffer memory without copying.  The
//     NeuralFrame's channel array is used directly; no intermediate allocation.
//
//   JIT warmup (Phase 4.4.2):
//     The first forward pass triggers kernel selection and JIT compilation,
//     adding 30–50 ms.  20 dummy passes during construction fully warm the
//     JIT cache before real data arrives.
//
//   No-grad inference (Phase 4.4.3):
//     torch::NoGradGuard disables autograd graph construction, reducing latency
//     by ~3× and eliminating gradient storage allocations.
//
// TCN receptive field (Phase 4.2.1):
//   R = 1 + (K−1) × (2^L − 1).  With K=3, L=10: R = 2047 samples ≈ 2 seconds
//   at 1 kHz – sufficient to capture motor imagery temporal dynamics.

#include <reuniclus/NeuralFrame.hpp>
#include <reuniclus/Logger.hpp>

#include <torch/script.h>
#include <torch/torch.h>

#include <string>
#include <vector>
#include <stdexcept>

namespace reuniclus {

class NeuralEngine {
public:
    /// @param model_path  Path to a TorchScript (.pt) file from train_tcn.py.
    /// @param latent_dim  Expected output dimensionality (must match the model).
    /// @param window_size Number of time steps fed per inference.
    explicit NeuralEngine(const std::string& model_path,
                          int                latent_dim  = 16,
                          int                window_size = 1000)
        : latent_dim_(latent_dim)
        , window_size_(window_size)
    {
        try {
            model_ = torch::jit::load(model_path);
        } catch (const c10::Error& e) {
            throw std::runtime_error("NeuralEngine: failed to load model from '"
                                     + model_path + "': " + e.what());
        }

        model_.eval();
        warmup();
        LOG_INFO("NeuralEngine loaded '{}' latent_dim={} window={} warmup=done",
                 model_path, latent_dim_, window_size_);
    }

    // ── Inference ─────────────────────────────────────────────────────────────
    //
    // Takes a pre-processed channel window [n_channels × window_size] stored
    // in a flat float buffer (row-major: channel-first), and returns the
    // latent manifold coordinates as a std::vector<float>.
    [[nodiscard]] std::vector<float> infer(const float* data,
                                           int           n_channels) const {
        // Zero-copy: wrap the caller's buffer.
        // IMPORTANT: data must remain valid for the duration of this call.
        auto tensor = torch::from_blob(
            const_cast<float*>(data),
            {1, n_channels, window_size_},
            torch::kFloat32
        );

        torch::NoGradGuard no_grad;
        auto output = model_.forward({tensor}).toTensor();
        output = output.contiguous();

        const float* ptr = output.data_ptr<float>();
        return {ptr, ptr + output.numel()};
    }

    // Overload accepting a NeuralFrame directly (uses its full channel array).
    [[nodiscard]] std::vector<float> infer(const NeuralFrame& frame) const {
        return infer(frame.channels.data(), static_cast<int>(kChannels));
    }

    [[nodiscard]] int latent_dim()  const noexcept { return latent_dim_; }
    [[nodiscard]] int window_size() const noexcept { return window_size_; }

private:
    void warmup() {
        auto dummy = torch::zeros({1, static_cast<int>(kChannels), window_size_});
        for (int i = 0; i < 20; ++i) {
            torch::NoGradGuard no_grad;
            model_.forward({dummy});
        }
    }

    mutable torch::jit::script::Module model_;
    int                                 latent_dim_;
    int                                 window_size_;
};

} // namespace reuniclus
