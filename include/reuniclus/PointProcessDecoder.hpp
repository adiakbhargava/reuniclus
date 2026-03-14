#pragma once

// Phase 5.3 – Point Process Decoder for Spike Trains
//
// Mathematical basis:
//   Each neuron k modelled as an inhomogeneous Poisson process with
//   conditional intensity (exponential link / GLM):
//       λ_k(t | z(t)) = exp(α_k + βₖᵀ z(t))
//   where α_k is baseline log-firing-rate and βₖ is the tuning vector.
//
//   Log-likelihood over interval [t, t+Δt]:
//       log L = Σ_k [ n_k · log λ_k − λ_k · Δt ]
//
//   Filtering:
//       1. Predict: z⁻ ~ N(A·ẑ, P⁻) from KF dynamics.
//       2. Update:  gradient of log-likelihood replaces the linear measurement
//          model, yielding a modified Kalman gain via Laplace approximation.
//
//   Goodness-of-fit:
//       Time-rescaling theorem: rescaled ISIs Λ_k = ∫λ_k ds ~ Exp(1).
//       KS test validates the model.
//
// Usage:
//   PointProcessDecoder decoder(alpha, beta, A, Q, dt);
//   auto estimate = decoder.update(spike_counts);
//   // estimate.mean contains posterior latent state
//
// The decoder shares the same latent manifold as KalmanFilter (Phase 5.1)
// and their posteriors can be fused via Bayesian product of Gaussians.

#include <Eigen/Dense>

#include <vector>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace reuniclus {

class PointProcessDecoder {
public:
    struct Estimate {
        Eigen::VectorXd mean;
        Eigen::MatrixXd cov;
        double          log_likelihood;
    };

    /// @param alpha   Baseline log-firing-rates  [n_neurons]
    /// @param beta    Tuning vectors             [n_neurons × latent_dim]
    /// @param A       State transition matrix    [latent_dim × latent_dim]
    /// @param Q       Process noise covariance   [latent_dim × latent_dim]
    /// @param dt      Timestep in seconds
    PointProcessDecoder(Eigen::VectorXd  alpha,
                        Eigen::MatrixXd  beta,
                        Eigen::MatrixXd  A,
                        Eigen::MatrixXd  Q,
                        double           dt = 1e-3)
        : alpha_(std::move(alpha))
        , beta_(std::move(beta))
        , A_(std::move(A))
        , Q_(std::move(Q))
        , dt_(dt)
        , d_(static_cast<int>(A_.rows()))
        , n_(static_cast<int>(alpha_.size()))
    {
        z_hat_ = Eigen::VectorXd::Zero(d_);
        P_     = Eigen::MatrixXd::Identity(d_, d_);
    }

    // ── One decode step given spike counts (one count per neuron) ─────────────
    Estimate update(const Eigen::VectorXi& spike_counts) {
        // ── Predict (same as Kalman) ──────────────────────────────────────────
        Eigen::VectorXd z_pred = A_ * z_hat_;
        Eigen::MatrixXd P_pred = A_ * P_ * A_.transpose() + Q_;

        // ── Update via Laplace approximation (Newton step on log-posterior) ───
        // Conditional intensities λ_k = exp(α_k + βₖᵀ z)
        Eigen::VectorXd lambda = compute_lambda(z_pred);

        // Gradient of log-likelihood w.r.t. z:  ∇ = Σ_k (n_k - λ_k·Δt) βₖ
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(d_);
        double ll = 0.0;
        for (int k = 0; k < n_; ++k) {
            double nk = spike_counts(k);
            grad += (nk - lambda(k) * dt_) * beta_.row(k).transpose();
            ll   += nk > 0 ? nk * std::log(lambda(k)) - lambda(k) * dt_ : -lambda(k) * dt_;
        }

        // Hessian (Fisher information) = -Σ_k λ_k·Δt · βₖ·βₖᵀ
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(d_, d_);
        for (int k = 0; k < n_; ++k)
            H -= lambda(k) * dt_ * beta_.row(k).transpose() * beta_.row(k);

        // Posterior covariance via Laplace approximation
        Eigen::MatrixXd P_inv = P_pred.inverse() - H;
        P_ = P_inv.inverse();

        // Posterior mean = predictive mean + cov · gradient
        z_hat_ = z_pred + P_ * grad;

        return {z_hat_, P_, ll};
    }

    // ── Time-rescaling goodness of fit ────────────────────────────────────────
    // Returns rescaled ISIs for neuron k.  If the model is correct these should
    // be Exp(1); test with KS test.
    [[nodiscard]] std::vector<double> rescaled_isis(
        int k,
        const std::vector<double>& spike_times,
        const std::vector<Eigen::VectorXd>& z_trajectory,
        double bin_dt) const
    {
        std::vector<double> result;
        if (spike_times.size() < 2) return result;

        for (std::size_t i = 0; i + 1 < spike_times.size(); ++i) {
            // Integrate λ_k over [t_i, t_{i+1}]
            double integral = 0.0;
            int t0 = static_cast<int>(spike_times[i] / bin_dt);
            int t1 = static_cast<int>(spike_times[i+1] / bin_dt);
            for (int t = t0; t < t1; ++t) {
                if (t < static_cast<int>(z_trajectory.size())) {
                    Eigen::VectorXd lam = compute_lambda(z_trajectory[t]);
                    integral += lam(k) * bin_dt;
                }
            }
            result.push_back(integral);
        }
        return result;
    }

    [[nodiscard]] const Eigen::VectorXd& mean()       const noexcept { return z_hat_; }
    [[nodiscard]] const Eigen::MatrixXd& covariance() const noexcept { return P_; }

private:
    [[nodiscard]] Eigen::VectorXd compute_lambda(
        const Eigen::VectorXd& z) const
    {
        return (alpha_ + beta_ * z).array().exp().matrix();
    }

    Eigen::VectorXd alpha_;
    Eigen::MatrixXd beta_;
    Eigen::MatrixXd A_, Q_;
    double          dt_;
    int             d_, n_;
    Eigen::VectorXd z_hat_;
    Eigen::MatrixXd P_;
};

// ── Bayesian Fusion of Two Gaussian Posteriors ────────────────────────────────
// Used to fuse KalmanFilter (continuous signal) + PointProcessDecoder (spikes).
// Product of Gaussians: P_fused⁻¹ = P1⁻¹ + P2⁻¹,  ẑ_fused = P_fused(P1⁻¹·z1 + P2⁻¹·z2)
inline std::pair<Eigen::VectorXd, Eigen::MatrixXd>
fuse_gaussian_posteriors(const Eigen::VectorXd& mean1, const Eigen::MatrixXd& cov1,
                         const Eigen::VectorXd& mean2, const Eigen::MatrixXd& cov2)
{
    Eigen::MatrixXd P1_inv = cov1.inverse();
    Eigen::MatrixXd P2_inv = cov2.inverse();
    Eigen::MatrixXd P_fused = (P1_inv + P2_inv).inverse();
    Eigen::VectorXd z_fused = P_fused * (P1_inv * mean1 + P2_inv * mean2);
    return {z_fused, P_fused};
}

// ── Confidence Gating (Phase 5.4) ────────────────────────────────────────────
struct GatedOutput {
    Eigen::VectorXd command;      // Motor intent estimate (possibly scaled)
    double          confidence;   // Posterior p-value [0, 1]
    bool            is_reliable;  // True when confidence >= threshold
};

inline GatedOutput gate(const KalmanFilter::Estimate& est, double threshold) {
    double conf  = est.confidence;
    double scale = std::min(conf / threshold, 1.0);
    return {est.mean * scale, conf, conf >= threshold};
}

} // namespace reuniclus
