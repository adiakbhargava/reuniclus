#pragma once

// Phase 5 – Kalman Filter for Neural State Tracking
//
// Mathematical basis:
//   Linear-Gaussian state-space model:
//       z(t+1) = A·z(t) + q(t),   q ~ N(0, Q)   [dynamics]
//       x(t)   = C·z(t) + r(t),   r ~ N(0, R)   [observation]
//
//   Posterior at each step: p(z | x_{1:t}) = N(ẑ, P)
//
//   Predict:
//       ẑ⁻ = A·ẑ
//       P⁻ = A·P·Aᵀ + Q
//
//   Update:
//       S  = C·P⁻·Cᵀ + R           (innovation covariance)
//       K  = P⁻·Cᵀ·S⁻¹             (Kalman gain)
//       ẑ  = ẑ⁻ + K·(x − C·ẑ⁻)    (posterior mean)
//       P  = (I − K·C)·P⁻           (posterior covariance)
//
// Confidence via Mahalanobis distance (Phase 5.1):
//   The innovation e = x − C·ẑ⁻ follows N(0, S) under the model.
//   D² = eᵀ·S⁻¹·e ~ χ²(dim_x).
//   confidence = 1 − CDF_{χ²}(D²) gives a p-value: low confidence when the
//   observation is statistically inconsistent with the model.
//
// Anomaly detection (Phase 5.1 mathematical foundation):
//   Mahalanobis distance exceeding the χ² threshold at chosen α flags
//   electrode malfunctions or patient state transitions.
//
// ReFIT extension (Phase 5.1.2):
//   After each trial, the observation matrix C is re-estimated via maximum
//   likelihood on the most recent N trials, providing online recalibration.
//
// SDE continuous-time formulation (Phase 5.2):
//   Transition matrix A is replaced by the matrix exponential e^{AΔt}
//   (computed via Eigen's MatrixExponential or Padé approximation).

#include <Eigen/Dense>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace reuniclus {

class KalmanFilter {
public:
    struct Estimate {
        Eigen::VectorXd mean;        // Posterior mean  ẑ(t)
        Eigen::MatrixXd cov;         // Posterior covariance P(t)
        double          confidence;  // p-value from Mahalanobis distance [0,1]
        double          mahal_dist;  // Raw Mahalanobis distance (for logging)
    };

    /// @param A          State transition matrix [d × d]
    /// @param C          Observation matrix     [obs_dim × d]
    /// @param Q          Process noise covariance [d × d], positive definite
    /// @param R          Observation noise covariance [obs_dim × obs_dim], PD
    KalmanFilter(Eigen::MatrixXd A,
                 Eigen::MatrixXd C,
                 Eigen::MatrixXd Q,
                 Eigen::MatrixXd R)
        : A_(std::move(A))
        , C_(std::move(C))
        , Q_(std::move(Q))
        , R_(std::move(R))
        , d_(static_cast<int>(A_.rows()))
    {
        if (A_.rows() != A_.cols())
            throw std::invalid_argument("KalmanFilter: A must be square");

        z_hat_ = Eigen::VectorXd::Zero(d_);
        P_     = Eigen::MatrixXd::Identity(d_, d_);
        I_     = Eigen::MatrixXd::Identity(d_, d_);
    }

    // ── One filter step ───────────────────────────────────────────────────────
    Estimate update(const Eigen::VectorXd& observation) {
        // ── Predict ──────────────────────────────────────────────────────────
        Eigen::VectorXd z_pred = A_ * z_hat_;
        Eigen::MatrixXd P_pred = A_ * P_ * A_.transpose() + Q_;

        // ── Update ───────────────────────────────────────────────────────────
        Eigen::MatrixXd S         = C_ * P_pred * C_.transpose() + R_;
        Eigen::MatrixXd K         = P_pred * C_.transpose() * S.inverse();
        Eigen::VectorXd innovation = observation - C_ * z_pred;

        z_hat_ = z_pred + K * innovation;
        P_     = (I_ - K * C_) * P_pred;

        // ── Confidence (Mahalanobis anomaly score) ────────────────────────────
        double maha = innovation.transpose() * S.inverse() * innovation;
        double conf = 1.0 - chi2_cdf(maha, static_cast<int>(observation.size()));

        return {z_hat_, P_, conf, maha};
    }

    // ── Posterior entropy  H = (d/2)log(2πe) + (1/2)log|P| ──────────────────
    // Used in Phase 6.3 online entropy monitoring.
    // log|P| computed via Cholesky for numerical stability.
    [[nodiscard]] double posterior_entropy() const {
        Eigen::LLT<Eigen::MatrixXd> llt(P_);
        if (llt.info() != Eigen::Success) {
            // Fall back to LU if P is numerically non-PD
            return 0.5 * P_.fullPivLu().logAbsDeterminant()
                   + 0.5 * d_ * std::log(2.0 * M_PI * M_E);
        }
        double log_det = 2.0 * llt.matrixL().toDenseMatrix()
                                  .diagonal().array().log().sum();
        return 0.5 * (d_ * std::log(2.0 * M_PI * M_E) + log_det);
    }

    // ── ReFIT Kalman: update observation matrix from new training data ────────
    // Phase 5.1.2: re-estimate C via least-squares over recent N trials.
    void refit(const Eigen::MatrixXd& X,   // [obs_dim × N] recent observations
               const Eigen::MatrixXd& Z) { // [d × N] intended latent states
        // C_new = X · Z† (Moore-Penrose pseudo-inverse)
        C_ = X * Z.transpose() * (Z * Z.transpose()).inverse();
    }

    // ── Accessors ─────────────────────────────────────────────────────────────
    [[nodiscard]] const Eigen::VectorXd& mean()       const noexcept { return z_hat_; }
    [[nodiscard]] const Eigen::MatrixXd& covariance() const noexcept { return P_; }
    [[nodiscard]] int                    latent_dim()  const noexcept { return d_; }

    // ── Matrix exponential for SDE continuous-time formulation (Phase 5.2) ───
    // Replace A with e^{AΔt} when using continuous-time dynamics.
    // Uses Padé approximation via Eigen's unsupported module if available,
    // otherwise eigendecomposition fallback.
    static Eigen::MatrixXd matrix_exponential(const Eigen::MatrixXd& A,
                                              double dt) {
        // For small d (≤ 20) eigendecomposition is fast and numerically stable.
        Eigen::EigenSolver<Eigen::MatrixXd> es(A * dt);
        if (es.info() != Eigen::Success) {
            // Fallback: truncated Taylor series e^X ≈ I + X + X²/2! + X³/3!
            Eigen::MatrixXd X = A * dt;
            Eigen::MatrixXd result = Eigen::MatrixXd::Identity(A.rows(), A.cols());
            Eigen::MatrixXd term   = X;
            double fact = 1.0;
            for (int k = 1; k <= 10; ++k) {
                fact *= k;
                result += term / fact;
                term *= X;
            }
            return result;
        }
        // e^{AΔt} = V · diag(e^{λ_i}) · V⁻¹
        auto V      = es.eigenvectors();
        auto lambda = es.eigenvalues();
        auto expD   = lambda.array().exp().matrix().asDiagonal();
        return (V * expD * V.inverse()).real();
    }

private:
    // Regularised incomplete gamma function approximation for χ² CDF.
    // χ²(x; k) CDF = γ(k/2, x/2) / Γ(k/2).
    // Uses the series expansion from Abramowitz & Stegun 6.5.29.
    static double chi2_cdf(double x, int k) noexcept {
        if (x <= 0.0) return 0.0;
        const double a  = 0.5 * k;
        const double xh = 0.5 * x;
        // Regularised lower incomplete gamma via series: γ(a,x)/Γ(a)
        double sum  = 1.0;
        double term = 1.0;
        for (int i = 1; i < 200; ++i) {
            term *= xh / (a + i);
            sum  += term;
            if (term < 1e-12) break;
        }
        double log_gamma_a = std::lgamma(a);
        return std::exp(-xh + a * std::log(xh) - log_gamma_a) * sum;
    }

    Eigen::MatrixXd A_, C_, Q_, R_;
    Eigen::VectorXd z_hat_;
    Eigen::MatrixXd P_;
    Eigen::MatrixXd I_;
    int             d_;
};

} // namespace reuniclus
