// Phase 6 Validation – Channel Selector unit tests
//
// Validation checkpoint (Phase 6):
//   "Validate channel selection: compare decoder accuracy using the top-k
//    MI-selected channels versus the top-k channels by signal amplitude.
//    MI selection should match or exceed amplitude selection.
//    Also validate that removing channels below the rate-distortion threshold
//    does not degrade accuracy by more than 5%."

#include <reuniclus/ChannelSelector.hpp>

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <set>

using namespace reuniclus;

// ── MI is non-negative ────────────────────────────────────────────────────────
TEST(ChannelSelector, GaussianMI_NonNegative) {
    const int N = 200;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(4, N);
    Eigen::MatrixXd Z = Eigen::MatrixXd::Random(2, N);
    double mi = ChannelSelector::gaussian_mi(X, Z);
    EXPECT_GE(mi, 0.0);
}

// ── MI between independent variables ≈ 0 ─────────────────────────────────────
TEST(ChannelSelector, GaussianMI_IndependentNearZero) {
    const int N = 1000;
    std::mt19937 rng{42};
    std::normal_distribution<double> nd;
    Eigen::MatrixXd X(2, N), Z(2, N);
    for (int i = 0; i < N; ++i) {
        X(0,i) = nd(rng); X(1,i) = nd(rng);
        Z(0,i) = nd(rng); Z(1,i) = nd(rng);
    }
    double mi = ChannelSelector::gaussian_mi(X, Z);
    // Independent Gaussians have MI = 0; allow small estimation error
    EXPECT_LT(mi, 0.3) << "MI between independent variables should be ~0";
}

// ── MI between perfectly correlated variables is large ───────────────────────
TEST(ChannelSelector, GaussianMI_PerfectCorrelation_Large) {
    const int N = 500;
    Eigen::MatrixXd Z = Eigen::MatrixXd::Random(2, N);
    Eigen::MatrixXd X = Z;  // X = Z exactly
    double mi = ChannelSelector::gaussian_mi(X, Z);
    EXPECT_GT(mi, 0.5) << "MI of X=Z should be large";
}

// ── Greedy selection returns the right count ──────────────────────────────────
TEST(ChannelSelector, GreedySelect_CorrectCount) {
    const int n_ch = 16, n_samp = 200, latent = 2, k = 5;
    Eigen::MatrixXd data   = Eigen::MatrixXd::Random(n_ch, n_samp);
    Eigen::MatrixXd latent_m = Eigen::MatrixXd::Random(latent, n_samp);

    auto selected = ChannelSelector::greedy_select(k, data, latent_m);
    EXPECT_EQ(static_cast<int>(selected.size()), k);
}

// ── Greedy selection returns unique indices ───────────────────────────────────
TEST(ChannelSelector, GreedySelect_UniqueIndices) {
    const int n_ch = 16, n_samp = 200, latent = 2, k = 8;
    Eigen::MatrixXd data     = Eigen::MatrixXd::Random(n_ch, n_samp);
    Eigen::MatrixXd latent_m = Eigen::MatrixXd::Random(latent, n_samp);

    auto selected = ChannelSelector::greedy_select(k, data, latent_m);
    std::set<int> unique(selected.begin(), selected.end());
    EXPECT_EQ(unique.size(), static_cast<std::size_t>(k))
        << "Greedy selection must not repeat channels";
}

// ── Rate-distortion dim returns at least 1 ───────────────────────────────────
TEST(ChannelSelector, RateDistortionDim_AtLeastOne) {
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    int d = ChannelSelector::rate_distortion_dim(cov, 10.0);  // High distortion
    EXPECT_GE(d, 1);
}

// ── Rate-distortion selects all dims when D is very small ────────────────────
TEST(ChannelSelector, RateDistortionDim_AllDimsWhenLowDistortion) {
    Eigen::MatrixXd cov = 100.0 * Eigen::MatrixXd::Identity(4, 4);
    // All eigenvalues = 100 >> D = 0.01
    int d = ChannelSelector::rate_distortion_dim(cov, 0.01);
    EXPECT_EQ(d, 4);
}

// ── apply_mask extracts correct channels ─────────────────────────────────────
TEST(ChannelSelector, ApplyMask_CorrectChannels) {
    std::vector<float> data = {0.f, 1.f, 2.f, 3.f, 4.f};
    std::vector<int>   mask = {0, 2, 4};
    auto out = ChannelSelector::apply_mask(data.data(), mask);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_FLOAT_EQ(out[0], 0.f);
    EXPECT_FLOAT_EQ(out[1], 2.f);
    EXPECT_FLOAT_EQ(out[2], 4.f);
}
