#pragma once

// Phase 7 – Connectome Graph Construction and Spectral Analysis
//
// Mathematical basis:
//   Adjacency matrix A_ij = total synaptic connectivity between electrode
//   populations i and j, normalised:
//       A_ij /= sqrt(pop_size_i × pop_size_j)
//
//   Graph Laplacian:
//       L = D − A,  D_ii = Σ_j A_ij
//
//   Spectrum L = UΛUᵀ reveals fundamental modes of information flow:
//     λ_0 = 0   (DC mode; corresponds to the uniform vector)
//     λ_1       (algebraic connectivity; governs propagation speed)
//     Larger λ  (increasingly localised patterns)
//
//   Connection to CAR (Phase 3.1):
//     CAR subtracts the DC mode of the *complete* graph Laplacian.
//     The connectome Laplacian generalises this by projecting out
//     the low-frequency modes of the biological wiring topology.
//
// This class is data-only.  The GNN decoder that uses it lives in
// ConnectomeDecoder.hpp.

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace reuniclus {

class ConnectomeGraph {
public:
    explicit ConnectomeGraph(int n_channels)
        : n_(n_channels)
        , A_(Eigen::MatrixXd::Zero(n_channels, n_channels))
    {}

    // ── Build adjacency matrix from synapse list ───────────────────────────
    // Each entry: (pre_channel, post_channel, weight)
    // Represents the aggregate synaptic connectivity between electrode
    // population i and j (see Phase 7.2.1 pseudocode).
    void add_synapse(int pre_ch, int post_ch, double weight) {
        if (pre_ch == post_ch) return;  // Self-loops excluded
        A_(pre_ch, post_ch) += weight;
        A_(post_ch, pre_ch) += weight;  // Symmetrised for undirected graph
    }

    // Normalise by population sizes (Phase 7.2.1):
    //   A_ij /= sqrt(pop_size_i * pop_size_j)
    void normalise(const std::vector<double>& pop_sizes) {
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < n_; ++j) {
                if (i != j) {
                    double denom = std::sqrt(pop_sizes[i] * pop_sizes[j]);
                    if (denom > 0.0) A_(i, j) /= denom;
                }
            }
        }
    }

    // ── Graph Laplacian L = D − A ──────────────────────────────────────────
    [[nodiscard]] Eigen::MatrixXd laplacian() const {
        Eigen::VectorXd degrees = A_.rowwise().sum();
        return degrees.asDiagonal() - A_;
    }

    // ── Spectral decomposition L = UΛUᵀ ───────────────────────────────────
    struct Spectrum {
        Eigen::VectorXd eigenvalues;   // Sorted ascending (λ_0 = 0)
        Eigen::MatrixXd eigenvectors;  // Columns are graph Fourier modes
        double          algebraic_connectivity;  // λ_1
    };

    [[nodiscard]] Spectrum compute_spectrum() const {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(laplacian());
        if (es.info() != Eigen::Success)
            throw std::runtime_error("ConnectomeGraph: eigendecomposition failed");

        Spectrum s;
        s.eigenvalues  = es.eigenvalues();
        s.eigenvectors = es.eigenvectors();
        // λ_0 ≈ 0; algebraic connectivity is λ_1 (the smallest nonzero eigenvalue)
        s.algebraic_connectivity = s.eigenvalues.size() > 1 ? s.eigenvalues(1) : 0.0;
        return s;
    }

    // ── Load from a CSV file (channel_i, channel_j, weight) ────────────────
    static ConnectomeGraph from_csv(const std::string& path, int n_channels) {
        ConnectomeGraph g(n_channels);
        std::ifstream f(path);
        if (!f) throw std::runtime_error("ConnectomeGraph: cannot open " + path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            int i, j; double w;
            char comma;
            if (ss >> i >> comma >> j >> comma >> w)
                g.add_synapse(i, j, w);
        }
        return g;
    }

    // ── Identity graph (fallback when no connectome data is available) ──────
    static ConnectomeGraph identity(int n_channels) {
        ConnectomeGraph g(n_channels);
        // Uniform nearest-neighbour connectivity: connects channel i to i±1
        for (int i = 0; i < n_channels - 1; ++i)
            g.add_synapse(i, i + 1, 1.0);
        std::vector<double> pops(n_channels, 1.0);
        g.normalise(pops);
        return g;
    }

    [[nodiscard]] const Eigen::MatrixXd& adjacency()  const noexcept { return A_; }
    [[nodiscard]] int                    n_channels()  const noexcept { return n_; }

private:
    int             n_;
    Eigen::MatrixXd A_;
};

} // namespace reuniclus
