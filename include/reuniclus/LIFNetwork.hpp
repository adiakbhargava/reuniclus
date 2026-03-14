#pragma once

// Phase 7.4 – Leaky Integrate-and-Fire (LIF) Network (Digital Twin)
//
// The digital twin runs a simplified cortical circuit emulation derived from
// connectome data.  Its role in the pipeline (Phase 7.4):
//
//   Agreement with live decoder:        High confidence in decoded intent.
//   Systematic divergence:              Fill in missing/corrupted channels via
//                                       biologically-grounded interpolation.
//   Spatially coherent divergence:      Possible neuroplasticity; log for clinician.
//
// Neuron model – Leaky Integrate-and-Fire (LIF):
//   τ dV/dt = −(V − V_rest) + R·I(t)
//   Discrete-time Euler: V(t+dt) = V(t) + dt/τ · (−(V(t)−V_rest) + R·I(t))
//   Fire when V ≥ V_thresh; reset to V_reset.
//
// This is the same model class used in Eon Systems' fly emulation (proven to
// generate complex naturalistic behaviour from connectome topology alone).

#include <reuniclus/ConnectomeGraph.hpp>
#include <reuniclus/SPSCRingBuffer.hpp>

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <random>

namespace reuniclus {

struct EmulationFrame {
    Eigen::VectorXd channel_activations;  // Mean firing rate per electrode channel
};

class LIFNetwork {
public:
    struct LIFParams {
        double tau_m     = 20e-3;   // Membrane time constant (s)
        double V_rest    = -70.0;   // Resting potential (mV)
        double V_thresh  = -50.0;   // Spike threshold (mV)
        double V_reset   = -75.0;   // Post-spike reset (mV)
        double R_m       = 10.0;    // Membrane resistance (MΩ)
        double dt        = 1e-3;    // Integration timestep (s)
        double noise_std = 0.5;     // Synaptic noise std (mV/step)
    };

    explicit LIFNetwork(const ConnectomeGraph& graph,
                        LIFParams params = {})
        : graph_(graph)
        , p_(params)
        , n_ch_(graph.n_channels())
        , V_(Eigen::VectorXd::Constant(n_ch_, params.V_rest))
        , spike_count_(Eigen::VectorXd::Zero(n_ch_))
        , rng_(std::random_device{}())
        , noise_(0.0, params.noise_std)
    {}

    // ── Apply external input current (e.g. task cue) ──────────────────────
    void set_input(const Eigen::VectorXd& I_ext) { I_ext_ = I_ext; }

    // ── Advance one timestep ──────────────────────────────────────────────
    void step() {
        const Eigen::MatrixXd& A = graph_.adjacency();
        const double dt_tau = p_.dt / p_.tau_m;

        // Recurrent synaptic input from spikes in previous step
        Eigen::VectorXd I_syn = A * spike_this_step_;

        for (int i = 0; i < n_ch_; ++i) {
            double I = (I_ext_.size() > 0 ? I_ext_(i) : 0.0)
                     + I_syn(i)
                     + noise_(rng_);

            V_(i) += dt_tau * (-(V_(i) - p_.V_rest) + p_.R_m * I);

            if (V_(i) >= p_.V_thresh) {
                spike_this_step_(i) = 1.0;
                spike_count_(i)    += 1.0;
                V_(i)               = p_.V_reset;
            } else {
                spike_this_step_(i) = 0.0;
            }
        }
        ++step_count_;
    }

    // ── Read instantaneous firing rate estimate ────────────────────────────
    // Returns mean spike rate per channel over the simulation so far.
    [[nodiscard]] EmulationFrame read_output() const {
        EmulationFrame f;
        double elapsed = step_count_ * p_.dt;
        f.channel_activations = elapsed > 0.0
            ? spike_count_ / elapsed
            : Eigen::VectorXd::Zero(n_ch_);
        return f;
    }

    // Reset spike accumulators (call at start of each trial window)
    void reset_accumulators() {
        spike_count_.setZero();
        step_count_ = 0;
    }

    [[nodiscard]] int n_channels() const noexcept { return n_ch_; }

private:
    const ConnectomeGraph& graph_;
    LIFParams              p_;
    int                    n_ch_;
    Eigen::VectorXd        V_;               // Membrane potentials
    Eigen::VectorXd        spike_this_step_; // 1 if fired this step
    Eigen::VectorXd        spike_count_;     // Cumulative spikes per channel
    Eigen::VectorXd        I_ext_;           // External input
    std::mt19937           rng_;
    std::normal_distribution<double> noise_;
    int                    step_count_{0};
};

} // namespace reuniclus
