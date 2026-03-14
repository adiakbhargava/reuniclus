#pragma once

// Phase 8.2 – Optimal Rehabilitation Difficulty Controller
//
// Mathematical basis – the Rehabilitation MDP (Phase 8.2):
//   State  s(t): patient's current neural capability (accuracy + engagement).
//   Action a(t): task difficulty parameters (target size, movement complexity,
//                assistance level).
//   Reward r(s,a): balanced to keep task success rate in the 60–80% range
//                  (optimal learning "desirable difficulty" – psychology lit.)
//                  plus a neural engagement term.
//   Policy: ε-greedy on a Q-table updated via fitted Q-iteration after each
//           trial.
//
// Bellman equation:
//   V*(s) = max_a [ r(s,a) + γ Σ_{s'} P(s'|s,a) V*(s') ]
//
// Connection to quant finance:
//   Identical to optimal execution: adjust speed (difficulty) to maximise
//   neuroplastic impact while maintaining patient engagement.  The Q-table
//   update rule is identical to Deep Q-Network target computation.
//
// Timescale: operates at trial granularity (5–10 s), never on the hot path.

#include <Eigen/Dense>
#include <reuniclus/Logger.hpp>

#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>
#include <fstream>

namespace reuniclus {

struct TaskParams {
    double target_size;        // Relative target size (1.0 = normal)
    double movement_complexity; // 1.0 = simple, 2.0 = complex
    double assistance_level;   // 0.0 = no assist, 1.0 = full assist
};

class DifficultyController {
public:
    static constexpr int kStateResolution = 10;  // Discretisation per dimension
    static constexpr int kNumActions      =  5;  // Difficulty levels 0-4
    static constexpr double kGamma        = 0.95; // Discount factor
    static constexpr double kEpsilon0     = 0.2;  // Initial exploration rate

    DifficultyController()
        : Q_(kStateResolution * kStateResolution, Eigen::VectorXd::Zero(kNumActions))
        , rng_(std::random_device{}())
        , unif_(0.0, 1.0)
    {
        state_ = Eigen::Vector2d(0.5, 0.5);  // [accuracy_capability, engagement]
    }

    // ── Called at the end of each trial (5–10 second window) ──────────────
    TaskParams next_trial(double trial_accuracy, double trial_engagement) {
        // Update capability state from performance
        state_ = update_capability(state_, trial_accuracy, trial_engagement);

        // Compute reward: penalise both too-easy (>80%) and too-hard (<60%)
        double reward = reward_fn(trial_accuracy, trial_engagement);

        // Q-learning update on previous transition
        if (last_state_idx_ >= 0) {
            int s_idx = discretize(last_state_);
            int s_next = discretize(state_);
            double best_next = Q_[s_next].maxCoeff();
            double td_error = reward + kGamma * best_next
                              - Q_[s_idx](last_action_);
            Q_[s_idx](last_action_) += kAlpha_ * td_error;
        }

        // Select action: ε-greedy
        int s_idx = discretize(state_);
        int action;
        double eps = kEpsilon0 * std::exp(-episode_count_ * 0.01);
        if (unif_(rng_) < eps) {
            action = std::uniform_int_distribution<int>{0, kNumActions-1}(rng_);
        } else {
            Q_[s_idx].maxCoeff(&action);
        }

        last_state_     = state_;
        last_state_idx_ = s_idx;
        last_action_    = action;
        ++episode_count_;

        LOG_INFO("DifficultyController: acc={:.2f} eng={:.2f} action={} reward={:.3f}",
                 trial_accuracy, trial_engagement, action, reward);

        return action_to_params(action);
    }

    // ── Export Q-table for offline analysis ────────────────────────────────
    void save_q_table(const std::string& path) const {
        std::ofstream f(path);
        for (const auto& row : Q_) {
            for (int a = 0; a < kNumActions; ++a)
                f << row(a) << (a < kNumActions-1 ? ',' : '\n');
        }
    }

private:
    // State update: exponential moving average of recent trial performance
    static Eigen::Vector2d update_capability(const Eigen::Vector2d& s,
                                              double acc, double eng) {
        constexpr double alpha = 0.3;
        return {(1 - alpha) * s(0) + alpha * acc,
                (1 - alpha) * s(1) + alpha * eng};
    }

    // Reward: maximum at 70% accuracy, penalise extremes.
    // Inspired by the "desirable difficulty" literature (Bjork 1994).
    static double reward_fn(double acc, double eng) {
        double acc_reward  = -4.0 * (acc - 0.70) * (acc - 0.70) + 1.0;
        double eng_bonus   = 0.3 * eng;
        return std::max(0.0, acc_reward) + eng_bonus;
    }

    int discretize(const Eigen::Vector2d& s) const {
        int i = static_cast<int>(std::clamp(s(0), 0.0, 0.9999) * kStateResolution);
        int j = static_cast<int>(std::clamp(s(1), 0.0, 0.9999) * kStateResolution);
        return i * kStateResolution + j;
    }

    static TaskParams action_to_params(int action) {
        // Action 0 = easiest, 4 = hardest
        static constexpr double sizes[5]       = {2.0, 1.5, 1.0, 0.75, 0.5};
        static constexpr double complexity[5]  = {1.0, 1.2, 1.5, 1.8,  2.0};
        static constexpr double assistance[5]  = {0.8, 0.6, 0.4, 0.2,  0.0};
        return {sizes[action], complexity[action], assistance[action]};
    }

    std::vector<Eigen::VectorXd> Q_;             // Q-table [state × action]
    Eigen::Vector2d              state_;         // [accuracy_cap, engagement]
    Eigen::Vector2d              last_state_;
    int                          last_state_idx_{-1};
    int                          last_action_{0};
    int                          episode_count_{0};
    static constexpr double      kAlpha_{0.1};  // Learning rate
    std::mt19937                 rng_;
    std::uniform_real_distribution<double> unif_;
};

} // namespace reuniclus
