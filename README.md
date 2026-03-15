![reuniclus](https://www.pokemon.com/static-assets/content-assets/cms2/img/pokedex/full/579.png)

# Reuniclus — Real-Time Neural Processing Kernel

A production-grade C++20 brain-computer interface (BCI) pipeline implementing
lock-free concurrency, SIMD-accelerated signal processing, deep-learning-based
neural decoding, Bayesian state estimation, information-theoretic channel
selection, and connectome-guided neural decoding.

---

## Architecture

```
Electrodes
    │
    ▼
[LSL / UDP / Synthetic Ingestor]     ← Thread 0
    │  (NeuralFrame × 1000/s)
    ▼
SPSC Ring Buffer (N=1024, lock-free)
    │
    ▼
[Processing Thread]                  ← Thread 1
    │
    ├─ Phase 3: Signal Processing
    │     ├─ CAR (AVX-512, 16 ch/cycle)
    │     └─ Butterworth Bandpass IIR (AVX-512 cross-channel)
    │
    ├─ Phase 6: Channel Selection (MI-based, greedy submodular)
    │
    ├─ Phase 4: TCN Inference (LibTorch, zero-copy, no-grad)
    │
    ├─ Phase 7: Graph Decoder (Connectome-constrained GNN + TCN)
    │
    ├─ Fuse estimates (weighted average by decoder confidence)
    │
    ├─ Phase 5: Kalman Filter (Bayesian posterior mean + covariance)
    │
    └─ Confidence Gating → Device Output → Telemetry

[Emulator Thread]                    ← Thread 2
    LIF cortical circuit (digital twin from connectome)

[Telemetry Thread]                   ← Thread 3
    Latency histogram, innovation whiteness, calibration plot → CSV
```

---

## Design Rationale

### Lock-Free SPSC Ring Buffer (Phase 1)

**Why lock-free?**
At 1 kHz, a frame arrives every 1 ms. A mutex-based queue can block for
50–100 µs at OS context-switch granularity — consuming 5–10% of the entire
frame budget just waiting. Lock-free guarantees at least one thread always
makes progress, eliminating worst-case blocking.

The SPSC (Single-Producer Single-Consumer) pattern avoids CAS loops entirely:
the producer owns `head_` exclusively, the consumer owns `tail_`. Each is
placed on a separate 64-byte-aligned cache line (`alignas(64)`) to prevent
false sharing — a pathology where writing to `head_` invalidates the consumer's
cache line for `tail_`, causing 2–5× throughput regression.

**Buffer size N = 1024:**
Queuing theory (Kingman's formula) with λ/µ = 0.5 gives overflow probability
`(λ/µ)^N = 0.5^1024 < 10^-308`. Engineering margin is generous; the math
shows N = 20 would suffice at this utilisation.

### SIMD-Accelerated Signal Processing (Phase 3)

**Why AVX-512?**
A scalar loop over 256 channels executes one float operation per cycle. An
AVX-512 instruction processes 16 floats simultaneously. For CAR (two passes
over 256 floats) this means 16 vector ops vs 512 scalar ops — a theoretical
16× improvement, practical 8–12× accounting for memory bandwidth.

**Why Butterworth?**
The Butterworth design places poles equally spaced on a circle in the s-domain,
producing maximally flat magnitude in the passband. Alternatives (Chebyshev,
Elliptic) achieve steeper rolloff but introduce ripple that distorts neural
signal amplitudes, creating phantom features. Amplitude distortion in neural
data leads to incorrect BCI commands.

**CAR as linear projection:**
CAR is not a heuristic — it is projection by the centering matrix
`H = I − (1/N)11ᵀ`. H is idempotent (`H² = H`), symmetric, and has rank N−1.
Understanding CAR as a projection connects it directly to the graph Laplacian
projections used in Phase 7.

### TCN Neural Inference Engine (Phase 4)

**Why TCN over RNN?**
- **Parallelisable training**: convolutions over the entire sequence simultaneously.
- **Stable gradients**: no vanishing/exploding gradient pathology.
- **Controllable receptive field**: `R = 1 + (K−1)(2^L − 1)`. With K=3, L=10:
  R = 2047 samples ≈ 2 s at 1 kHz — tuned to motor imagery temporal dynamics.

**Zero-copy tensor creation:**
`torch::from_blob` wraps the existing `NeuralFrame::channels` buffer without
copying. This eliminates the largest single source of inference latency.

**JIT warmup:**
LibTorch's JIT compiles kernels on the first forward pass (30–50 ms). 20 dummy
passes during `NeuralEngine` construction fully warm the cache.

### Kalman Filter & Bayesian State Estimation (Phase 5)

**Why probabilistic estimation?**
A point estimate of "move left" with no confidence is dangerous in a
rehabilitation BCI. A high-velocity command issued with low confidence can
trigger unintended prosthetic movement. The Kalman filter maintains a full
Gaussian posterior `p(z | x_{1:t}) = N(ẑ, P)`, providing calibrated
uncertainty at every timestep.

**Innovation-based anomaly detection:**
The innovation `e(t) = x − Cẑ⁻` follows `N(0, S)` under the model. The
Mahalanobis distance `eᵀS⁻¹e ~ χ²(dim_x)` provides a principled anomaly
detector: exceeding the χ² threshold flags electrode malfunction, impedance
changes, or patient state transitions — all critical safety events.

**ReFIT recalibration (Phase 5.1.2):**
After each trial, the observation matrix C is re-estimated via maximum
likelihood on recent trials (online EM). This closed-loop recalibration
improves decoding accuracy without requiring explicit calibration tasks.

**SDE continuous-time formulation (Phase 5.2):**
Replaces the discrete transition matrix A with `e^{AΔt}` (matrix exponential
via eigendecomposition), making dynamics model predictions independent of
sampling rate.

### Information-Theoretic Channel Selection (Phase 6)

**Why mutual information over amplitude?**
Signal amplitude selects channels that are loud, not necessarily informative.
MI `I(X; Z) = H(Z) − H(Z|X)` directly measures how many bits each channel
contributes about motor intent.

**Why greedy submodular maximisation?**
Selecting top-k channels by individual MI ignores redundancy. The joint MI
`I(X_S; Z)` is submodular (diminishing returns), so the greedy algorithm
achieves ≥ (1 − 1/e) ≈ 63% of optimal with O(N²) complexity.

**Rate-distortion threshold (Phase 6.2):**
Eigenvalues below the reverse water-filling threshold θ are below the noise
floor; dropping those dimensions does not degrade decoder accuracy > 5%.

### Connectome-Guided Decoding (Phase 7)

**Why connectome data?**
Current data-driven decoders must recalibrate frequently because they cannot
distinguish electrode drift from neuroplasticity — the entire point of
rehabilitation therapy. The connectome encodes structural invariants that change
far more slowly than firing patterns.

**Stability via structural prior:**
During each session's 2-minute resting state, observed cross-channel
correlations are compared to the connectome-predicted structure. Deviations
inconsistent with known wiring are corrected (electrode drift); consistent
deviations are preserved (neuroplasticity).

**Digital twin (Phase 7.4):**
A Leaky Integrate-and-Fire network driven by the connectome adjacency matrix
runs in parallel, producing reference predictions. Agreement confirms decoder
interpretation; divergence flags either channel failure or neuroplastic
reorganisation.

### Optimal Rehabilitation Control (Phase 8.2)

The difficulty controller formulates a Markov Decision Process where:
- **State**: patient's current capability estimate (accuracy + engagement).
- **Action**: task difficulty (target size, movement complexity, assistance).
- **Reward**: maximised at 70% success rate (the "desirable difficulty"
  optimum from the psychology literature on neuroplasticity).
- **Policy**: ε-greedy on a Q-table updated via online Q-learning after each
  trial (~5–10 s), never touching the hot-path latency.

---

## Mathematical Framework

| Component | Key Equation |
|-----------|-------------|
| CAR | `y_i = x_i − (1/N)Σx_j` ≡ `Hx`, H = I − (1/N)11ᵀ |
| IIR Biquad | `y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2] − a1·y[n-1] − a2·y[n-2]` |
| TCN Receptive Field | `R = 1 + (K−1)(2^L − 1)` |
| Graph Convolution | `h_i^{l+1} = σ(Σ_{j∈N(i)} A_ij W^l h_j^l)` |
| Kalman Predict | `ẑ⁻ = Aẑ`, `P⁻ = AP Aᵀ + Q` |
| Kalman Update | `K = P⁻Cᵀ(CP⁻Cᵀ+R)⁻¹`, `ẑ = ẑ⁻ + K(x − Cẑ⁻)` |
| SDE Dynamics | `dz = f(z)dt + σdW`, linear: `z(t+Δt) ~ N(e^{AΔt}z(t), Q_c)` |
| Point Process GLM | `λ_k = exp(α_k + βₖᵀz)`, `log L = Σ_k[n_k log λ_k − λ_k Δt]` |
| Gaussian MI | `I(X;Z) = (1/2)log(det Σ_X det Σ_Z / det Σ_{XZ})` |
| Posterior Entropy | `H = (d/2)log(2πe) + (1/2)log|P|` |
| Graph Laplacian | `L = D − A`, spectrum L = UΛUᵀ |
| Bellman Equation | `V*(s) = max_a[r(s,a) + γΣP(s'|s,a)V*(s')]` |

---

## Build & Validation

### Phase 0 Validation
```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build . --parallel
# Validation: compiles without warnings, LibTorch links, Eigen matrix multiply works.
```

### Phase 1 Benchmark (SPSC < 100 ns)
```bash
./reuniclus_benchmarks --benchmark_filter=BM_SPSC
# Target: BM_SPSC_CrossThread_RoundTrip mean_ns < 100
```

### Phase 2 Validation (1 kHz zero drop)
```bash
# Start synthetic ingestor for 60 s, confirm dropped_frames=0 in log.
```

### Phase 3 Validation (< 1e-5 filter error)
```bash
python tools/compute_filter_coeffs.py  # Generates reference data
# Compare C++ filter output against data/filter_reference.npy
```

### Phase 4 Validation (< 1 ms inference)
```bash
python tools/train_tcn.py --synthetic --output models/tcn_decoder.pt
./reuniclus_benchmarks --benchmark_filter=BM_NeuralEngine
# Target: p99 < 2 ms
```

### Phase 5 Validation (calibration + innovation whiteness)
```bash
./reuniclus_tests --gtest_filter="KalmanFilter*"
# Coverage ≥ 85%, innovation Ljung-Box Q < 31.4
```

### Phase 6 Validation (MI vs amplitude selection)
```bash
./reuniclus_tests --gtest_filter="ChannelSelector*"
```

### Phase 7 Validation (7-day stability)
```bash
python tools/train_graph_decoder.py --synthetic
# Graph decoder degrades < 20% vs TCN-only on shifted data
```

### Phase 8 Final Validation (10-minute run)
```bash
./reuniclus  # Runs 600 s with synthetic stream
# Targets: mean<500µs, p99<1ms, dropped=0, LB Q<31.4
```

---

## Project Structure

```
reuniclus/
  include/reuniclus/
    Version.hpp              # Phase 0: constexpr version string
    Logger.hpp               # Phase 0: lightweight std::format logger
    NeuralFrame.hpp          # Phase 2: POD data unit (256 ch + timestamp)
    SPSCRingBuffer.hpp       # Phase 1: lock-free single-producer ring buffer
    Ingestor.hpp             # Phase 2: abstract data source interface
    LSLIngestor.hpp          # Phase 2: liblsl implementation
    UDPIngestor.hpp          # Phase 2: raw UDP implementation
    SyntheticIngestor.hpp    # Phase 2: synthetic stream for testing
    SignalProcessor.hpp      # Phase 3: AVX-512 CAR + Butterworth IIR
    NeuralEngine.hpp         # Phase 4: LibTorch TCN inference
    KalmanFilter.hpp         # Phase 5: Bayesian state estimator
    PointProcessDecoder.hpp  # Phase 5: spike train GLM decoder + fusion
    ChannelSelector.hpp      # Phase 6: mutual information channel selection
    ConnectomeGraph.hpp      # Phase 7: graph construction + spectral analysis
    LIFNetwork.hpp           # Phase 7: LIF digital twin
    ConnectomeDecoder.hpp    # Phase 7: GNN + TCN decoder + adaptive calibration
    Telemetry.hpp            # Phase 8: lock-free telemetry + Ljung-Box
    DifficultyController.hpp # Phase 8: rehabilitation MDP controller
  src/
    main.cpp                 # Phase 8: 4-thread integration loop
  tools/
    train_tcn.py             # Phase 4: TCN training + TorchScript export
    train_graph_decoder.py   # Phase 7: GNN decoder training + export
    compute_filter_coeffs.py # Phase 3: Butterworth coefficient computation
  tests/
    test_spsc.cpp            # Phase 1: ring buffer correctness + concurrency
    test_signal.cpp          # Phase 3: CAR + bandpass filter validation
    test_kalman.cpp          # Phase 5: Kalman calibration + innovation tests
    test_channel_selector.cpp# Phase 6: MI channel selection tests
  benchmarks/
    bench_spsc.cpp           # Phase 1: round-trip latency (target < 100 ns)
    bench_signal.cpp         # Phase 3: SIMD throughput measurement
  models/                    # Exported TorchScript .pt files
  data/                      # Connectome adjacency matrices, filter references
  third_party/               # googletest, benchmark, liblsl, libtorch, eigen
  CMakeLists.txt
```

---

## Benchmark Targets

| Metric | Target | Phase |
|--------|--------|-------|
| SPSC round-trip latency | < 100 ns | 1 |
| Frame drop rate (1 kHz × 256 ch) | 0 | 2 |
| Bandpass filter error vs Python | < 1e-5 relative | 3 |
| TCN inference mean / p99 | < 1 ms / < 2 ms | 4 |
| Kalman posterior coverage | ≥ 95% | 5 |
| Innovation whiteness (Ljung-Box Q) | < 31.4 (p > 0.05) | 5 |
| Channel selection vs amplitude | MI ≥ amplitude accuracy | 6 |
| Graph decoder 7-day stability | ≥ 20% less degradation vs TCN | 7 |
| End-to-end mean / p99 latency | < 500 µs / < 1 ms | 8 |
| Heap allocations on hot path | 0 | 8 |
| Decoder confidence calibration error | < 5% | 8 |
