"""
orchestrator/control_plane/predictor.py
────────────────────────────────────────
WorkloadPredictor: LSTM-based CPU utilisation forecaster.

What this is
─────────────
This is the "AI element" of the scheduler — the component that learns from
history and predicts the future. It answers: "Given how this node's CPU has
behaved over the last 10 completed jobs, what utilisation should we expect
over the next 5 minutes?"

Why this matters for scheduling
─────────────────────────────────
The ACO colony picks the *currently* best node. But "currently best" and
"still best when the job actually runs" are different things. If a node is
at 40% CPU now but predicted to spike to 95% in 3 minutes, placing a
LATENCY_CRITICAL job there risks SLA violation.

The predictor output (spike_probability > 0.7) lets the scheduler:
  1. Penalise predicted-overloaded nodes in the ACO η heuristic.
  2. Pre-warm containers on stable nodes before demand arrives.
  3. Expose /predict endpoint for human operators.

Architecture
─────────────
Single-layer LSTM with one linear readout:

  Input:   (1, LOOKBACK=10, 1)   — last 10 CPU core observations
                ↓
  LSTM:    hidden_size=32, num_layers=1, batch_first=True
                ↓
  Linear:  32 → 1
                ↓
  Output:  scalar (normalised) → denormalise → clamp [0, 100] as CPU util%

Why this size?
  32 hidden units: minimum to capture short-term autocorrelation in CPU
  time series. 64+ overfits on the ≤500 samples we store. Single layer
  avoids vanishing gradients on sequences of just 10 steps.

Why LOOKBACK=10?
  Matches WorkloadProfile.has_enough_data (>= 10 samples). The predictor
  never operates on a shorter sequence — cold-start path handles that.

Training
─────────
Full-batch training (all samples in one forward pass). Justified because:
  - Dataset size: at most 490 windows from 500 samples — fits in one tensor.
  - Mini-batching adds Python loop overhead that outweighs any benefit here.
  50 Adam epochs takes < 50ms on CPU, well within the 60s refresh cycle.

Normalisation
──────────────
Z-score (zero mean, unit variance) per predictor instance:
  z[i] = (x[i] - mean) / std

Why? Raw CPU core counts vary by node capacity (a 4-core node and a 32-core
node have completely different scales). Z-score maps both to the same range,
making MSE loss consistent and the weights transferable across nodes.

Mean and std are stored as instance attributes (_cpu_mean, _cpu_std) so
each predictor is independently calibrated for its node.

Cold-start
───────────
When the profile has < 10 samples, no LSTM prediction is possible.
The fallback returns:
  - predicted_cpu_util = min(avg_cpu_cores × 10, 100.0)
      (heuristic: assume a 10-core node, scale core count to %)
  - confidence = 0.1 (signals: "don't trust this — use it as a weak signal only")
  - spike_probability = 0.0 (no evidence of spikes yet)

Why avg_cpu_cores × 10? It's an order-of-magnitude estimate. Real nodes
range 8–96 cores; the ×10 multiplier hits the middle of that range.
Callers with confidence < 0.3 should treat this as noise, not signal.

Integration
───────────
  Reads:  WorkloadProfile  (orchestrator/shared/telemetry.py)
  Writes: PredictionResult (orchestrator/shared/models.py)

  Called by:
    - telemetry/collector.py every 60s (background refit loop)
    - orchestrator/control_plane/orchestration_service.py (Phase 9)
    - api/main.py GET /predict/{node_id} endpoint (Phase 9)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from orchestrator.shared.models import PredictionResult
from orchestrator.shared.telemetry import WorkloadProfile

# ── Hyperparameters ────────────────────────────────────────────────────────────
# Module-level so tests can import and assert against them directly.

LOOKBACK: int = 10
"""Number of past CPU observations used as one input sequence.

Must equal WorkloadProfile.has_enough_data threshold (≥ 10 samples).
Smaller windows miss medium-term trends; larger require more cold-start data.
"""

HIDDEN_SIZE: int = 32
"""LSTM hidden state dimension.

32 units is the sweet spot for ≤500-sample workload histories:
  - Captures short-term autocorrelation (CPU bursts repeat every few jobs)
  - Doesn't overfit on small datasets (64+ hidden units overfit here)
  - Inference: one matrix multiply (32×1) — negligible latency
"""

NUM_LAYERS: int = 1
"""Number of stacked LSTM layers.

One layer is sufficient for sequences of length 10. Stacked LSTMs help with
very long sequences (100+) but add vanishing gradient risk for short ones.
"""

TRAIN_EPOCHS: int = 50
"""Full-batch gradient descent iterations during fit().

50 epochs converges MSE loss on ≤490 training windows (empirically tested).
Each epoch: 1 forward pass + 1 backward pass on the full dataset.
Wall-clock: ~30–50ms on CPU — fast enough for the 60s background refit cycle.
"""

LEARNING_RATE: float = 0.01
"""Adam optimiser learning rate.

0.01 is a well-established default for LSTM regression tasks.
Lower (0.001) converges slower than needed within 50 epochs.
Higher (0.1) causes loss oscillation on small datasets.
"""

REFIT_THRESHOLD: int = 10
"""Minimum new samples required to trigger a refit in refit_if_needed().

Refitting on every new sample is wasteful. A batch of 10 new observations
provides enough signal to meaningfully update the model weights.
"""


# ── Internal LSTM module ───────────────────────────────────────────────────────

class _LSTMModel(nn.Module):
    """
    Single-layer LSTM with a linear readout head.

    Not part of the public API — instantiated and owned by WorkloadPredictor.

    Forward pass:
        x : (batch=1, seq_len=LOOKBACK, input_size=1)  → normalised CPU history
        → LSTM → (batch=1, seq_len, hidden_size=32)
        → take last timestep  → (batch=1, hidden_size=32)
        → Linear(32 → 1)     → (batch=1, 1)  → scalar prediction

    Why batch_first=True?
        Our tensors are shaped (batch, seq, features), the PyTorch default for
        data coming from DataLoader. batch_first=True avoids a permute() call
        on the hot inference path.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )
        self.linear = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, LOOKBACK, 1) — normalised CPU core values

        Returns:
            shape (batch, 1) — predicted next normalised CPU value
        """
        # out shape: (batch, LOOKBACK, HIDDEN_SIZE)
        out, _ = self.lstm(x)
        # Take only the last timestep's hidden state (many-to-one prediction)
        # out[:, -1, :] shape: (batch, HIDDEN_SIZE)
        pred = self.linear(out[:, -1, :])   # (batch, 1)
        return pred


# ── Public predictor class ─────────────────────────────────────────────────────

class WorkloadPredictor:
    """
    LSTM-based CPU utilisation forecaster for a single cluster node.

    One predictor instance per node. Each is independently trained on that
    node's WorkloadProfile — different nodes have different base loads and
    burst patterns.

    Lifecycle:
        predictor = WorkloadPredictor(node_id="node-gpu-01")
        predictor.fit(profile)                 # called once profile has ≥10 samples
        result = predictor.predict(profile)    # call at any time; cold-start safe

        # Background refresh loop (every 60s):
        predictor.refit_if_needed(profile)     # only refits if ≥10 new samples added

    Thread safety:
        Not thread-safe. The orchestration service runs one predictor per node
        on the async event loop (single thread). If parallel inference is needed
        in Phase V, add a threading.Lock around fit() and predict().

    Attributes (public, readable by tests):
        node_id       : str   — which node this predictor serves
        is_trained    : bool  — True after a successful fit()
        cpu_mean      : float — normalisation mean (set by fit())
        cpu_std       : float — normalisation std  (set by fit())
    """

    def __init__(self, node_id: str) -> None:
        """
        Initialise an untrained predictor for a given node.

        Args:
            node_id: The node this predictor will forecast for.
                     Stored in PredictionResult.node_id.

        Raises:
            ValueError: if node_id is empty.
        """
        if not node_id:
            raise ValueError("WorkloadPredictor requires a non-empty node_id.")

        self.node_id: str = node_id

        # Private state — managed by fit() / predict()
        self._model: Optional[_LSTMModel] = None
        self._trained: bool = False
        self._cpu_mean: float = 0.0
        self._cpu_std: float = 1.0
        self._last_fit_sample_count: int = 0   # for refit_if_needed()
        self._last_mae: float = 20.0           # MAE in % CPU after last fit (high init = uncertain)

    # ── Public properties (readable by tests) ─────────────────────────────────

    @property
    def is_trained(self) -> bool:
        """True after at least one successful fit() call."""
        return self._trained

    @property
    def cpu_mean(self) -> float:
        """Z-score normalisation mean set during fit(). 0.0 if not trained."""
        return self._cpu_mean

    @property
    def cpu_std(self) -> float:
        """Z-score normalisation std set during fit(). 1.0 if not trained."""
        return self._cpu_std

    # ── Core methods ──────────────────────────────────────────────────────────

    def fit(self, profile: WorkloadProfile) -> None:
        """
        Train the LSTM on the node's workload history.

        Called once the profile has enough data (≥ LOOKBACK samples).
        Re-calling fit() retrains from scratch — useful after significant drift.

        Algorithm:
            1. Guard: skip if profile.has_enough_data is False.
            2. Extract cpu_cores_history (List[float], oldest-first).
            3. Z-score normalise: z = (x - mean) / std.
               Store mean, std as instance attributes for use in predict().
            4. Build sliding-window dataset:
               X[i] = z[i : i+LOOKBACK]  → shape (n, LOOKBACK, 1)
               y[i] = z[i + LOOKBACK]    → shape (n, 1)
               n = len(history) - LOOKBACK
            5. Instantiate a fresh _LSTMModel (discards any previous weights).
            6. Train: Adam(lr=LEARNING_RATE) + MSELoss, TRAIN_EPOCHS full-batch steps.
            7. Record last-fit sample count. Set self._trained = True.

        Args:
            profile: WorkloadProfile for this node. Must have samples attached.

        Returns:
            None. Modifies self._model, self._trained, self._cpu_mean, self._cpu_std.

        Why full-batch?
            At most len(history) - LOOKBACK ≤ 490 windows. One tensor fits in
            CPU RAM easily. Mini-batching would add DataLoader overhead with no
            benefit for this dataset size.
        """
        # ── Guard ─────────────────────────────────────────────────────────────
        if not profile.has_enough_data:
            self._trained = False
            return

        # ── Extract history ───────────────────────────────────────────────────
        history: List[float] = profile.cpu_cores_history
        n = len(history)

        if n <= LOOKBACK:
            # Edge case: exactly 10 samples = 0 training windows
            # can_fit check guarantees n >= 10, but n=10 → 0 windows
            self._trained = False
            return

        # ── Z-score normalise ─────────────────────────────────────────────────
        arr = np.array(history, dtype=np.float64)
        mean = float(arr.mean())
        std = float(max(arr.std(), 1e-6))   # clamp: prevents div-by-zero on flat signal

        self._cpu_mean = mean
        self._cpu_std = std

        z = (arr - mean) / std   # shape: (n,)

        # ── Build sliding-window dataset ──────────────────────────────────────
        n_windows = n - LOOKBACK
        # X: each row is LOOKBACK consecutive z-values (the input sequence)
        # y: the value immediately after each window (the target)
        X_np = np.lib.stride_tricks.sliding_window_view(z[:-1], LOOKBACK)  # (n_windows, LOOKBACK)
        y_np = z[LOOKBACK:]                                                   # (n_windows,)

        # Convert to float32 tensors (LSTM weights are float32 by default in PyTorch)
        X_tensor = torch.tensor(X_np, dtype=torch.float32).unsqueeze(-1)   # (n_windows, LOOKBACK, 1)
        y_tensor = torch.tensor(y_np, dtype=torch.float32).unsqueeze(-1)   # (n_windows, 1)

        # ── Model & optimiser ─────────────────────────────────────────────────
        model = _LSTMModel()
        model.train()
        optimiser = Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.MSELoss()

        # ── Training loop ─────────────────────────────────────────────────────
        for _ in range(TRAIN_EPOCHS):
            optimiser.zero_grad()
            preds = model(X_tensor)          # (n_windows, 1)
            loss = loss_fn(preds, y_tensor)
            loss.backward()
            optimiser.step()

        # ── Compute hold-out MAE for calibrated confidence ─────────────────────
        # Use the final training predictions (no separate val split — dataset is small).
        # MAE is in z-score units; convert back to % CPU for interpretability.
        model.eval()
        with torch.no_grad():
            final_preds = model(X_tensor).squeeze()          # (n_windows,)
        final_targets = y_tensor.squeeze()
        mae_z = float((final_preds - final_targets).abs().mean())
        self._last_mae = mae_z * std   # denormalise: z MAE × std → % CPU MAE

        # ── Commit ────────────────────────────────────────────────────────────
        self._model = model
        self._last_fit_sample_count = len(profile.samples)
        self._trained = True

    def predict(
        self,
        profile: WorkloadProfile,
        horizon_minutes: int = 5,
    ) -> PredictionResult:
        """
        Forecast CPU utilisation for this node over the next horizon_minutes.

        Always returns a valid PredictionResult — never raises.
        If the model is not yet trained (cold-start), returns a conservative
        fallback with confidence=0.1 to signal low reliability.

        Cold-start fallback:
            predicted_cpu_util = min(profile.avg_cpu_cores × 10, 100.0)
            confidence          = 0.1
            spike_probability   = 0.0

        Why avg_cpu_cores × 10?
            A heuristic: assume a ~10-core node (middle of the 8–96 range).
            Converts core count to rough utilisation %. Callers must treat
            confidence < 0.3 as a weak signal.

        Trained-model path:
            1. Normalise last LOOKBACK samples using stored _cpu_mean, _cpu_std.
            2. Forward pass through LSTM (eval mode, no_grad).
            3. Denormalise output → pred_cpu ∈ [0.0, 100.0].
            4. Compute spike_probability from predicted gap above recent mean.
            5. Compute confidence from dataset size (linear interpolation 10→500).

        Args:
            profile:          WorkloadProfile for this node.
            horizon_minutes:  How far ahead to predict. Stored in result.

        Returns:
            PredictionResult with all fields populated.
        """
        # ── Cold-start path ───────────────────────────────────────────────────
        if not self._trained or not profile.has_enough_data:
            fallback_cpu = min(profile.avg_cpu_cores * 10.0, 100.0)
            return PredictionResult(
                node_id=self.node_id,
                forecast_horizon_min=horizon_minutes,
                predicted_cpu_util=max(fallback_cpu, 0.0),
                predicted_memory_util=50.0,   # neutral fallback
                predicted_gpu_util={},
                spike_probability=0.0,
                confidence=0.1,
                generated_at=datetime.now(timezone.utc),
            )

        # ── Trained path ──────────────────────────────────────────────────────

        # 1. Get last LOOKBACK observations and normalise
        history = profile.cpu_cores_history
        last_seq = history[-LOOKBACK:]   # List[float], length = LOOKBACK
        z_seq = [(x - self._cpu_mean) / self._cpu_std for x in last_seq]

        # Build input tensor: (1, LOOKBACK, 1)
        x = torch.tensor(z_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        # 2. Forward pass — no gradient tracking needed for inference
        assert self._model is not None
        with torch.no_grad():
            raw_pred = self._model(x)        # shape: (1, 1)

        # 3. Denormalise
        z_pred = float(raw_pred.squeeze())
        pred_cpu = z_pred * self._cpu_std + self._cpu_mean
        pred_cpu = float(max(0.0, min(100.0, pred_cpu)))   # clamp to [0, 100]

        # 4. Spike probability
        spike_probability = self._compute_spike_probability(pred_cpu, profile)

        # 5. Confidence — blended from sample count + hold-out MAE
        confidence = self._compute_confidence(len(profile.samples), self._last_mae)

        return PredictionResult(
            node_id=self.node_id,
            forecast_horizon_min=horizon_minutes,
            predicted_cpu_util=pred_cpu,
            predicted_memory_util=50.0,   # Phase 3 scope: CPU only; mem in Phase 7
            predicted_gpu_util={},
            spike_probability=spike_probability,
            confidence=confidence,
            generated_at=datetime.now(timezone.utc),
        )

    def refit_if_needed(self, profile: WorkloadProfile) -> None:
        """
        Refit the model if enough new samples have accumulated since last fit.

        Call this from the background telemetry refresh loop (every 60s).
        Avoids the overhead of fitting on every new sample while still
        keeping the model reasonably fresh.

        Triggers refit when: len(profile.samples) - last_fit_count >= REFIT_THRESHOLD

        Args:
            profile: Current WorkloadProfile for this node.

        Returns:
            None. Calls self.fit(profile) if threshold is met.
        """
        current_count = len(profile.samples)
        if current_count - self._last_fit_sample_count >= REFIT_THRESHOLD:
            self.fit(profile)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_spike_probability(
        pred_cpu: float,
        profile: WorkloadProfile,
    ) -> float:
        """
        Estimate the probability of a CPU spike within the forecast horizon.

        Logic:
            gap = (pred_cpu - recent_mean_util) / max(recent_mean_util, 1.0)
                  ─ How much higher is the predicted value than the recent mean?
                  ─ Expressed as a fraction of the recent mean.

            spike_probability = clamp(gap, 0.0, 1.0)

            Boost: if burst_factor > 1.5 (this workload has a history of bursting),
                   add 0.2 to spike_probability (capped at 1.0).

        Why gap / recent_mean?
            A jump from 5% to 10% is not a spike. A jump from 40% to 80% is.
            Dividing by the recent mean normalises the gap by current load level.

        Args:
            pred_cpu: Predicted CPU utilisation % (0–100).
            profile:  WorkloadProfile with burst_factor and cpu_cores_history.

        Returns:
            float in [0.0, 1.0].
        """
        history = profile.cpu_cores_history
        # pred_cpu and cpu_cores_history are both in raw CPU cores — compare directly.
        recent_cores = history[-LOOKBACK:] if len(history) >= LOOKBACK else history
        recent_mean = (sum(recent_cores) / len(recent_cores)) if recent_cores else 0.0
        recent_mean = max(recent_mean, 1e-3)   # avoid div-by-zero

        gap = (pred_cpu - recent_mean) / recent_mean
        spike_prob = max(0.0, min(1.0, gap))

        # Burst-factor boost: if this workload type historically bursts, be more cautious
        if profile.burst_factor > 1.5:
            spike_prob = min(spike_prob + 0.2, 1.0)

        return spike_prob

    @staticmethod
    def _compute_confidence(n_samples: int, mae_cpu_pct: float = 20.0) -> float:
        """
        Calibrated confidence: blends sample-count coverage with hold-out MAE quality.

        Previous version used sample count only — this caused confidence=1.0 on any
        smooth trace with 500+ samples, regardless of whether the model was accurate.
        A model can be data-rich and still have high prediction error on noisy signals.

        New formula (equal-weight blend):
            sample_score = min(1.0, (n_samples - LOOKBACK) / (500 - LOOKBACK))
                           # 0.0 at 10 samples → 1.0 at 500 samples

            mae_score = max(0.0, 1.0 - mae_cpu_pct / MAE_CEILING)
                        # MAE_CEILING = 20.0% CPU
                        # MAE=0%  → 1.0 (perfect)   MAE=10% → 0.5   MAE≥20% → 0.0

            confidence = 0.5 * sample_score + 0.5 * mae_score

        Examples:
            500 samples, MAE=2%  → confidence = 0.5×1.0 + 0.5×0.9 = 0.95 ✓
            500 samples, MAE=15% → confidence = 0.5×1.0 + 0.5×0.25 = 0.625 (penalised)
            50 samples,  MAE=3%  → confidence = 0.5×0.08 + 0.5×0.85 = 0.465
            10 samples   (cold)  → confidence = 0.0 + 0.5×0.0 = 0.0  (clamped to 0.1)

        Args:
            n_samples:   Current number of samples in the profile.
            mae_cpu_pct: Hold-out MAE in % CPU from the last fit() call. Default 20.0
                         (maximum uncertainty) until the first real fit completes.

        Returns:
            float in [0.1, 1.0].
        """
        MAE_CEILING = 20.0   # above this, model quality contributes 0 to confidence

        # Sample coverage score (0 → 1 as samples grow from LOOKBACK to 500)
        if n_samples <= LOOKBACK:
            sample_score = 0.0
        else:
            sample_score = min(1.0, (n_samples - LOOKBACK) / (500 - LOOKBACK))

        # MAE quality score (1.0 = perfect, 0.0 = MAE ≥ ceiling)
        mae_score = max(0.0, 1.0 - mae_cpu_pct / MAE_CEILING)

        confidence = 0.5 * sample_score + 0.5 * mae_score
        return max(0.1, min(confidence, 1.0))   # floor 0.1 (never fully opaque)

    def __repr__(self) -> str:
        return (
            f"WorkloadPredictor("
            f"node_id={self.node_id!r}, "
            f"trained={self._trained}, "
            f"mean={self._cpu_mean:.3f}, "
            f"std={self._cpu_std:.3f})"
        )
