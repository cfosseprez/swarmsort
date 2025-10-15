"""
SwarmSort Kalman Filtering Module

This module provides Kalman filtering implementations for object tracking,
including both simple and OC-SORT style filters for motion prediction
and state estimation.

Functions:
    simple_kalman_update: Simple Kalman filter update step
    simple_kalman_predict: Simple Kalman filter prediction step
    oc_sort_predict: OC-SORT style prediction using observation history
    oc_sort_update: OC-SORT style update for observation history
    compute_oc_sort_cost_matrix: OC-SORT style cost matrix computation
"""

# ============================================================================
# STANDARD IMPORTS
# ============================================================================
import numpy as np
import numba as nb
from typing import Tuple


# ============================================================================
# SIMPLE KALMAN FILTER FUNCTIONS
# ============================================================================

def simple_kalman_update(x_pred: np.ndarray, z: np.ndarray, is_reid: bool = False) -> np.ndarray:
    """
    Simplified Kalman update with proper type handling.

    Args:
        x_pred: Predicted state [x, y, vx, vy]
        z: Measurement [x, y]
        is_reid: Whether this update is for re-identification (resets velocity)

    Returns:
        Updated state vector
    """
    alpha = 0.7
    x_updated = np.zeros(4, dtype=np.float32)

    # Ensure we're getting scalars by using .item() or flatten
    z0 = z[0] if z.ndim == 1 else z.flat[0]
    z1 = z[1] if z.ndim == 1 else z.flat[1]

    x_updated[0] = alpha * z0 + (1 - alpha) * x_pred[0]
    x_updated[1] = alpha * z1 + (1 - alpha) * x_pred[1]

    if is_reid:
        # Reset velocity on re-identification
        x_updated[2] = 0.0
        x_updated[3] = 0.0
    else:
        x_updated[2] = x_pred[2]
        x_updated[3] = x_pred[3]

    return x_updated


@nb.njit(fastmath=True, cache=True)
def simple_kalman_predict(x: np.ndarray) -> np.ndarray:
    """
    Simplified Kalman prediction.

    Args:
        x: Current state [x, y, vx, vy]

    Returns:
        Predicted state vector
    """
    x_pred = np.zeros(4, dtype=np.float32)
    x_pred[0] = x[0] + x[2]
    x_pred[1] = x[1] + x[3]
    x_pred[2] = x[2] * 0.95
    x_pred[3] = x[3] * 0.95
    return x_pred


# ============================================================================
# OC-SORT STYLE FUNCTIONS
# ============================================================================

@nb.njit(fastmath=True, cache=True)
def oc_sort_predict(observation_history: np.ndarray,
                    observation_frames: np.ndarray,
                    current_frame: int) -> np.ndarray:
    """
    OC-SORT style prediction using observation history.

    Args:
        observation_history: Array of past observations [N, 2]
        observation_frames: Array of frame numbers for observations [N]
        current_frame: Current frame number

    Returns:
        Predicted state [x, y, vx, vy]
    """
    n_obs = len(observation_frames)

    if n_obs == 0:
        # No observations, return zeros
        return np.zeros(4, dtype=np.float32)

    if n_obs == 1:
        # Single observation, no velocity
        pred = np.zeros(4, dtype=np.float32)
        pred[0] = observation_history[0, 0]
        pred[1] = observation_history[0, 1]
        return pred

    # Use last two observations for velocity estimation
    if n_obs >= 2:
        # Time delta between last two observations
        dt = observation_frames[-1] - observation_frames[-2]
        if dt == 0:
            dt = 1

        # Velocity from last two observations
        vx = (observation_history[-1, 0] - observation_history[-2, 0]) / dt
        vy = (observation_history[-1, 1] - observation_history[-2, 1]) / dt

        # Time since last observation
        delta_t = current_frame - observation_frames[-1]

        # Predict position
        pred = np.zeros(4, dtype=np.float32)
        pred[0] = observation_history[-1, 0] + vx * delta_t
        pred[1] = observation_history[-1, 1] + vy * delta_t
        pred[2] = vx
        pred[3] = vy

        return pred

    return np.zeros(4, dtype=np.float32)


@nb.njit(fastmath=True, cache=True)
def oc_sort_update(observation_history: np.ndarray,
                   observation_frames: np.ndarray,
                   new_observation: np.ndarray,
                   current_frame: int,
                   max_history: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    OC-SORT style update - stores observations without filtering.

    Args:
        observation_history: Current observation history [N, 2]
        observation_frames: Current frame history [N]
        new_observation: New observation [2]
        current_frame: Current frame number
        max_history: Maximum history size

    Returns:
        Tuple of (updated_history, updated_frames)
    """
    n_obs = len(observation_frames)

    if n_obs < max_history:
        # Append to history
        new_history = np.zeros((n_obs + 1, 2), dtype=np.float32)
        new_frames = np.zeros(n_obs + 1, dtype=np.int32)

        if n_obs > 0:
            new_history[:n_obs] = observation_history
            new_frames[:n_obs] = observation_frames

        new_history[n_obs] = new_observation
        new_frames[n_obs] = current_frame
    else:
        # Shift history (remove oldest)
        new_history = np.zeros((max_history, 2), dtype=np.float32)
        new_frames = np.zeros(max_history, dtype=np.int32)

        new_history[:-1] = observation_history[1:]
        new_frames[:-1] = observation_frames[1:]

        new_history[-1] = new_observation
        new_frames[-1] = current_frame

    return new_history, new_frames


@nb.njit(fastmath=True, cache=True)
def compute_oc_sort_cost_matrix(
    det_positions: np.ndarray,
    track_last_observed_positions: np.ndarray,
    track_velocities: np.ndarray,
    track_misses: np.ndarray,
    max_distance: float,
    velocity_weight: float = 0.2
) -> np.ndarray:
    """
    OC-SORT style cost computation with adaptive thresholds.

    Args:
        det_positions: Detection positions [N_det, 2]
        track_last_observed_positions: Last observed track positions [N_track, 2]
        track_velocities: Track velocities [N_track, 2]
        track_misses: Number of misses for each track [N_track]
        max_distance: Maximum allowed distance for matching
        velocity_weight: Weight for velocity consistency term

    Returns:
        Cost matrix [N_det, N_track]
    """
    n_dets = det_positions.shape[0]
    n_tracks = track_last_observed_positions.shape[0]
    cost_matrix = np.full((n_dets, n_tracks), np.inf, dtype=np.float32)

    for i in range(n_dets):
        for j in range(n_tracks):
            # Base distance from last observed position
            dist = np.sqrt(
                (det_positions[i, 0] - track_last_observed_positions[j, 0])**2 +
                (det_positions[i, 1] - track_last_observed_positions[j, 1])**2
            )

            # Adaptive threshold based on misses
            adaptive_threshold = max_distance * (1.0 + 0.2 * track_misses[j])

            if dist > adaptive_threshold:
                continue

            # Velocity consistency for recently seen tracks
            if track_misses[j] == 0:
                # Expected position based on velocity
                expected_x = track_last_observed_positions[j, 0] + track_velocities[j, 0]
                expected_y = track_last_observed_positions[j, 1] + track_velocities[j, 1]

                velocity_error = np.sqrt(
                    (det_positions[i, 0] - expected_x)**2 +
                    (det_positions[i, 1] - expected_y)**2
                )

                # Combined cost with velocity term
                cost_matrix[i, j] = dist + velocity_weight * velocity_error
            else:
                cost_matrix[i, j] = dist

    return cost_matrix