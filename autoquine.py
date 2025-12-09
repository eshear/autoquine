"""Autoquine: differential gauguine simulation and self-emission.

This module implements a tiny differentiable gauguine, inspired by the
shared write-up about combining Bayesian self-inference with
Futamura-style specialization.  The gauguine alternates between
"daytime" generation using a hidden true parameter and "nighttime"
self-fitting via gradient ascent on the likelihood of its own history.

After running the diurnal cycle it can emit its own source code,
providing a lightweight autoquine demonstration.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def _mean(history: Sequence[float]) -> float:
    values = list(history)
    return sum(values) / len(values) if values else 0.0


def generate_next_guess(
    theta_true: float,
    history: Sequence[float],
    *,
    obs_sigma: float = 0.3,
    rng: random.Random | None = None,
) -> float:
    """Sample the next observation from the "true" program.

    The mean depends on the average of the previous history, filtered
    through a tanh nonlinearity to keep values bounded.
    """
    rng = rng or random
    mean_input = _mean(history)
    mean = theta_true * math.tanh(mean_input)
    return rng.gauss(mean, obs_sigma)


class DifferentiableGauguine:
    """Minimal differentiable self-model.

    This object tracks a scalar belief ``phi`` about the underlying
    program parameter ``theta_true``.  Given a history of observations it
    performs gradient ascent on the log likelihood to update ``phi``.
    """

    def __init__(self, *, init_phi: float = 0.0, obs_sigma: float = 0.3):
        self.phi = float(init_phi)
        self.obs_sigma = float(obs_sigma)

    def predict_next(self, history: Sequence[float]) -> float:
        if not history:
            return 0.0
        return self.phi * math.tanh(_mean(history))

    def log_likelihood(self, history: Sequence[float]) -> float:
        history_list = list(history)
        if len(history_list) <= 1:
            return 0.0

        total = 0.0
        for t in range(1, len(history_list)):
            prev = history_list[:t]
            target = history_list[t]
            mean_pred = self.phi * math.tanh(_mean(prev))
            total += _log_normal_pdf(target, mean_pred, self.obs_sigma)
        return total

    def posterior_update(self, history: Sequence[float], *, steps: int = 20, lr: float = 0.1) -> float:
        """Perform in-place gradient ascent on ``phi``.

        Returns the updated ``phi`` value.
        """
        history_list = list(history)
        if len(history_list) <= 1:
            return self.phi

        for _ in range(steps):
            gradient = 0.0
            for t in range(1, len(history_list)):
                prev = history_list[:t]
                target = history_list[t]
                prev_mean = _mean(prev)
                mean_pred = self.phi * math.tanh(prev_mean)
                error = target - mean_pred
                dmean_dphi = math.tanh(prev_mean)
                gradient += error * dmean_dphi / (self.obs_sigma ** 2)
            self.phi += lr * gradient
        return self.phi


def _log_normal_pdf(x: float, mean: float, sigma: float) -> float:
    variance = sigma ** 2
    return -0.5 * math.log(2 * math.pi * variance) - 0.5 * ((x - mean) ** 2) / variance


def run_diurnal_cycle(
    *,
    theta_true_val: float = 2.0,
    obs_sigma: float = 0.3,
    num_days: int = 20,
    night_steps: int = 30,
    lr: float = 0.1,
    seed: int | None = None,
) -> List[Tuple[int, float, float]]:
    """Run a full daytime/nighttime simulation.

    Returns a list of tuples ``(day, observation, phi_estimate)``.
    """
    rng = random.Random(seed)
    history: List[float] = []
    g = DifferentiableGauguine(init_phi=0.0, obs_sigma=obs_sigma)

    timeline: List[Tuple[int, float, float]] = []
    for day in range(1, num_days + 1):
        obs = generate_next_guess(theta_true_val, history, obs_sigma=obs_sigma, rng=rng)
        history.append(obs)
        g.posterior_update(history, steps=night_steps, lr=lr)
        timeline.append((day, obs, g.phi))
    return timeline


def emit_source() -> str:
    """Return the current module's source code (a tiny quine)."""
    return Path(__file__).read_text()


def _print_diurnal_table(timeline: Iterable[Tuple[int, float, float]], theta_true_val: float) -> None:
    print(f"True θ* = {theta_true_val:.3f}")
    print(f"{'day':>3} | {'guess h_t':>10} | {'φ (self-estimate)':>18}")
    print("-" * 40)
    for day, obs, phi in timeline:
        print(f"{day:3d} | {obs:10.4f} | {phi:18.4f}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Differentiable gauguine autoquine demo")
    parser.add_argument("--theta", type=float, default=2.0, help="true underlying parameter θ*")
    parser.add_argument("--obs-sigma", type=float, default=0.3, help="observation noise σ")
    parser.add_argument("--days", type=int, default=20, help="number of daytime/night cycles")
    parser.add_argument("--night-steps", type=int, default=30, help="gradient steps per night")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate for φ updates")
    parser.add_argument("--seed", type=int, default=None, help="PRNG seed for reproducibility")
    parser.add_argument("--no-quine", action="store_true", help="suppress emitting the program source")

    args = parser.parse_args(argv)

    timeline = run_diurnal_cycle(
        theta_true_val=args.theta,
        obs_sigma=args.obs_sigma,
        num_days=args.days,
        night_steps=args.night_steps,
        lr=args.lr,
        seed=args.seed,
    )

    _print_diurnal_table(timeline, args.theta)

    if not args.no_quine:
        print("\n--- autoquine source ---\n")
        print(emit_source())


if __name__ == "__main__":  # pragma: no cover
    main()
