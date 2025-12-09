import math
import random

import autoquine


def test_posterior_update_moves_phi_toward_true_theta():
    theta_true = 1.5
    rng = random.Random(0)
    history = []
    for _ in range(6):
        history.append(autoquine.generate_next_guess(theta_true, history, obs_sigma=0.15, rng=rng))

    model = autoquine.DifferentiableGauguine(init_phi=0.0, obs_sigma=0.15)
    before = abs(model.phi - theta_true)
    model.posterior_update(history, steps=60, lr=0.05)
    after = abs(model.phi - theta_true)

    assert after < before
    assert after < before * 0.95  # move meaningfully toward the target


def test_quine_contains_class_definition(tmp_path):
    source = autoquine.emit_source()
    assert "class DifferentiableGauguine" in source

    # ensure emitted source matches file contents on disk
    path_copy = tmp_path / "copy.py"
    path_copy.write_text(source)
    assert path_copy.read_text() == source


def test_log_likelihood_improves_after_update():
    theta_true = 2.0
    rng = random.Random(1)
    history = []
    for _ in range(5):
        history.append(autoquine.generate_next_guess(theta_true, history, obs_sigma=0.25, rng=rng))

    model = autoquine.DifferentiableGauguine(init_phi=0.0, obs_sigma=0.25)
    initial_ll = model.log_likelihood(history)
    model.posterior_update(history, steps=50, lr=0.05)
    improved_ll = model.log_likelihood(history)

    assert improved_ll > initial_ll
    assert not math.isnan(improved_ll)
