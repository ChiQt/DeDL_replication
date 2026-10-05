"""Structured response calculations and the influence score in Eq. (5)."""
import numpy as np
from .data import COMBOS


def response_and_gradient(theta):
    """theta=[a,b1,b2,b3,c]; return G(n,8) and derivative dG/dtheta(n,8,5)."""
    treatment_design = np.column_stack([np.ones(8), COMBOS])
    index = theta[:, :4] @ treatment_design.T
    probability = 1 / (1 + np.exp(-np.clip(index, -500, 500)))
    mean = theta[:, [4]] * probability
    index_derivative = theta[:, [4]] * probability * (1 - probability)
    gradient = np.concatenate([
        index_derivative[:, :, None] * treatment_design[None, :, :],
        probability[:, :, None]
    ], axis=2)
    return mean, gradient


def influence_scores(theta, treatment_index, y, ridge):
    """Return plug-in and corrected individual contrasts, plus conditioning diagnostics.

    Lambda uses the five actual support points at probability 1/5. The loss
    gradient ALWAYS uses observed T; target t only enters the contrast H.
    Solving a linear system avoids explicitly inverting each 5x5 matrix.
    """
    predictions, gradient = response_and_gradient(theta)
    observed_gradient = gradient[:, :5]
    hessian = 2 * np.einsum("nki,nkj->nij", observed_gradient, observed_gradient) / 5
    users = np.arange(len(y))
    residual = predictions[users, treatment_index] - y
    loss_gradient = 2 * gradient[users, treatment_index] * residual[:, None]
    adjustment = np.linalg.solve(
        hessian + ridge * np.eye(5), loss_gradient[:, :, None]
    )[:, :, 0]
    plugin = predictions - predictions[:, [0]]
    contrast_gradient = gradient - gradient[:, [0]]
    corrected = plugin - np.einsum("nki,ni->nk", contrast_gradient, adjustment)
    eigenvalues = np.linalg.eigvalsh(hessian)
    diagnostics = {
        "lambda_median_min_eigenvalue": float(np.median(eigenvalues[:, 0])),
        "median_regularized_condition": float(np.median(
            (eigenvalues[:, -1] + ridge) / (eigenvalues[:, 0] + ridge)
        )),
        "max_absolute_score": float(np.max(np.abs(corrected)))
    }
    return plugin, corrected, diagnostics


def metrics(estimate, truth, mape_mask):
    """MAPE uses a fixed independent-reference mask; absolute errors use all effects."""
    error = np.asarray(estimate) - truth
    unseen = np.arange(8) >= 5
    selected_unseen = unseen & mape_mask
    return {
        "mape": float(100 * np.mean(np.abs(error[mape_mask] / truth[mape_mask]))),
        "mae": float(np.mean(np.abs(error[1:]))),
        "rmse": float(np.sqrt(np.mean(error[1:] ** 2))),
        "unseen_mae": float(np.mean(np.abs(error[unseen]))),
        "unseen_mape": float(100 * np.mean(np.abs(
            error[selected_unseen] / truth[selected_unseen]
        ))) if selected_unseen.any() else None
    }


def check_formula():
    """Tests target statistical identities, not just matching a second implementation."""
    theta = np.array([[0.2, 0.7, -0.8, 0.9, 12.0]])
    predictions, gradient = response_and_gradient(theta)
    for j in range(5):
        perturbation = np.zeros_like(theta)
        perturbation[:, j] = 1e-5
        numeric = (response_and_gradient(theta + perturbation)[0]
                   - response_and_gradient(theta - perturbation)[0]) / 2e-5
        np.testing.assert_allclose(numeric, gradient[:, :, j], atol=1e-8, rtol=1e-6)

    repeated_theta = np.repeat(theta, 5, axis=0)
    treatment_index = np.arange(5)
    y = predictions[0, :5]
    plugin, corrected, _ = influence_scores(repeated_theta, treatment_index, y, 0.0005)
    np.testing.assert_allclose(plugin, corrected, atol=1e-12)
    for j in range(5):
        perturbation = np.zeros_like(repeated_theta)
        perturbation[:, j] = 1e-5
        plus = influence_scores(repeated_theta + perturbation, treatment_index, y, 0)[1]
        minus = influence_scores(repeated_theta - perturbation, treatment_index, y, 0)[1]
        np.testing.assert_allclose((plus.mean(0) - minus.mean(0)) / 2e-5, 0, atol=2e-6)
    return "PASS: analytic gradient, zero-residual correction, unregularized Neyman orthogonality"
