"""Método de Newton con modificación de Hessiana y búsqueda Wolfe."""
from __future__ import annotations

import numpy as np

from .wolfe import wolfe_line_search


def newton_method(
    f,
    grad_f,
    hess_f,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 200,
    c1: float = 1e-4,
    c2: float = 0.9,
) -> dict:
    """Método de Newton amortiguado.

    Resuelve H*p = -grad en cada iteración. Si H no es definida positiva,
    aplica modificación tipo Levenberg (H + tau*I) para garantizar dirección
    de descenso. El paso se calcula con búsqueda de línea Wolfe.
    """
    x = np.array(x0, dtype=float).copy()
    history = []

    stop_reason = "max_iter"
    for k in range(max_iter):
        g = grad_f(x)
        gnorm = float(np.linalg.norm(g))
        fx = float(f(x))

        history.append({
            "iter": k,
            "x": x.copy(),
            "f": fx,
            "grad_norm": gnorm,
            "alpha": np.nan,
        })

        if gnorm < tol:
            stop_reason = "||grad|| < tol"
            break

        H = hess_f(x)
        p = _solve_newton_direction(H, g)

        # Garantizar descenso
        if float(np.dot(g, p)) >= 0:
            p = -g

        alpha = wolfe_line_search(f, grad_f, x, p, c1=c1, c2=c2)
        x_new = x + alpha * p

        history[-1]["alpha"] = alpha

        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            stop_reason = "||x_{k+1} - x_k|| < tol"
            break

        x = x_new

    history.append({
        "iter": len(history),
        "x": x.copy(),
        "f": float(f(x)),
        "grad_norm": float(np.linalg.norm(grad_f(x))),
        "alpha": np.nan,
    })

    return {
        "x_min": x,
        "f_min": float(f(x)),
        "iterations": len(history) - 1,
        "stop_reason": stop_reason,
        "history": history,
        "method": "Método de Newton",
    }


def _solve_newton_direction(H: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Resuelve H*p = -g. Si H no es definida positiva, agrega tau*I."""
    n = len(g)
    H_sym = 0.5 * (H + H.T)

    beta = 1e-3
    min_diag = float(np.min(np.diag(H_sym)))
    tau = 0.0 if min_diag > 0 else -min_diag + beta

    for _ in range(50):
        try:
            L = np.linalg.cholesky(H_sym + tau * np.eye(n))
            y = np.linalg.solve(L, -g)
            p = np.linalg.solve(L.T, y)
            return p
        except np.linalg.LinAlgError:
            tau = max(2.0 * tau, beta)

    # Fallback robusto
    return np.linalg.solve(H_sym + tau * np.eye(n), -g)
