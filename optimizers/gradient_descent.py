"""Método del gradiente (descenso máximo) con búsqueda de línea Wolfe."""
from __future__ import annotations

import numpy as np

from .wolfe import wolfe_line_search


def gradient_descent(
    f,
    grad_f,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 500,
    c1: float = 1e-4,
    c2: float = 0.9,
) -> dict:
    """Minimiza f por el método del gradiente.

    Dirección: p_k = -grad f(x_k).
    Paso: búsqueda de línea con condiciones de Wolfe.
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
        "method": "Método del gradiente",
    }
