"""Método del gradiente conjugado no lineal (Polak-Ribière+) con Wolfe."""
from __future__ import annotations

import numpy as np

from .wolfe import wolfe_line_search


def conjugate_gradient(
    f,
    grad_f,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 500,
    c1: float = 1e-4,
    c2: float = 0.1,
    variant: str = "PR+",
) -> dict:
    """Gradiente conjugado no lineal.

    variant: "FR" (Fletcher-Reeves) o "PR+" (Polak-Ribière con reinicio).
    Reinicia la dirección a -grad cada n iteraciones para mantener
    convergencia global.
    """
    x = np.array(x0, dtype=float).copy()
    n = len(x)
    g = grad_f(x)
    p = -g.copy()
    history = []

    stop_reason = "max_iter"
    for k in range(max_iter):
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

        # Asegurar dirección de descenso; si no, reiniciar
        if float(np.dot(g, p)) >= 0:
            p = -g.copy()

        alpha = wolfe_line_search(f, grad_f, x, p, c1=c1, c2=c2)
        x_new = x + alpha * p
        g_new = grad_f(x_new)

        history[-1]["alpha"] = alpha

        # Coeficiente beta
        if variant == "FR":
            beta = float(np.dot(g_new, g_new) / max(np.dot(g, g), 1e-16))
        else:  # PR+
            beta = float(np.dot(g_new, g_new - g) / max(np.dot(g, g), 1e-16))
            beta = max(beta, 0.0)

        # Reinicio periódico
        if (k + 1) % n == 0:
            p_new = -g_new
        else:
            p_new = -g_new + beta * p

        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            g = g_new
            stop_reason = "||x_{k+1} - x_k|| < tol"
            break

        x, g, p = x_new, g_new, p_new

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
        "method": f"Método del gradiente conjugado ({variant})",
    }
