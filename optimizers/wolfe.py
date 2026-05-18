"""Búsqueda de línea con condiciones de Wolfe (fuerte).

Implementa el algoritmo de Nocedal & Wright (Numerical Optimization, 2ed,
Algoritmos 3.5 y 3.6) para encontrar un paso alpha que satisface:

  Armijo (1ra de Wolfe):
      f(x + alpha*p) <= f(x) + c1 * alpha * grad_f(x)^T p

  Curvatura fuerte (2da de Wolfe):
      |grad_f(x + alpha*p)^T p| <= c2 * |grad_f(x)^T p|

Valores típicos: c1 = 1e-4, c2 = 0.9 (Newton/cuasi-Newton),
c2 = 0.1 (gradiente conjugado).
"""
from __future__ import annotations

import numpy as np


def wolfe_line_search(
    f,
    grad_f,
    x: np.ndarray,
    p: np.ndarray,
    c1: float = 1e-4,
    c2: float = 0.9,
    alpha_max: float = 10.0,
    max_iter: int = 50,
) -> float:
    """Devuelve un paso alpha que cumple las condiciones de Wolfe fuertes.

    Si no converge en max_iter, devuelve el mejor candidato encontrado.
    """
    phi0 = f(x)
    g0 = grad_f(x)
    dphi0 = float(np.dot(g0, p))

    if dphi0 >= 0:
        # p no es dirección de descenso; devolvemos un paso pequeño positivo
        return 1e-4

    alpha_prev = 0.0
    phi_prev = phi0
    alpha = 1.0 if alpha_max >= 1.0 else 0.5 * alpha_max

    for i in range(1, max_iter + 1):
        x_new = x + alpha * p
        phi = f(x_new)

        if (phi > phi0 + c1 * alpha * dphi0) or (i > 1 and phi >= phi_prev):
            return _zoom(f, grad_f, x, p, alpha_prev, alpha,
                         phi0, dphi0, c1, c2, max_iter)

        g_new = grad_f(x_new)
        dphi = float(np.dot(g_new, p))

        if abs(dphi) <= -c2 * dphi0:
            return alpha

        if dphi >= 0:
            return _zoom(f, grad_f, x, p, alpha, alpha_prev,
                         phi0, dphi0, c1, c2, max_iter)

        alpha_prev = alpha
        phi_prev = phi
        alpha = min(2.0 * alpha, alpha_max)

        if alpha >= alpha_max:
            return alpha_max

    return alpha


def _zoom(f, grad_f, x, p, alpha_lo, alpha_hi,
          phi0, dphi0, c1, c2, max_iter) -> float:
    for _ in range(max_iter):
        alpha = 0.5 * (alpha_lo + alpha_hi)
        x_new = x + alpha * p
        phi = f(x_new)
        phi_lo = f(x + alpha_lo * p)

        if (phi > phi0 + c1 * alpha * dphi0) or (phi >= phi_lo):
            alpha_hi = alpha
        else:
            g_new = grad_f(x_new)
            dphi = float(np.dot(g_new, p))
            if abs(dphi) <= -c2 * dphi0:
                return alpha
            if dphi * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha

        if abs(alpha_hi - alpha_lo) < 1e-12:
            return alpha
    return 0.5 * (alpha_lo + alpha_hi)
