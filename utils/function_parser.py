"""Parser de funciones objetivo basado en SymPy.

Convierte una expresión simbólica en callables vectorizados de NumPy para
la función, el gradiente y la Hessiana.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import sympy as sp


@dataclass
class ParsedFunction:
    expr: sp.Expr
    variables: List[sp.Symbol]
    f: Callable[[np.ndarray], float]
    grad: Callable[[np.ndarray], np.ndarray]
    hess: Callable[[np.ndarray], np.ndarray]
    latex: str
    grad_latex: str
    hess_latex: str


# Funciones de prueba clásicas
PRESETS = {
    "Esfera (cuadrática simple)": {
        "expr": "x1**2 + x2**2",
        "n": 2,
        "x0": [3.0, 4.0],
        "min": "(0, 0)",
    },
    "Rosenbrock (banana)": {
        "expr": "100*(x2 - x1**2)**2 + (1 - x1)**2",
        "n": 2,
        "x0": [-1.2, 1.0],
        "min": "(1, 1)",
    },
    "Himmelblau": {
        "expr": "(x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2",
        "n": 2,
        "x0": [0.0, 0.0],
        "min": "(3, 2) y otros 3 mínimos",
    },
    "Booth": {
        "expr": "(x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2",
        "n": 2,
        "x0": [0.0, 0.0],
        "min": "(1, 3)",
    },
    "Beale": {
        "expr": "(1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2",
        "n": 2,
        "x0": [1.0, 1.0],
        "min": "(3, 0.5)",
    },
    "Cuadrática 3D": {
        "expr": "x1**2 + 2*x2**2 + 3*x3**2 - x1*x2 + x2*x3",
        "n": 3,
        "x0": [1.0, 1.0, 1.0],
        "min": "(0, 0, 0)",
    },
}


def parse_function(expr_str: str, n_vars: int) -> ParsedFunction:
    """Parsea una expresión simbólica con variables x1, x2, ..., xn.

    Devuelve callables vectorizados para f, grad, hess más el LaTeX.
    """
    variables = sp.symbols(f"x1:{n_vars + 1}", real=True)

    # Diccionario seguro para sympify
    local_dict = {f"x{i+1}": variables[i] for i in range(n_vars)}
    # Funciones matemáticas comunes
    local_dict.update({
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "exp": sp.exp, "log": sp.log, "ln": sp.log,
        "sqrt": sp.sqrt, "abs": sp.Abs,
        "pi": sp.pi, "e": sp.E,
    })

    try:
        expr = sp.sympify(expr_str, locals=local_dict)
    except (sp.SympifyError, SyntaxError, TypeError) as e:
        raise ValueError(f"No se pudo interpretar la función: {e}") from e

    used = expr.free_symbols
    extra = used - set(variables)
    if extra:
        raise ValueError(
            f"La función usa variables no declaradas: {sorted(s.name for s in extra)}. "
            f"Usa solo x1..x{n_vars}."
        )

    grad_expr = sp.Matrix([sp.diff(expr, v) for v in variables])
    hess_expr = sp.hessian(expr, variables)

    f_lambd = sp.lambdify(variables, expr, modules="numpy")
    grad_lambd = sp.lambdify(variables, list(grad_expr), modules="numpy")
    hess_lambd = sp.lambdify(variables, hess_expr.tolist(), modules="numpy")

    def f(x: np.ndarray) -> float:
        return float(f_lambd(*x))

    def grad(x: np.ndarray) -> np.ndarray:
        return np.array(grad_lambd(*x), dtype=float).flatten()

    def hess(x: np.ndarray) -> np.ndarray:
        H = np.array(hess_lambd(*x), dtype=float)
        if H.ndim == 0:
            H = H.reshape(1, 1)
        return H

    return ParsedFunction(
        expr=expr,
        variables=list(variables),
        f=f,
        grad=grad,
        hess=hess,
        latex=sp.latex(expr),
        grad_latex=sp.latex(grad_expr),
        hess_latex=sp.latex(hess_expr),
    )
