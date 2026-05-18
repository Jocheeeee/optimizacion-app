"""Visualizaciones interactivas con Plotly."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go


_PALETTE = {
    "primary": "#0F62FE",
    "accent": "#FF7A1A",
    "path": "#E63946",
    "grid": "#E5E7EB",
    "text": "#1A1A1A",
}


def _layout(title: str, xaxis: str = "", yaxis: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(size=18, color=_PALETTE["text"])),
        xaxis=dict(title=xaxis, gridcolor=_PALETTE["grid"], zerolinecolor=_PALETTE["grid"]),
        yaxis=dict(title=yaxis, gridcolor=_PALETTE["grid"], zerolinecolor=_PALETTE["grid"]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color=_PALETTE["text"]),
        margin=dict(l=60, r=30, t=60, b=50),
        hovermode="x unified",
    )


def plot_convergence(history: List[dict]) -> go.Figure:
    """Error (norma del gradiente) vs número de iteración, escala log."""
    iters = [h["iter"] for h in history]
    grad_norms = [max(h["grad_norm"], 1e-16) for h in history]
    f_vals = [h["f"] for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters, y=grad_norms, mode="lines+markers",
        name="‖∇f(x_k)‖",
        line=dict(color=_PALETTE["primary"], width=2.5),
        marker=dict(size=6),
        hovertemplate="iter %{x}<br>‖∇f‖ = %{y:.3e}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=iters, y=f_vals, mode="lines+markers",
        name="f(x_k)", yaxis="y2",
        line=dict(color=_PALETTE["accent"], width=2.5, dash="dot"),
        marker=dict(size=6),
        hovertemplate="iter %{x}<br>f = %{y:.6g}<extra></extra>",
    ))

    layout = _layout("Convergencia", "Iteración", "‖∇f(x_k)‖  (escala log)")
    layout["yaxis"]["type"] = "log"
    layout["yaxis2"] = dict(
        title="f(x_k)", overlaying="y", side="right",
        gridcolor=_PALETTE["grid"], showgrid=False,
    )
    layout["legend"] = dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
    )
    fig.update_layout(**layout)
    return fig


def plot_contour_path(f, history: List[dict], margin: float = 1.5) -> go.Figure:
    """Curvas de nivel + trayectoria de iteraciones (solo n=2)."""
    pts = np.array([h["x"] for h in history])
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    dx = max(x_max - x_min, 1.0) * margin
    dy = max(y_max - y_min, 1.0) * margin
    cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)

    xs = np.linspace(cx - dx, cx + dx, 120)
    ys = np.linspace(cy - dy, cy + dy, 120)
    X, Y = np.meshgrid(xs, ys)
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
            except Exception:
                Z[i, j] = np.nan

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=Z,
        colorscale="Blues",
        contours=dict(showlines=True, coloring="heatmap"),
        opacity=0.85,
        colorbar=dict(title="f(x)"),
        hovertemplate="x1=%{x:.3f}<br>x2=%{y:.3f}<br>f=%{z:.4g}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=pts[:, 0], y=pts[:, 1],
        mode="lines+markers",
        name="Trayectoria",
        line=dict(color=_PALETTE["path"], width=2.5),
        marker=dict(size=7, color=_PALETTE["path"],
                    line=dict(color="white", width=1)),
        hovertemplate="iter %{pointNumber}<br>x1=%{x:.4f}<br>x2=%{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[pts[0, 0]], y=[pts[0, 1]],
        mode="markers", name="Inicio",
        marker=dict(size=14, color="#22c55e", symbol="circle",
                    line=dict(color="white", width=2)),
    ))
    fig.add_trace(go.Scatter(
        x=[pts[-1, 0]], y=[pts[-1, 1]],
        mode="markers", name="Mínimo",
        marker=dict(size=16, color="#facc15", symbol="star",
                    line=dict(color="#1A1A1A", width=1.5)),
    ))

    fig.update_layout(**_layout("Curvas de nivel y trayectoria", "x₁", "x₂"))
    fig.update_layout(hovermode="closest")
    return fig


def plot_surface_path(f, history: List[dict], margin: float = 1.5) -> go.Figure:
    """Superficie 3D + trayectoria (solo n=2)."""
    pts = np.array([h["x"] for h in history])
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    dx = max(x_max - x_min, 1.0) * margin
    dy = max(y_max - y_min, 1.0) * margin
    cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)

    xs = np.linspace(cx - dx, cx + dx, 60)
    ys = np.linspace(cy - dy, cy + dy, 60)
    X, Y = np.meshgrid(xs, ys)
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
            except Exception:
                Z[i, j] = np.nan

    z_path = np.array([h["f"] for h in history])

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=Z,
        colorscale="Blues", opacity=0.85,
        showscale=True, colorbar=dict(title="f(x)"),
    ))
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=z_path,
        mode="lines+markers", name="Trayectoria",
        line=dict(color=_PALETTE["path"], width=5),
        marker=dict(size=4, color=_PALETTE["path"]),
    ))
    fig.update_layout(
        title=dict(text="Superficie f(x₁, x₂) y trayectoria",
                   font=dict(size=18, color=_PALETTE["text"])),
        scene=dict(
            xaxis_title="x₁", yaxis_title="x₂", zaxis_title="f(x)",
            bgcolor="white",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color=_PALETTE["text"]),
    )
    return fig


def iterations_dataframe(history: List[dict]) -> pd.DataFrame:
    rows = []
    for h in history:
        row = {"iter": h["iter"]}
        for i, xi in enumerate(h["x"]):
            row[f"x{i+1}"] = xi
        row["f(x_k)"] = h["f"]
        row["||grad||"] = h["grad_norm"]
        row["alpha"] = h["alpha"]
        rows.append(row)
    return pd.DataFrame(rows)
