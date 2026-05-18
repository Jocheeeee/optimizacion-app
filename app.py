"""Aplicación web para optimización irrestricta.

Métodos disponibles:
    - Método del gradiente (descenso máximo)
    - Gradiente conjugado no lineal (Polak-Ribière+)
    - Método de Newton (amortiguado)

Todos usan búsqueda de línea con condiciones de Wolfe.
"""
from __future__ import annotations

import time

import numpy as np
import streamlit as st

from optimizers import (
    conjugate_gradient,
    gradient_descent,
    newton_method,
)
from utils import (
    PRESETS,
    iterations_dataframe,
    parse_function,
    plot_contour_path,
    plot_convergence,
    plot_surface_path,
)

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Optimización Irrestricta",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS personalizado para apariencia profesional
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        /* Header banner */
        .hero {
            background: linear-gradient(135deg, #0F62FE 0%, #6E36FA 100%);
            padding: 2.2rem 2.4rem;
            border-radius: 14px;
            color: white;
            margin-bottom: 1.6rem;
            box-shadow: 0 10px 30px rgba(15, 98, 254, 0.18);
        }
        .hero h1 {
            color: white;
            margin: 0 0 0.4rem 0;
            font-weight: 700;
            font-size: 2rem;
        }
        .hero p {
            margin: 0;
            opacity: 0.92;
            font-size: 1rem;
            font-weight: 400;
        }

        /* Métricas como cards */
        .metric-card {
            background: white;
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            height: 100%;
        }
        .metric-label {
            color: #6B7280;
            font-size: 0.82rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.4rem;
        }
        .metric-value {
            color: #0F172A;
            font-size: 1.45rem;
            font-weight: 700;
            word-break: break-all;
        }
        .metric-sub {
            color: #6B7280;
            font-size: 0.78rem;
            margin-top: 0.3rem;
        }

        /* Botón primario */
        .stButton>button {
            background: linear-gradient(135deg, #0F62FE 0%, #6E36FA 100%);
            color: white;
            border: none;
            font-weight: 600;
            padding: 0.6rem 1.4rem;
            border-radius: 10px;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(15, 98, 254, 0.25);
            color: white;
        }

        /* Sidebar look */
        [data-testid="stSidebar"] {
            background-color: #F8FAFC;
            border-right: 1px solid #E5E7EB;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 8px 8px 0 0;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background-color: #EEF4FF;
            color: #0F62FE;
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #94A3B8;
            font-size: 0.8rem;
            padding: 1.2rem 0 0.4rem 0;
        }

        /* Esconder marca Streamlit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📈 Optimización Irrestricta</h1>
        <p>Minimización de funciones por método del gradiente, método del gradiente
        conjugado y método de Newton — todos implementados con las condiciones de Wolfe
        (primera y segunda condición).</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar: parámetros de entrada
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuración")

    preset_names = ["— Personalizada —"] + list(PRESETS.keys())
    preset_choice = st.selectbox(
        "Función predefinida",
        preset_names,
        index=2,  # Rosenbrock por defecto
        help="Selecciona una función clásica de prueba o personalizada.",
    )

    if preset_choice != "— Personalizada —":
        preset = PRESETS[preset_choice]
        default_n = preset["n"]
        default_expr = preset["expr"]
        default_x0 = ", ".join(str(v) for v in preset["x0"])
        default_min = preset.get("min", "—")
    else:
        default_n = 2
        default_expr = "x1**2 + x2**2"
        default_x0 = "1, 1"
        default_min = None

    n_vars = st.number_input(
        "Número de variables (n)",
        min_value=1, max_value=10, value=default_n, step=1,
    )

    expr_str = st.text_area(
        "Función objetivo f(x)",
        value=default_expr,
        height=80,
        help="Usa x1, x2, …, xn. Soporta: + - * / **, sin, cos, exp, log, sqrt, pi, e.",
    )

    method = st.selectbox(
        "Método de optimización",
        ["Método del gradiente",
         "Método del gradiente conjugado",
         "Método de Newton"],
    )

    x0_str = st.text_input(
        "Punto de partida x₀ (separado por comas)",
        value=default_x0,
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        max_iter = st.number_input(
            "Máx. iteraciones", min_value=10, max_value=10000,
            value=500, step=50,
        )
    with col_s2:
        tol = st.number_input(
            "Tolerancia",
            min_value=1e-14, max_value=1e-1, value=1e-6,
            step=1e-7, format="%.1e",
        )

    st.markdown("**Condiciones de Wolfe**")
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        c1 = st.number_input(
            "c₁ (Armijo)",
            min_value=1e-6, max_value=0.5,
            value=1e-4, step=1e-5, format="%.1e",
            help="Típico: 1e-4",
        )
    with col_w2:
        default_c2 = 0.1 if method == "Método del gradiente conjugado" else 0.9
        c2 = st.number_input(
            "c₂ (curvatura)",
            min_value=0.001, max_value=0.999,
            value=default_c2, step=0.05, format="%.3f",
            help="Típico: 0.9 (Newton/gradiente), 0.1 (gradiente conjugado)",
        )

    if method == "Método del gradiente conjugado":
        cg_variant = st.radio(
            "Variante de β",
            ["PR+", "FR"],
            horizontal=True,
            help="PR+ = Polak-Ribière con reinicio (recomendado). FR = Fletcher-Reeves.",
        )
    else:
        cg_variant = "PR+"

    run = st.button("▶  Ejecutar optimización", use_container_width=True)


# ---------------------------------------------------------------------------
# Cuerpo principal
# ---------------------------------------------------------------------------
if not run:
    st.info(
        "👈 Configura los parámetros en el panel lateral y presiona "
        "**Ejecutar optimización**.\n\n"
        "**Sugerencia:** prueba con la función de Rosenbrock — un clásico difícil "
        "para el método del gradiente, mientras que Newton la resuelve en pocas iteraciones."
    )

    with st.expander("ℹ️ ¿Qué hace cada método?"):
        st.markdown(
            r"""
            - **Método del gradiente:** dirección $p_k = -\nabla f(x_k)$.
              Simple y robusto pero lento en valles estrechos (Rosenbrock).

            - **Método del gradiente conjugado:** combina la dirección del gradiente con
              la anterior usando el coeficiente $\beta_k$ (Polak-Ribière+).
              Más rápido que el gradiente, sin necesitar la Hessiana.

            - **Método de Newton:** dirección $p_k = -H^{-1}\nabla f(x_k)$.
              Convergencia cuadrática cerca del mínimo, pero requiere la
              Hessiana. Si no es definida positiva, agregamos $\tau I$
              (modificación tipo Levenberg).

            **Condiciones de Wolfe** garantizan suficiente decrecimiento
            (Armijo, c₁) y curvatura adecuada (c₂):
            $$
            f(x_k + \alpha p_k) \le f(x_k) + c_1 \alpha \nabla f^T p_k
            $$
            $$
            |\nabla f(x_k + \alpha p_k)^T p_k| \le c_2 |\nabla f(x_k)^T p_k|
            $$
            """
        )

    st.markdown(
        '<div class="footer">Hecho con Streamlit · SymPy · Plotly</div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ---------------------------------------------------------------------------
# Validación y ejecución
# ---------------------------------------------------------------------------
try:
    parsed = parse_function(expr_str, n_vars)
except ValueError as e:
    st.error(f"⚠️ Error al interpretar la función: {e}")
    st.stop()

try:
    x0 = np.array([float(v.strip()) for v in x0_str.split(",")], dtype=float)
except ValueError:
    st.error("⚠️ El punto de partida debe ser una lista de números separados por comas.")
    st.stop()

if len(x0) != n_vars:
    st.error(f"⚠️ El punto de partida tiene {len(x0)} valores pero n = {n_vars}.")
    st.stop()

# Mostrar la función parseada
with st.expander("🔎 Función, gradiente y Hessiana (revisión simbólica)", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Función:**")
        st.latex(f"f(x) = {parsed.latex}")
        st.markdown("**Gradiente:**")
        st.latex(f"\\nabla f(x) = {parsed.grad_latex}")
    with col_b:
        st.markdown("**Hessiana:**")
        st.latex(f"H(x) = {parsed.hess_latex}")

# Ejecutar
with st.spinner("Optimizando..."):
    t0 = time.perf_counter()
    if method == "Método del gradiente":
        result = gradient_descent(
            parsed.f, parsed.grad, x0,
            tol=tol, max_iter=int(max_iter), c1=c1, c2=c2,
        )
    elif method == "Método del gradiente conjugado":
        result = conjugate_gradient(
            parsed.f, parsed.grad, x0,
            tol=tol, max_iter=int(max_iter), c1=c1, c2=c2,
            variant=cg_variant,
        )
    else:
        result = newton_method(
            parsed.f, parsed.grad, parsed.hess, x0,
            tol=tol, max_iter=int(max_iter), c1=c1, c2=c2,
        )
    elapsed = time.perf_counter() - t0

# ---------------------------------------------------------------------------
# Resultados — tarjetas
# ---------------------------------------------------------------------------
st.markdown(f"### Resultados — {result['method']}")

x_min_str = ", ".join(f"{v:.6g}" for v in result["x_min"])
f_min_str = f"{result['f_min']:.6g}"
grad_norm_final = float(np.linalg.norm(parsed.grad(result["x_min"])))

c1_col, c2_col, c3_col, c4_col = st.columns(4)
with c1_col:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Punto mínimo x*</div>
            <div class="metric-value">({x_min_str})</div>
            <div class="metric-sub">solución encontrada</div>
        </div>
        """, unsafe_allow_html=True,
    )
with c2_col:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">f(x*)</div>
            <div class="metric-value">{f_min_str}</div>
            <div class="metric-sub">valor de la función</div>
        </div>
        """, unsafe_allow_html=True,
    )
with c3_col:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Iteraciones</div>
            <div class="metric-value">{result['iterations']}</div>
            <div class="metric-sub">tiempo: {elapsed*1000:.1f} ms</div>
        </div>
        """, unsafe_allow_html=True,
    )
with c4_col:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Criterio de parada</div>
            <div class="metric-value" style="font-size:1.05rem;">{result['stop_reason']}</div>
            <div class="metric-sub">‖∇f(x*)‖ = {grad_norm_final:.3e}</div>
        </div>
        """, unsafe_allow_html=True,
    )

if default_min:
    st.caption(f"🎯 Mínimo teórico de la función: **{default_min}**")

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs con visualizaciones
# ---------------------------------------------------------------------------
tab_conv, tab_path, tab_3d, tab_table = st.tabs([
    "📉 Convergencia",
    "🗺️ Curvas de nivel",
    "🌐 Superficie 3D",
    "📋 Tabla de iteraciones",
])

with tab_conv:
    st.plotly_chart(plot_convergence(result["history"]), use_container_width=True)

with tab_path:
    if n_vars == 2:
        st.plotly_chart(
            plot_contour_path(parsed.f, result["history"]),
            use_container_width=True,
        )
    else:
        st.info("Las curvas de nivel solo están disponibles para n = 2.")

with tab_3d:
    if n_vars == 2:
        st.plotly_chart(
            plot_surface_path(parsed.f, result["history"]),
            use_container_width=True,
        )
    else:
        st.info("La superficie 3D solo está disponible para n = 2.")

with tab_table:
    df = iterations_dataframe(result["history"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇  Descargar tabla (CSV)",
        data=csv,
        file_name=f"iteraciones_{method.replace(' ', '_')}.csv",
        mime="text/csv",
    )

st.markdown(
    '<div class="footer">Hecho con Streamlit · SymPy · Plotly · '
    'Métodos numéricos basados en Nocedal & Wright (2ed)</div>',
    unsafe_allow_html=True,
)
