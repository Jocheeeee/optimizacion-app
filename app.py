import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import re
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, convert_xor,
    split_symbols_custom, _token_splittable,
    implicit_multiplication, implicit_application
)

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Métodos de Optimización",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# ─── LOGIN SCREEN ──────────────────────────────────────────────────────────────
if not st.session_state.user_name:
    st.markdown("""
    <style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0d1117 !important;
    }
    [data-testid="stSidebar"] { display: none !important; }
    p, span, label { color: #e6edf3 !important; }
    input {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    [data-baseweb="input"] > div {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #8957e5) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        padding: 12px !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(88,166,255,0.35) !important;
    }
    [data-testid="stAlert"] {
        background: rgba(210,153,34,0.08) !important;
        border: 1px solid rgba(210,153,34,0.25) !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
                padding: 80px 20px 48px; text-align:center; min-height:55vh;">
        <div style="display:inline-flex; align-items:center; gap:8px;
                    background:rgba(88,166,255,0.1); border:1px solid rgba(88,166,255,0.3);
                    color:#58a6ff; font-size:11px; font-weight:700; letter-spacing:3px;
                    text-transform:uppercase; padding:5px 18px; border-radius:20px; margin-bottom:28px;">
            ⚡ &nbsp;Cálculo Numérico · Universidad
        </div>
        <div style="font-size:clamp(2.8rem,7vw,5rem); font-weight:900;
                    background:linear-gradient(90deg,#58a6ff 0%,#bc8cff 55%,#58a6ff 100%);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text; letter-spacing:-2.5px; line-height:1.05; margin-bottom:6px;">
            Métodos de
        </div>
        <div style="font-size:clamp(2.8rem,7vw,5rem); font-weight:900;
                    background:linear-gradient(90deg,#bc8cff 0%,#58a6ff 60%,#bc8cff 100%);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text; letter-spacing:-2.5px; line-height:1.05; margin-bottom:28px;">
            Optimización
        </div>
        <div style="width:64px; height:3px; background:linear-gradient(90deg,#58a6ff,#bc8cff);
                    border-radius:2px; margin:0 auto 24px;"></div>
        <p style="color:#8b949e; font-size:12px; letter-spacing:4px; text-transform:uppercase; margin-bottom:56px;">
            Gradiente &nbsp;·&nbsp; Newton &nbsp;·&nbsp; Gradiente Conjugado
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2, 1, 1.2])
    with col2:
        name_input = st.text_input("", placeholder="Tu nombre...", label_visibility="collapsed")
        if st.button("Entrar →", use_container_width=True):
            if name_input:
                st.session_state.user_name = name_input
                st.rerun()
            else:
                st.warning("Por favor, escribe un nombre primero.")

    st.markdown("""
    <div style="text-align:center; margin-top:48px;">
        <p style="color:#21262d; font-size:12px; letter-spacing:1px;">
            Gradiente Descendente · Método de Newton · Fletcher-Reeves · Polak-Ribière
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── GLOBAL DARK CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #0d1117 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {
    background-color: #010409 !important;
    border-right: 1px solid #21262d !important;
}
[data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"] {
    display: none !important;
}
.hero-header {
    background: linear-gradient(160deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 52px 52px 44px;
    margin-bottom: 36px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse at 15% 60%, rgba(88,166,255,0.13) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 40%, rgba(188,140,255,0.09) 0%, transparent 55%);
    pointer-events: none;
}
.hero-header::after {
    content: '\\2207  H  \\2202\\00B2  \\222B  \\03A3  \\03B1  \\03B2  \\03B5';
    position: absolute;
    right: 48px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 36px;
    color: rgba(88,166,255,0.05);
    letter-spacing: 18px;
    white-space: nowrap;
    pointer-events: none;
    user-select: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(88,166,255,0.1);
    border: 1px solid rgba(88,166,255,0.25);
    color: #58a6ff !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding: 5px 16px;
    border-radius: 20px;
    margin-bottom: 22px;
}
.hero-title {
    font-size: clamp(2rem, 4vw, 3.6rem) !important;
    font-weight: 900 !important;
    background: linear-gradient(90deg, #58a6ff 0%, #bc8cff 55%, #58a6ff 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 0 18px 0 !important;
    line-height: 1.1 !important;
    letter-spacing: -1.5px;
}
.hero-divider {
    width: 64px;
    height: 3px;
    background: linear-gradient(90deg, #58a6ff, #bc8cff);
    border-radius: 2px;
    margin: 0 0 18px 0;
}
.hero-subtitle {
    color: #8b949e !important;
    font-size: 12px !important;
    margin: 0 !important;
    letter-spacing: 3.5px;
    font-weight: 400 !important;
    text-transform: uppercase;
}
html, body, * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
/* Restaurar fuente de íconos Material (la flecha del expander, etc.) */
[data-testid="stIconMaterial"],
span[class*="material-symbols"], span[class*="material-icons"],
.material-icons, .material-symbols-rounded, .material-symbols-outlined {
    font-family: 'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important;
}
/* Ícono de ayuda (?) visible */
[data-testid="stTooltipHoverTarget"], [data-testid="stTooltipIcon"],
[data-testid="stTooltipHoverTarget"] svg, [data-testid="stTooltipIcon"] svg {
    color: #8b949e !important;
    fill: #8b949e !important;
    opacity: 1 !important;
}
[data-testid="stTooltipHoverTarget"]:hover, [data-testid="stTooltipIcon"]:hover {
    color: #58a6ff !important;
    fill: #58a6ff !important;
}
/* Globo de ayuda (popup): fondo oscuro + texto claro legible */
[data-testid="stTooltipContent"], [data-testid="stTooltipContent"] *,
div[data-baseweb="tooltip"], div[data-baseweb="tooltip"] * {
    background-color: #1f2733 !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}
p, span, label, li, .stMarkdown, [data-testid="stWidgetLabel"] p {
    color: #e6edf3 !important;
    font-size: 14px !important;
}
h1, h2, h3, h4 {
    color: #e6edf3 !important;
    font-weight: 700 !important;
}
h1 { font-size: 24px !important; }
h2 { font-size: 20px !important; }
h3 { font-size: 16px !important; margin-top: 28px !important; color: #c9d1d9 !important; }
h4 { font-size: 15px !important; color: #c9d1d9 !important; }
[data-testid="stForm"], .stFormCreator {
    background-color: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 12px !important;
    padding: 24px !important;
}
.instructions-box {
    background: linear-gradient(135deg, #161b22, #1a2035);
    border: 1px solid #21262d;
    border-left: 4px solid #58a6ff;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 28px;
}
.instructions-box h4 { color: #58a6ff !important; margin-bottom: 12px !important; }
.instructions-box ol { margin-bottom: 0; padding-left: 20px; }
.instructions-box li { color: #c9d1d9 !important; margin-bottom: 7px !important; }
.instructions-box code {
    background: rgba(88,166,255,0.1);
    color: #79c0ff !important;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 13px !important;
    border: 1px solid rgba(88,166,255,0.2);
}
.user-badge {
    background: rgba(88,166,255,0.06);
    border: 1px solid rgba(88,166,255,0.15);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 24px;
}
.user-badge span { color: #58a6ff !important; font-weight: 600 !important; }
.sidebar-title {
    font-size: 17px !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #58a6ff, #bc8cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2px !important;
    display: block;
}
.sidebar-subtitle {
    color: #8b949e !important;
    font-size: 11px !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 20px !important;
    display: block;
}
input, textarea {
    background-color: #161b22 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}
[data-baseweb="input"] > div {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}
[data-testid="stWidgetLabel"] *, input *, [data-baseweb="select"] span {
    color: #e6edf3 !important;
}
[data-baseweb="select"] svg { fill: #8b949e !important; }
[data-baseweb="select"] > div {
    background-color: #161b22 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}
[data-testid="stNumberInputStepUp"],
[data-testid="stNumberInputStepDown"],
div[data-baseweb="input"] button {
    background-color: #21262d !important;
    color: #8b949e !important;
    border: none !important;
    transition: all 0.15s ease !important;
}
[data-testid="stNumberInputStepUp"]:hover,
[data-testid="stNumberInputStepDown"]:hover,
div[data-baseweb="input"] button:hover {
    background-color: #58a6ff !important;
    color: #0d1117 !important;
}
[data-testid="stNumberInputStepUp"] svg,
[data-testid="stNumberInputStepDown"] svg,
div[data-baseweb="input"] button svg { fill: currentColor !important; }
div[data-baseweb="popover"], div[data-baseweb="popover"] *,
div[role="listbox"], div[role="listbox"] *,
ul[role="listbox"], ul[role="listbox"] *,
li[role="option"], li[role="option"] * {
    background-color: #161b22 !important;
    color: #e6edf3 !important;
}
li[role="option"]:hover, li[role="option"]:hover *,
div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li:hover * {
    background-color: #1f6feb !important;
    color: #ffffff !important;
}
.stButton > button {
    background: #161b22 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1f6feb, #8957e5) !important;
    color: #ffffff !important;
    border-color: transparent !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(88,166,255,0.25) !important;
}
.stButton > button:hover * { color: #ffffff !important; }
.stButton > button p { white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important; }
.resolve-btn .stButton > button {
    background: linear-gradient(135deg, #1f6feb 0%, #8957e5 100%) !important;
    color: #ffffff !important;
    border: none !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    padding: 14px 32px !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 20px rgba(88,166,255,0.25) !important;
    letter-spacing: 0.5px;
}
.resolve-btn .stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 32px rgba(88,166,255,0.42) !important;
    background: linear-gradient(135deg, #2d79f3 0%, #9d6af5 100%) !important;
}
[data-testid="stRadio"] label span { color: #c9d1d9 !important; }
[data-testid="stCheckbox"] label span { color: #c9d1d9 !important; }
[data-testid="stAlert"] { border-radius: 10px !important; }
[data-testid="stInfo"] {
    background: rgba(88,166,255,0.08) !important;
    border: 1px solid rgba(88,166,255,0.2) !important;
}
[data-testid="stInfo"] p { color: #79c0ff !important; }
[data-testid="stWarning"] {
    background: rgba(210,153,34,0.08) !important;
    border: 1px solid rgba(210,153,34,0.2) !important;
}
[data-testid="stError"] {
    background: rgba(218,54,51,0.08) !important;
    border: 1px solid rgba(218,54,51,0.2) !important;
}
[data-testid="stDataFrame"] {
    background-color: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* LATEX / FÓRMULAS (que no se corten los exponentes) */
[data-testid="stLatex"], .katex-display, .stMarkdown .katex-display {
    overflow: visible !important;
    padding: 6px 2px !important;
}
.katex { font-size: 1.05em !important; }

/* Tarjetas del modo examen (st.container border) */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #161b22 !important;
    border-radius: 10px !important;
}

[data-testid="stDialog"], div[data-baseweb="modal"], div[role="dialog"] {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
}
div[role="dialog"] > div { background-color: #161b22 !important; }
div[role="dialog"] h2, div[role="dialog"] h3 { color: #e6edf3 !important; }
div[role="dialog"] p, div[role="dialog"] li { color: #c9d1d9 !important; }
hr { border-color: #21262d !important; }
.section-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 18px;
    margin-top: 8px;
}
.section-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    background: linear-gradient(135deg, #1f6feb, #8957e5);
    border-radius: 7px;
    font-size: 13px;
    font-weight: 700;
    color: white !important;
    flex-shrink: 0;
}
.section-label-text {
    font-size: 17px !important;
    font-weight: 700 !important;
    color: #e6edf3 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR: USER INFO ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="user-badge">
        <span>👤 &nbsp;{st.session_state.user_name}</span>
    </div>
    """, unsafe_allow_html=True)

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
def format_latex_array(arr):
    if np.isscalar(arr):
        return f"{arr:.4f}"
    formatted = [f"{x:.4f}" for x in arr]
    return "[" + ", ".join(formatted) + "]^T"

def format_matrix_latex(mat):
    if np.isscalar(mat) or mat.ndim == 1:
        return format_latex_array(mat)
    res = r"\begin{bmatrix} "
    for row in mat:
        res += " & ".join([f"{x:.4f}" for x in row]) + r" \\ "
    res += r"\end{bmatrix}"
    return res

def parse_function(func_str, vars_list):
    var_names = [str(v) for v in vars_list]
    try:
        s = func_str.replace('^', '**')
        s = re.sub(r'\bln\b', 'log', s)
        s = re.sub(r'(\d),(\d)', r'\1.\2', s)
        varset = set(var_names)
        def can_split(symbol):
            if symbol in varset:
                return False
            if any(c.isdigit() for c in symbol):
                return False
            return _token_splittable(symbol)
        T = (standard_transformations + (
            convert_xor, split_symbols_custom(can_split),
            implicit_multiplication, implicit_application))
        ld = {v: sp.Symbol(v) for v in var_names}
        return parse_expr(s, transformations=T, local_dict=ld)
    except Exception:
        try:
            s = func_str.replace('^', '**').replace('ln', 'log')
            s = re.sub(r'(\d),(\d)', r'\1.\2', s)
            s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)
            return sp.sympify(s)
        except Exception:
            return None

def parse_start_point(raw):
    raw = raw.strip()
    parts = raw.split(';') if ';' in raw else re.split(r'\s+', raw)
    vals = []
    for p in parts:
        p = p.strip().replace(',', '.')
        p = re.sub(r'[^0-9.\-]', '', p)
        if p:
            vals.append(float(p))
    return vals

def pretty_latex(s):
    """Convierte el texto del usuario a LaTeX RESPETANDO su orden exacto
    (no usa SymPy, así no reordena los términos)."""
    s = s.replace(' ', '')
    s = re.sub(r'(\d),(\d)', r'\1.\2', s)
    s = s.replace('**', '^')
    s = re.sub(r'\^(\-?\d+\.?\d*)', r'^{\1}', s)
    s = re.sub(r'\bln\b', r'\\ln ', s)
    s = re.sub(r'\bexp\b', r'\\exp ', s)
    s = re.sub(r'\bsqrt\b', r'\\sqrt', s)
    s = re.sub(r'\b(sin|cos|tan|log)\b', r'\\\1 ', s)
    s = s.replace('*', r' \cdot ')
    return s

def compute_gradient(expr, variables):
    return [sp.diff(expr, var) for var in variables]

def compute_hessian(expr, variables):
    return sp.hessian(expr, variables)

def calcular_error(curr_x, prev_x, norm_type):
    if norm_type == "L_infinito (Máximo)":
        num = np.linalg.norm(curr_x - prev_x, ord=np.inf)
        den = np.linalg.norm(curr_x, ord=np.inf)
    elif norm_type == "L1 (Manhattan)":
        num = np.linalg.norm(curr_x - prev_x, ord=1)
        den = np.linalg.norm(curr_x, ord=1)
    else:
        num = np.linalg.norm(curr_x - prev_x)
        den = np.linalg.norm(curr_x)
    return num / (den if den != 0 else 1e-8)

def evaluate_func_safe(f_lambdified, x, vars_sym):
    try:
        val = f_lambdified(*x) if len(vars_sym) > 1 else f_lambdified(x[0])
        if np.isnan(val) or np.isinf(val):
            return 1e9
        return val
    except Exception:
        return 1e9

def run_gradient_descent(expr, vars_sym, x0, alpha_type, alpha_val, wolfe_params, max_iter, tol, norm_type):
    history = []
    backtrack_log = []
    exam_log = []

    f_lambdified = sp.lambdify(vars_sym, expr, 'numpy')
    grad_exprs = compute_gradient(expr, vars_sym)
    grad_lambdified = [sp.lambdify(vars_sym, g, 'numpy') for g in grad_exprs]
    curr_x = np.array(x0, dtype=float)

    for k in range(max_iter + 1):
        f_val = evaluate_func_safe(f_lambdified, curr_x, vars_sym)
        grad_val = (np.array([g(*curr_x) for g in grad_lambdified]) if len(vars_sym) > 1
                    else np.array([grad_lambdified[0](curr_x[0])]))

        rel_error = 0.0
        if k > 0:
            prev_x = np.array([history[-1][f'{v}'] for v in vars_sym])
            rel_error = calcular_error(curr_x, prev_x, norm_type)

        entry = {'Iteración': k, 'C(x)': f_val, '||∇C(x)||': np.linalg.norm(grad_val),
                 'Error Rel. (%)': rel_error * 100}
        for i, val in enumerate(curr_x): entry[f'{vars_sym[i]}'] = val
        for i, val in enumerate(grad_val): entry[f'g_{vars_sym[i]}'] = val
        history.append(entry)

        if k > 0 and rel_error < tol:
            break

        if k < max_iter:
            direction = -grad_val
            alpha = alpha_val

            if alpha_type == "Wolfe (Armijo)":
                alpha = wolfe_params['alpha_init']
                c1 = wolfe_params['c1']
                rho = wolfe_params['rho']
                for intento in range(50):
                    new_x = curr_x + alpha * direction
                    f_new = evaluate_func_safe(f_lambdified, new_x, vars_sym)
                    limite_armijo = f_val + c1 * alpha * np.dot(grad_val, direction)
                    armijo_cumple = f_new <= limite_armijo
                    if k < 3:
                        backtrack_log.append({
                            'Iteración k': k, 'Intento': intento + 1, 'Alfa (α)': alpha,
                            'C(x + αd)': f_new, 'Cota de Armijo': limite_armijo, 'Cumple': armijo_cumple
                        })
                    if not armijo_cumple:
                        alpha *= rho
                    else:
                        break

            if k < 3:
                step_info = ""
                step_info += f"- **Punto actual ($x_{k}$):** ${format_latex_array(curr_x)}$\n"
                step_info += f"- **Evaluación del Gradiente $\\nabla C(x_{k})$:** ${format_latex_array(grad_val)}$\n"
                step_info += f"- **Dirección de descenso ($d_{k} = -\\nabla C$):** ${format_latex_array(direction)}$\n"
                step_info += f"- **Tamaño de paso ($\\alpha_k$):** `{alpha:.4f}`\n"
                step_info += f"- **Actualización:** $x_{k+1} = x_{k} + \\alpha_k d_{k} = {format_latex_array(curr_x + alpha * direction)}$\n"
                exam_log.append(step_info)

            curr_x = curr_x - alpha * grad_val

    return pd.DataFrame(history), pd.DataFrame(backtrack_log), grad_exprs, exam_log

def run_newton_method(expr, vars_sym, x0, max_iter, tol, norm_type):
    history = []
    exam_log = []

    f_lambdified = sp.lambdify(vars_sym, expr, 'numpy')
    grad_exprs = compute_gradient(expr, vars_sym)
    hess_expr = compute_hessian(expr, variables=vars_sym)
    grad_lambdified = [sp.lambdify(vars_sym, g, 'numpy') for g in grad_exprs]
    hess_lambdified = sp.lambdify(vars_sym, hess_expr, 'numpy')
    curr_x = np.array(x0, dtype=float)

    for k in range(max_iter + 1):
        f_val = evaluate_func_safe(f_lambdified, curr_x, vars_sym)
        grad_val = (np.array([g(*curr_x) for g in grad_lambdified]) if len(vars_sym) > 1
                    else np.array([grad_lambdified[0](curr_x[0])]))

        rel_error = 0.0
        if k > 0:
            prev_x = np.array([history[-1][f'{v}'] for v in vars_sym])
            rel_error = calcular_error(curr_x, prev_x, norm_type)

        accion_str = ""
        hess_inv = None
        hess_val = None

        if k < max_iter:
            try:
                n = len(vars_sym)
                hess_val = np.array(hess_lambdified(*curr_x), dtype=float).reshape(n, n)
                eigvals = np.linalg.eigvals(hess_val)
                if np.any(eigvals <= 1e-8):
                    hess_inv = np.eye(len(vars_sym))
                    accion_str = "Falla matriz positiva -> Uso Matriz Identidad"
                else:
                    hess_inv = np.linalg.inv(hess_val)
                    accion_str = "Hessiana OK -> Newton Estándar"
            except Exception:
                hess_inv = np.eye(len(vars_sym))
                accion_str = "Error al invertir -> Uso Matriz Identidad"
                hess_val = np.eye(len(vars_sym))

        if k > 0 and rel_error < tol:
            accion_str = "Óptimo alcanzado"

        entry = {'Iteración': k, 'C(x)': f_val, '||∇C(x)||': np.linalg.norm(grad_val),
                 'Error Rel. (%)': rel_error * 100, 'Acción para sig. paso': accion_str}
        for i, val in enumerate(curr_x): entry[f'{vars_sym[i]}'] = val
        for i, val in enumerate(grad_val): entry[f'g_{vars_sym[i]}'] = val
        history.append(entry)

        if k > 0 and rel_error < tol:
            break

        if k < max_iter and hess_inv is not None:
            if k < 3:
                step_info = ""
                step_info += f"- **Punto actual ($x_{k}$):** ${format_latex_array(curr_x)}$\n"
                step_info += f"- **Evaluación del Gradiente $\\nabla C(x_{k})$:** ${format_latex_array(grad_val)}$\n"
                step_info += f"- **Matriz Hessiana $H(x_{k})$:**\n $${format_matrix_latex(hess_val)}$$\n"
                step_info += f"- **Inversa de la Hessiana $H(x_{k})^{{-1}}$:** *(Nota: {accion_str})*\n $${format_matrix_latex(hess_inv)}$$\n"
                d_k = -hess_inv.dot(grad_val)
                step_info += f"- **Paso de Newton ($d_{k} = -H^{{-1}}\\nabla C$):** ${format_latex_array(d_k)}$\n"
                step_info += f"- **Actualización:** $x_{k+1} = x_{k} + d_{k} = {format_latex_array(curr_x + d_k)}$\n"
                exam_log.append(step_info)

            curr_x = curr_x - hess_inv.dot(grad_val)

    return pd.DataFrame(history), exam_log

def run_conjugate_gradient(expr, vars_sym, x0, alpha_type, alpha_val, max_iter, tol, norm_type):
    history = []
    exam_log = []

    f_lambdified = sp.lambdify(vars_sym, expr, 'numpy')
    grad_exprs = compute_gradient(expr, vars_sym)
    grad_lambdified = [sp.lambdify(vars_sym, g, 'numpy') for g in grad_exprs]
    curr_x = np.array(x0, dtype=float)
    grad_val = (np.array([g(*curr_x) for g in grad_lambdified]) if len(vars_sym) > 1
                else np.array([grad_lambdified[0](curr_x[0])]))
    p = -grad_val.copy()

    for k in range(max_iter + 1):
        f_val = evaluate_func_safe(f_lambdified, curr_x, vars_sym)
        grad_val = (np.array([g(*curr_x) for g in grad_lambdified]) if len(vars_sym) > 1
                    else np.array([grad_lambdified[0](curr_x[0])]))
        norm_grad = np.linalg.norm(grad_val)

        rel_error = 0.0
        if k > 0:
            prev_x = np.array([history[-1][f'{v}'] for v in vars_sym])
            rel_error = calcular_error(curr_x, prev_x, norm_type)

        entry = {'Iteración': k, 'C(x)': f_val, '||∇C(x)||': norm_grad, 'Error Rel. (%)': rel_error * 100}
        for i, val in enumerate(curr_x): entry[f'{vars_sym[i]}'] = val
        for i, val in enumerate(grad_val): entry[f'g_{vars_sym[i]}'] = val
        history.append(entry)

        if k > 0 and (rel_error < tol or norm_grad < 1e-6):
            break

        if k < max_iter:
            if np.dot(grad_val, p) >= 0:
                p = -grad_val

            if alpha_type == "Fijo":
                alpha = alpha_val
            else:
                alpha = alpha_val
                c1 = 1e-4
                rho = 0.5
                for _ in range(50):
                    new_x = curr_x + alpha * p
                    f_new = evaluate_func_safe(f_lambdified, new_x, vars_sym)
                    if f_new <= f_val + c1 * alpha * np.dot(grad_val, p):
                        break
                    alpha *= rho

            next_x = curr_x + alpha * p
            grad_next_val = (np.array([g(*next_x) for g in grad_lambdified]) if len(vars_sym) > 1
                             else np.array([grad_lambdified[0](next_x[0])]))
            denom = np.dot(grad_val, grad_val)
            beta_fr = 0.0 if denom < 1e-12 else np.dot(grad_next_val, grad_next_val) / denom
            fr_exploto = beta_fr > 10

            if fr_exploto:
                yk = grad_next_val - grad_val
                beta_pr = np.dot(grad_next_val, yk) / denom
                beta = max(0.0, beta_pr)
                p_next = -grad_next_val + beta * p
            else:
                beta = beta_fr
                p_next = -grad_next_val + beta * p

            if k < 3:
                step_info = ""
                step_info += f"- **Punto actual ($x_{k}$):** ${format_latex_array(curr_x)}$\n"
                step_info += f"- **Gradiente $\\nabla C(x_{k})$:** ${format_latex_array(grad_val)}$\n"
                step_info += f"- **Dirección conjugada ($p_{k}$):** ${format_latex_array(p)}$\n"
                step_info += f"- **Tamaño de paso ($\\alpha_k$):** `{alpha:.4f}`\n"
                step_info += f"- **Nuevo punto ($x_{k+1} = x_k + \\alpha_k p_k$):** ${format_latex_array(next_x)}$\n"
                step_info += f"- **Nuevo Gradiente $\\nabla C(x_{k+1})$:** ${format_latex_array(grad_next_val)}$\n"
                step_info += f"- **Factor de conjugación ($\\beta_k$):** `{beta:.4f}`\n"
                if fr_exploto:
                    step_info += "\n\n⚠️ **Observación Académica**\n\n"
                    step_info += (
                        "No es recomendable continuar utilizando Fletcher-Reeves en esta iteración. "
                        "El factor de conjugación β obtuvo un valor excesivamente grande, provocando "
                        "una dirección conjugada muy agresiva y potencialmente inestable. "
                        "Por esta razón, el algoritmo cambia automáticamente a "
                        "**Polak-Ribière**, más robusto en funciones no cuadráticas.\n\n"
                        f"**β (Fletcher-Reeves original) = {beta_fr:.4f}**"
                    )
                exam_log.append(step_info)

            p = p_next
            curr_x = next_x

    return pd.DataFrame(history), exam_log

# ─── DIALOG ────────────────────────────────────────────────────────────────────
@st.dialog("📖 Definición del Método")
def mostrar_metodo():
    st.markdown("""
    <style>
    div[role="dialog"] { background-color: #161b22 !important; }
    div[role="dialog"] > div { background-color: #161b22 !important; }
    div[role="dialog"] h2, div[role="dialog"] h3 { color: #e6edf3 !important; }
    div[role="dialog"] p, div[role="dialog"] li { color: #c9d1d9 !important; }
    </style>
    """, unsafe_allow_html=True)

    metodo = st.session_state.metodo_info
    if metodo == "gradiente":
        st.subheader("📉 Método del Gradiente")
        st.write("""
        Busca mínimos moviéndose en la dirección opuesta al gradiente.
        • Fácil de implementar.
        • Utiliza derivadas de primer orden.
        • Puede requerir muchas iteraciones.
        """)
    elif metodo == "newton":
        st.subheader("🚀 Método de Newton")
        st.write("""
        Utiliza gradiente y Hessiana para aproximar rápidamente el óptimo.
        • Convergencia rápida (cuadrática cerca del óptimo).
        • Usa derivadas de segundo orden.
        • Si la Hessiana no es definida positiva, se usa Matriz Identidad automáticamente.
        """)
    elif metodo == "conjugado":
        st.subheader("🎯 Gradiente Conjugado")
        st.write("""
        Genera direcciones conjugadas para evitar recorrer caminos repetidos.
        • Más eficiente que gradiente clásico.
        • Muy útil en problemas grandes.
        • No requiere Hessiana completa.
        """)
    if st.button("Cerrar"):
        del st.session_state["metodo_info"]
        st.rerun()

# ─── MAIN APP ──────────────────────────────────────────────────────────────────
def main_app():
    if 'func_text' not in st.session_state:
        st.session_state.func_text = "x**2 + y**2"
    if 'vars_text' not in st.session_state:
        st.session_state.vars_text = "x, y"
    if 'start_text' not in st.session_state:
        st.session_state.start_text = "1 ; 1"
    if 'func_ctr' not in st.session_state:
        st.session_state.func_ctr = 0

    with st.sidebar:
        st.markdown("""
        <span class="sidebar-title">💡 Diccionario</span>
        <span class="sidebar-subtitle">de Métodos</span>
        """, unsafe_allow_html=True)
        if st.button("📉 Método del Gradiente"):
            st.session_state.metodo_info = "gradiente"
            mostrar_metodo()
        if st.button("🚀 Método de Newton"):
            st.session_state.metodo_info = "newton"
            mostrar_metodo()
        if st.button("🎯 Gradiente Conjugado"):
            st.session_state.metodo_info = "conjugado"
            mostrar_metodo()

    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">⚡ &nbsp;Cálculo Numérico · Optimización Numérica</div>
        <h1 class="hero-title">Métodos de Optimización</h1>
        <div class="hero-divider"></div>
        <p class="hero-subtitle">Gradiente Descendente &nbsp;·&nbsp; Newton &nbsp;·&nbsp; Gradiente Conjugado</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="instructions-box">
        <h4>📖 Resuelve tus Guías de Estudio</h4>
        <ol>
            <li><strong>Variables Flexibles:</strong> Escribe <code>x, y</code> o <code>x1, x2</code> dependiendo de tu problema.</li>
            <li><strong>Manejo de logaritmos:</strong> Usa <code>ln()</code> o <code>log()</code> sin problema.</li>
            <li><strong>Activa el "Modo Examen"</strong> para ver la traza paso a paso (ideal para aprender).</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-badge">
        <span class="section-num">1</span>
        <span class="section-label-text">Definición del Problema</span>
    </div>
    """, unsafe_allow_html=True)

    _examples = {
        "— Elige un ejemplo o escribe el tuyo —": None,
        "Cuadrática simple · x² + y²": ("x, y", "x**2 + y**2", "1 ; 1"),
        "Gradiente 1D · x² (α=0,3)": ("x", "x**2", "-1,5"),
        "Clase 6 · Ej5 (Gradiente+Armijo)": ("x, y", "2*x**2 - 4*x*y + y**4 + 5*y**2 - 10*y", "0 ; 0"),
        "Clase 6 · Ej6 (Gradiente+Armijo)": ("x, y", "x**2 + y**4 - 2*x - 4*y", "0 ; 0"),
        "Clase 6 · Ej7 (Gradiente Conjugado)": ("x, y", "x - y + 2*x**2 + 2*x*y + y**2", "0 ; 0"),
        "Prueba S3 · Newton 1D (eˣ - x²/2 - x)": ("x", "exp(x) - x**2/2 - x", "1"),
        "Rosenbrock · (1-x)² + 100(y-x²)²": ("x, y", "(1-x)**2 + 100*(y - x**2)**2", "-1 ; 1"),
        "Logarítmica · ln(x²+y²) - 2xy": ("x, y", "ln(x**2 + y**2) - 2*x*y", "-1 ; 0"),
    }
    _ex_choice = st.selectbox("📚 Ejemplos precargados:", list(_examples.keys()))
    if _examples[_ex_choice] is not None and st.session_state.get('last_example') != _ex_choice:
        _v, _f, _s = _examples[_ex_choice]
        st.session_state.vars_text = _v
        st.session_state.func_text = _f
        st.session_state.start_text = _s
        st.session_state.last_example = _ex_choice
        st.session_state.func_ctr += 1
        st.rerun()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        vars_input = st.text_input(
            "Variables",
            value=st.session_state.vars_text,
            key=f'vars_w_{st.session_state.func_ctr}',
            placeholder="x, y",
            help="Sepáralas por coma. Ejemplos: x, y  ·  x1, x2"
        )
        st.session_state.vars_text = vars_input
        vars_names = [v.strip() for v in vars_input.split(',')]
    with col2:
        func_input = st.text_input(
            f"Función C({', '.join(vars_names)})",
            value=st.session_state.func_text,
            key=f'func_w_{st.session_state.func_ctr}',
            placeholder="x**2 + y**2",
            help="Potencias con ** o ^ · decimales con coma · funciones ln(), exp(), sqrt()"
        )
        st.session_state.func_text = func_input
    with col3:
        start_point = st.text_input(
            "Punto inicial",
            value=st.session_state.start_text,
            key=f'start_w_{st.session_state.func_ctr}',
            placeholder="1 ; 1",
            help="Separa las coordenadas con punto y coma (;). Ejemplo: 1,5 ; -0,5"
        )
        st.session_state.start_text = start_point

    # ── PREVIEW (a todo el ancho, debajo de los campos) ────────────────────────
    try:
        _syms_p = sp.symbols(' '.join(vars_names))
        _syms_p = [_syms_p] if len(vars_names) == 1 else list(_syms_p)
        _expr_p = parse_function(func_input, _syms_p)
        if _expr_p is not None:
            st.markdown('<p style="color:#3fb950; font-size:13px; font-weight:600; margin:10px 0 2px;">✅ Función reconocida:</p>', unsafe_allow_html=True)
            st.latex(pretty_latex(func_input))
        elif func_input.strip():
            st.markdown('<p style="color:#f85149; font-size:13px; margin-top:10px;">⚠ Función no reconocida — revisa la sintaxis</p>', unsafe_allow_html=True)
    except Exception:
        pass

    with st.expander("¿Cómo escribir la función?  —  guía rápida de sintaxis"):
        st.markdown("""
        <div style="font-size:13px; line-height:1.9; color:#c9d1d9;">
        <b style="color:#58a6ff;">Potencias</b> &nbsp;→&nbsp; <code>x**2</code> (x al cuadrado), <code>y**4</code>, <code>x**0.5</code><br>
        <b style="color:#f97316;">Multiplicación</b> &nbsp;→&nbsp; puedes pegar las variables:
        <code>4xy</code> = <code>4*x*y</code> · <code>2x1</code> = <code>2*x1</code>
        (el <code>*</code> es opcional). Para dos subíndices juntos usa <code>x1*x2</code>.<br>
        <b style="color:#bc8cff;">Funciones</b> &nbsp;→&nbsp;
        <code>ln(x)</code> o <code>log(x)</code>, <code>exp(x)</code> (= eˣ),
        <code>sqrt(x)</code> (= raíz), <code>sin()</code>, <code>cos()</code>, <code>tan()</code>, <code>abs()</code><br>
        <b style="color:#3fb950;">Constantes</b> &nbsp;→&nbsp; <code>pi</code> (π), <code>E</code> (número e)<br>
        <b style="color:#e3b341;">Decimales</b> &nbsp;→&nbsp; usa <b>coma</b>:
        <code>0,5*x</code> · <code>2,75*y</code> (también acepta punto <code>0.5</code>)<br>
        <b style="color:#8b949e;">Variables</b> &nbsp;→&nbsp; separadas por coma en su campo
        (<code>x, y</code> o <code>x1, x2</code>). El <b>punto inicial</b> separa coordenadas
        con <b>punto y coma</b>: <code>-1 ; 0</code> &nbsp;o&nbsp; <code>1,5 ; -0,5</code>
        <hr style="border-color:#21262d; margin:10px 0;">
        <span style="color:#8b949e;">Ejemplo completo:</span>
        &nbsp; función <code>2*x**2 - 4*x*y + y**4 + 5*y**2 - 10*y</code>,
        punto <code>0 ; 0</code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin:20px 0 6px;">
        <span style="font-size:11px; font-weight:700; color:#8b949e; letter-spacing:2px; text-transform:uppercase;">
            🎹 Teclado Matemático — clic para insertar en la función
        </span>
    </div>
    """, unsafe_allow_html=True)

    def _insert(code):
        if code == "BACK":
            st.session_state.func_text = st.session_state.func_text[:-1]
        elif code == "CLEAR":
            st.session_state.func_text = ""
        else:
            st.session_state.func_text += code
        st.session_state.func_ctr += 1
        st.rerun()

    _KB_PER_ROW = 9

    def _render_keys(items, prefix):
        for _start in range(0, len(items), _KB_PER_ROW):
            _chunk = items[_start:_start + _KB_PER_ROW]
            _cols = st.columns(_KB_PER_ROW)
            for _ci, (_d, _code) in enumerate(_chunk):
                with _cols[_ci]:
                    if st.button(_d, key=f"{prefix}_{_start + _ci}", use_container_width=True):
                        _insert(_code)

    st.markdown('<p style="color:#58a6ff; font-size:10px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; margin:8px 0 2px;">Variables</p>', unsafe_allow_html=True)
    _user_vars = [(v, v) for v in vars_names]
    _extra_vars = [(v, v) for v in ['x', 'y', 'z', 'x1', 'x2', 'y1', 'y2'] if v not in vars_names]
    _render_keys((_user_vars + _extra_vars)[:9], "kv")

    st.markdown('<p style="color:#f97316; font-size:10px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; margin:8px 0 2px;">Operadores</p>', unsafe_allow_html=True)
    _ops = [("(", "("), (")", ")"), ("x²", "**2"), ("x³", "**3"), ("xⁿ", "**"),
            ("×", "*"), ("÷", "/"), ("+", "+"), ("−", "-"), ("√", "sqrt("), ("⌫", "BACK")]
    _render_keys(_ops, "ko")

    st.markdown('<p style="color:#bc8cff; font-size:10px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; margin:8px 0 2px;">Funciones</p>', unsafe_allow_html=True)
    _fns = [("ln", "log("), ("log", "log10("), ("eˣ", "exp("),
            ("sin", "sin("), ("cos", "cos("), ("tan", "tan("),
            ("|x|", "abs("), ("π", "3.14159265"), ("e", "2.71828182"), ("🗑", "CLEAR")]
    _render_keys(_fns, "kf")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-badge" style="margin-top:24px;">
        <span class="section-num">2</span>
        <span class="section-label-text">Parámetros del Algoritmo</span>
    </div>
    """, unsafe_allow_html=True)

    if 'method_idx' not in st.session_state:
        st.session_state.method_idx = 0

    _methods = [
        ("Gradiente",  "📉", "Método del Gradiente",             "#58a6ff", "rgba(88,166,255,0.08)",  "1er orden · Robusto · Fijo/Wolfe"),
        ("Newton",     "🚀", "Método de Newton",                 "#bc8cff", "rgba(188,140,255,0.08)", "2do orden · Convergencia rápida"),
        ("Conjugado",  "🎯", "Método del Gradiente Conjugado",   "#3fb950", "rgba(63,185,80,0.08)",   "Sin Hessiana · Eficiente"),
    ]

    card_cols = st.columns(3)
    for i, (col, (short, icon, full, color, bg, desc)) in enumerate(zip(card_cols, _methods)):
        with col:
            is_sel = st.session_state.method_idx == i
            st.markdown(f"""
            <div style="background:{'rgba(88,166,255,0.08)' if i==0 and is_sel else 'rgba(188,140,255,0.08)' if i==1 and is_sel else 'rgba(63,185,80,0.08)' if i==2 and is_sel else '#0d1117'};
                        border:{'2px solid '+color if is_sel else '1px solid #21262d'};
                        border-radius:12px; padding:22px 16px 8px; text-align:center; margin-bottom:6px;">
                <div style="font-size:32px; margin-bottom:8px;">{icon}</div>
                <div style="font-weight:700; color:#e6edf3; font-size:15px; margin-bottom:5px;">{short}</div>
                <div style="color:#8b949e; font-size:11px; line-height:1.6; margin-bottom:10px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("✓ Activo" if is_sel else "Seleccionar", key=f"mcard_{i}", use_container_width=True):
                st.session_state.method_idx = i
                st.rerun()

    method = _methods[st.session_state.method_idx][2]

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Fila 1: parámetros generales ───────────────────────────────────────────
    gen1, gen2, gen3 = st.columns(3)
    with gen1:
        max_iter = st.number_input("Iteraciones máximas", value=10, min_value=1)
    with gen2:
        tolerancia = st.number_input("Tolerancia", value=0.001, format="%.4f",
                                     help="El algoritmo se detiene cuando el error relativo es menor a este valor.")
    with gen3:
        norm_type = st.selectbox("Norma para Error Relativo",
                                 ["L_infinito (Máximo)", "L2 (Euclidiana)", "L1 (Manhattan)"])

    st.markdown("""
    <p style="font-size:13px; font-weight:700; color:#58a6ff; letter-spacing:1px;
              text-transform:uppercase; margin:18px 0 4px;">⚙️ Cálculo del tamaño de paso (α)</p>
    """, unsafe_allow_html=True)

    # ── Fila 2: tamaño de paso según método ────────────────────────────────────
    if method == "Método de Newton":
        st.info("El Método de Newton calcula su propio paso. Si la Hessiana falla (no definida positiva), usa Matriz Identidad automáticamente.")
        alpha_type = "Newton"
    elif method == "Método del Gradiente Conjugado":
        alpha_type = st.radio("Estrategia:", ["Fijo", "Búsqueda de línea (Armijo)"], horizontal=True)
        pc1, _pc2, _pc3 = st.columns(3)
        with pc1:
            if alpha_type == "Fijo":
                cg_alpha_val = st.number_input("Valor de α", value=0.01, format="%.4f")
            else:
                cg_alpha_val = st.number_input("α inicial", value=1.0, format="%.4f")
    else:  # Gradiente
        alpha_type = st.radio("Estrategia:", ["Fijo", "Wolfe (Armijo)"], index=1, horizontal=True)
        if alpha_type == "Fijo":
            pc1, _pc2, _pc3 = st.columns(3)
            with pc1:
                alpha_val = st.number_input("Valor de α", value=0.01, format="%.4f")
        else:
            alpha_val = 0.0
            wolfe_params = {}
            wp1, wp2, wp3, wp4 = st.columns(4)
            with wp1:
                wolfe_params['alpha_init'] = st.number_input("α₀ inicial", value=0.5, format="%.4f",
                                                             help="Paso inicial antes de reducir con backtracking.")
            with wp2:
                wolfe_params['rho'] = st.number_input("ρ (reducción)", value=0.5, format="%.4f",
                                                      help="Factor por el que se reduce α en cada intento.")
            with wp3:
                wolfe_params['c1'] = st.number_input("β (Armijo)", value=0.25, format="%.4f",
                                                    help="Constante de la condición de descenso suficiente (Armijo).")
            with wp4:
                wolfe_sigma = st.number_input("σ (Curvatura)", value=0.5, format="%.4f",
                                             help="Constante de la condición de curvatura (2da condición de Wolfe).")

    st.markdown("<br>", unsafe_allow_html=True)
    show_exam_mode = st.checkbox("📝 Mostrar Detalles en Modo Examen", value=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        st.markdown('<div class="resolve-btn">', unsafe_allow_html=True)
        execute = st.button("⚡ Resolver Problema")
        st.markdown('</div>', unsafe_allow_html=True)

    if execute:
        st.markdown("""
        <div class="section-badge" style="margin-top:32px;">
            <span class="section-num">3</span>
            <span class="section-label-text">Resultados y Análisis</span>
        </div>
        """, unsafe_allow_html=True)

        vars_sym = sp.symbols(' '.join(vars_names))
        if len(vars_names) == 1:
            vars_sym = [vars_sym]
        expr = parse_function(func_input, vars_sym)

        try:
            x0 = parse_start_point(start_point)

            if expr is not None and len(x0) == len(vars_names):
                exam_log = []

                if method == "Método del Gradiente":
                    results, bt_log, grad_exprs, exam_log = run_gradient_descent(
                        expr, vars_sym, x0, alpha_type, alpha_val,
                        wolfe_params if alpha_type == "Wolfe (Armijo)" else None,
                        int(max_iter), tolerancia, norm_type
                    )
                elif method == "Método de Newton":
                    results, exam_log = run_newton_method(
                        expr, vars_sym, x0, int(max_iter), tolerancia, norm_type
                    )
                else:
                    results, exam_log = run_conjugate_gradient(
                        expr, vars_sym, x0, alpha_type,
                        cg_alpha_val if alpha_type == "Fijo" else cg_alpha_val,
                        int(max_iter), tolerancia, norm_type
                    )

                if show_exam_mode and exam_log:
                    st.markdown("#### 📝 Apuntes del Profesor: Primeras iteraciones")
                    st.info(f"Mostrando los cálculos detallados para las primeras **{len(exam_log)}** iteraciones — **{method}**.")
                    cols_exam = st.columns(len(exam_log))
                    for idx, step_txt in enumerate(exam_log):
                        with cols_exam[idx]:
                            with st.container(border=True):
                                st.markdown(
                                    f"<div style='color:#f97316; font-weight:800; font-size:12px; "
                                    f"letter-spacing:2px; text-transform:uppercase; margin-bottom:8px;'>"
                                    f"🔸 Iteración k = {idx}</div>",
                                    unsafe_allow_html=True
                                )
                                st.markdown(step_txt)
                    st.markdown("<br>", unsafe_allow_html=True)

                col_g1, col_g2 = st.columns(2)

                with col_g1:
                    st.markdown("#### Trayectoria y Curvas de Nivel")
                    f_lambdified_plot = sp.lambdify(vars_sym, expr, 'numpy')

                    if len(vars_names) == 1:
                        x_hist = results[f'{vars_names[0]}'].values
                        f_hist = results['C(x)'].values
                        margin = max(1.0, (max(x_hist) - min(x_hist)) * 0.5)
                        x_range = np.linspace(min(x_hist) - margin, max(x_hist) + margin, 500)
                        y_range_plot = [f_lambdified_plot(val) for val in x_range]

                        fig, ax = plt.subplots(figsize=(7, 5))
                        fig.patch.set_facecolor('#0d1117')
                        ax.set_facecolor('#161b22')
                        ax.plot(x_range, y_range_plot, label='C(x)', color='#58a6ff', linewidth=2.5)
                        ax.plot(x_hist, f_hist, label='Iteraciones', color='#f97316',
                                marker='o', linestyle=':', markersize=7, linewidth=2)
                        ax.set_title("Comportamiento del Algoritmo", color='#e6edf3',
                                     fontweight='bold', pad=15, fontsize=13)
                        ax.set_xlabel(f"{vars_names[0]}", color='#8b949e', fontsize=11)
                        ax.set_ylabel("C(x)", color='#8b949e', fontsize=11)
                        ax.tick_params(colors='#8b949e', which='both')
                        for spine in ax.spines.values():
                            spine.set_edgecolor('#30363d')
                        ax.grid(True, linestyle='--', alpha=0.12, color='#58a6ff')
                        ax.legend(facecolor='#161b22', edgecolor='#30363d',
                                  labelcolor='#e6edf3', framealpha=0.95)
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                    elif len(vars_names) == 2:
                        x_hist = results[f'{vars_names[0]}'].values
                        y_hist = results[f'{vars_names[1]}'].values
                        z_hist = results['C(x)'].values

                        margin_x = max(1.0, (max(x_hist) - min(x_hist)) * 0.5)
                        margin_y = max(1.0, (max(y_hist) - min(y_hist)) * 0.5)
                        x_range = np.linspace(min(x_hist) - margin_x, max(x_hist) + margin_x, 50)
                        y_range_arr = np.linspace(min(y_hist) - margin_y, max(y_hist) + margin_y, 50)
                        X, Y = np.meshgrid(x_range, y_range_arr)
                        Z = np.zeros_like(X)
                        for i in range(X.shape[0]):
                            for j in range(X.shape[1]):
                                try:
                                    val = f_lambdified_plot(X[i, j], Y[i, j])
                                    Z[i, j] = val if not np.isnan(val) and not np.isinf(val) else np.nan
                                except Exception:
                                    Z[i, j] = np.nan

                        fig_3d = go.Figure()
                        fig_3d.add_trace(go.Surface(
                            x=X, y=Y, z=Z, colorscale='Plasma', opacity=0.85,
                            showscale=False, name='Superficie'
                        ))
                        fig_3d.add_trace(go.Scatter3d(
                            x=x_hist, y=y_hist, z=z_hist, mode='lines+markers',
                            marker=dict(size=5, color='#f97316', symbol='circle',
                                        line=dict(color='#0d1117', width=1)),
                            line=dict(color='#f97316', width=5), name='Trayectoria'
                        ))
                        fig_3d.update_layout(
                            template="plotly_dark", paper_bgcolor='#0d1117',
                            title=dict(text="Vista 3D: Superficie y Trayectoria",
                                       font=dict(color='#e6edf3', size=13)),
                            scene=dict(
                                xaxis=dict(title=f"{vars_names[0]}", gridcolor='#21262d',
                                           color='#8b949e', backgroundcolor='#0d1117'),
                                yaxis=dict(title=f"{vars_names[1]}", gridcolor='#21262d',
                                           color='#8b949e', backgroundcolor='#0d1117'),
                                zaxis=dict(title="C(x,y)", gridcolor='#21262d',
                                           color='#8b949e', backgroundcolor='#0d1117'),
                                bgcolor='#0d1117',
                            ),
                            margin=dict(l=0, r=0, b=0, t=50), font=dict(color='#e6edf3')
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)

                        fig_contour = go.Figure()
                        fig_contour.add_trace(go.Contour(
                            x=x_range, y=y_range_arr, z=Z, colorscale='Plasma',
                            contours=dict(showlabels=True, labelfont=dict(size=10, color='white')),
                            colorbar=dict(title=dict(text="C(x,y)", font=dict(color='#8b949e')),
                                          tickfont=dict(color='#8b949e')),
                            name='Curvas de Nivel'
                        ))
                        fig_contour.add_trace(go.Scatter(
                            x=x_hist, y=y_hist, mode='lines+markers',
                            marker=dict(size=7, color='#f97316', symbol='circle',
                                        line=dict(color='#0d1117', width=1)),
                            line=dict(color='#f97316', width=2.5), name='Trayectoria'
                        ))
                        fig_contour.update_layout(
                            template="plotly_dark", paper_bgcolor='#0d1117',
                            plot_bgcolor='#161b22',
                            title=dict(text="Vista Topográfica: Curvas de Nivel",
                                       font=dict(color='#e6edf3', size=13)),
                            xaxis=dict(title=f"{vars_names[0]}", gridcolor='#21262d',
                                       color='#8b949e', zerolinecolor='#30363d'),
                            yaxis=dict(title=f"{vars_names[1]}", gridcolor='#21262d',
                                       color='#8b949e', zerolinecolor='#30363d'),
                            margin=dict(l=0, r=0, b=0, t=50), font=dict(color='#e6edf3')
                        )
                        st.plotly_chart(fig_contour, use_container_width=True)

                    else:
                        st.info("La gráfica de trayectoria interactiva solo está disponible para 1 o 2 variables.")

                with col_g2:
                    st.markdown("#### Análisis de Convergencia")
                    if len(results) > 0:
                        iter_vals = results['Iteración'].values
                        grad_norms = results['||∇C(x)||'].values
                        f_vals = results['C(x)'].values

                        fig_conv = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_conv.add_trace(
                            go.Scatter(
                                x=iter_vals, y=grad_norms, name="||∇f(x_k)||",
                                mode="lines+markers",
                                line=dict(color="#58a6ff", width=2.5),
                                marker=dict(size=6, color='#58a6ff',
                                            line=dict(color='#0d1117', width=1))
                            ),
                            secondary_y=False,
                        )
                        fig_conv.add_trace(
                            go.Scatter(
                                x=iter_vals, y=f_vals, name="f(x_k)",
                                mode="lines+markers",
                                line=dict(color="#f97316", width=2.5, dash="dot"),
                                marker=dict(size=6, color='#f97316',
                                            line=dict(color='#0d1117', width=1))
                            ),
                            secondary_y=True,
                        )
                        fig_conv.update_layout(
                            template="plotly_dark", paper_bgcolor='#0d1117',
                            plot_bgcolor='#161b22',
                            title=dict(text="Convergencia", font=dict(color='#e6edf3', size=15)),
                            xaxis=dict(title="Iteración", gridcolor='#21262d',
                                       color='#8b949e', zerolinecolor='#30363d'),
                            legend=dict(
                                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                bgcolor='rgba(22,27,34,0.9)', bordercolor='#30363d',
                                font=dict(color='#e6edf3')
                            ),
                            margin=dict(l=0, r=0, b=0, t=60), font=dict(color='#e6edf3')
                        )
                        fig_conv.update_yaxes(
                            title_text="||∇f(x_k)|| (escala log)", type="log", secondary_y=False,
                            gridcolor='#21262d', color='#8b949e', zerolinecolor='#30363d'
                        )
                        fig_conv.update_yaxes(
                            title_text="f(x_k)", secondary_y=True,
                            gridcolor='#21262d', color='#8b949e', zerolinecolor='#30363d'
                        )
                        st.plotly_chart(fig_conv, use_container_width=True)
                    else:
                        st.info("El algoritmo no generó iteraciones válidas.")

                st.markdown("#### Tabla General del Historial de Iteraciones")
                st.dataframe(results, use_container_width=True)

            else:
                st.error("Error: Asegúrate que la cantidad de valores en el punto inicial coincida con las variables.")
        except Exception as e:
            st.error(f"Se encontró un error matemático/sintáctico: {e}")

if __name__ == "__main__":
    main_app()
