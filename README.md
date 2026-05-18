# 📈 Optimización Irrestricta — App Web

Aplicación web profesional para minimizar funciones no lineales por:

- **Método del gradiente** (descenso máximo)
- **Gradiente conjugado** no lineal (Polak-Ribière+ / Fletcher-Reeves)
- **Método de Newton** amortiguado (con modificación de Hessiana)

Todos los métodos usan **búsqueda de línea con condiciones de Wolfe fuertes**
(Armijo + curvatura).

---

## 🚀 Probar en línea

Una vez desplegada, la app estará disponible en:
`https://<tu-usuario>-optimizacion-app.streamlit.app`

---

## 🖥️ Ejecutar localmente

### 1. Requisitos

- Python 3.10 o superior

### 2. Instalación

```powershell
# Desde la carpeta del proyecto
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Lanzar la app

```powershell
streamlit run app.py
```

Se abrirá automáticamente en `http://localhost:8501`.

---

## ☁️ Desplegar públicamente (Streamlit Community Cloud — gratis)

1. Sube esta carpeta a un repositorio público de GitHub.
2. Entra a <https://share.streamlit.io/> e inicia sesión con GitHub.
3. Pulsa **New app** → selecciona el repo y el archivo `app.py`.
4. Streamlit instalará `requirements.txt` automáticamente y publicará la URL.

> Tiempo de despliegue: ~2 minutos. URL pública estable y gratuita.

---

## 📂 Estructura del proyecto

```
optimizacion-app/
├── app.py                       Aplicación Streamlit (UI principal)
├── requirements.txt             Dependencias
├── .streamlit/
│   └── config.toml              Tema visual personalizado
├── optimizers/
│   ├── wolfe.py                 Búsqueda de línea con Wolfe fuerte
│   ├── gradient_descent.py      Método del gradiente
│   ├── conjugate_gradient.py    Gradiente conjugado (PR+ / FR)
│   └── newton.py                Newton con modificación Levenberg
└── utils/
    ├── function_parser.py       Parser simbólico (SymPy)
    └── plotting.py              Gráficos Plotly interactivos
```

---

## 📋 Funcionalidades

### Entrada
- Número de variables (1–10)
- Método de optimización
- Función objetivo (expresión simbólica con `x1, x2, ..., xn`)
- Punto de partida
- Máximo de iteraciones
- Tolerancia de convergencia
- Parámetros de Wolfe (c₁, c₂)
- Variante del gradiente conjugado (PR+ o FR)

### Funciones predefinidas
- Esfera, Rosenbrock, Himmelblau, Booth, Beale, cuadrática 3D

### Salida
- Punto mínimo encontrado **x\***
- Valor de la función **f(x\*)**
- Número de iteraciones y tiempo de cómputo
- Criterio de parada alcanzado
- **Gráfico de convergencia** (‖∇f‖ y f(x) vs iteración, escala log)
- **Curvas de nivel + trayectoria** (cuando n = 2)
- **Superficie 3D interactiva** con trayectoria (cuando n = 2)
- **Tabla detallada de iteraciones** descargable como CSV

---

## 🧮 Detalles técnicos

### Wolfe fuerte
Algoritmo de Nocedal & Wright (*Numerical Optimization*, 2ed, Alg. 3.5–3.6):

- **Armijo:** `f(x + α·p) ≤ f(x) + c₁·α·∇f(x)ᵀp`
- **Curvatura:** `|∇f(x + α·p)ᵀp| ≤ c₂·|∇f(x)ᵀp|`

### Newton modificado
Si la Hessiana no es definida positiva, se resuelve
`(H + τI)·p = -∇f` con τ creciente hasta que la factorización de Cholesky
sea estable.

### Gradiente conjugado
- **PR+** (Polak-Ribière con `max(β, 0)`)
- **FR** (Fletcher-Reeves)
- Reinicio de la dirección cada *n* iteraciones para preservar convergencia.

---

## 👥 Trabajo grupal

Universidad — Métodos Numéricos para Optimización.
