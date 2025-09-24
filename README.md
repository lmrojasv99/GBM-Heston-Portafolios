# Simulación de Procesos Estocásticos en Finanzas

Un módulo completo de Python para la simulación y análisis de procesos estocásticos aplicados a finanzas cuantitativas, implementando Movimiento Browniano, Movimiento Browniano Geométrico (GBM) y el Modelo de Heston.

## Descripción General

Este proyecto demuestra técnicas avanzadas de matemáticas financieras usando Python, específicamente diseñado para el modelado de precios de activos y optimización de carteras usando procesos estocásticos. El módulo implementa conceptos fundamentales de finanzas cuantitativas y proporciona herramientas para la simulación de trayectorias de precios.

## Características Principales

### Movimiento Browniano Estándar
- **Propiedades Matemáticas**: Implementación completa con condición inicial, incrementos independientes y distribución normal
- **Simulación de Trayectorias**: Generación de múltiples caminos aleatorios
- **Visualización**: Gráficos interactivos de las trayectorias simuladas

### Movimiento Browniano Geométrico (GBM)
- **Modelado de Precios**: Simulación realista de precios de activos financieros
- **Calibración con Datos Reales**: Integración con Yahoo Finance para estimación de parámetros
- **Solución Analítica**: Implementación de la fórmula cerrada del GBM

### Modelo de Heston
- **Volatilidad Estocástica**: Modelado avanzado con volatilidad variable en el tiempo
- **Correlación**: Incorpora correlación entre precio y volatilidad
- **Efectos de Mercado**: Captura volatility clustering y sonrisa de volatilidad

### Optimización de Carteras
- **Frontera Eficiente**: Construcción usando simulaciones Monte Carlo
- **Ratio de Sharpe**: Optimización del rendimiento ajustado por riesgo
- **Activos Correlacionados**: Manejo de matrices de correlación multivariadas

## Fundamentos Matemáticos

### Movimiento Browniano Estándar
Un proceso estocástico $W_t$ que cumple:
1. $W_0 = 0$ casi seguramente
2. Incrementos independientes: $W_t - W_s$ independiente para $0 \leq s < t$
3. $W_t - W_s \sim \mathcal{N}(0, t-s)$
4. Trayectorias continuas

### Movimiento Browniano Geométrico


Donde $\mathbb{E}[dW_t^1 dW_t^2] = ρ dt$

## Ejemplo de Uso

### Simulación Básica de GBM

```python
import numpy as np
import matplotlib.pyplot as plt

# Parámetros del modelo
S0 = 100       # Precio inicial
mu = 0.08      # Rendimiento esperado (8% anual)
sigma = 0.25   # Volatilidad (25% anual)
T = 1          # Horizonte temporal (1 año)
N = 252        # Número de días hábiles

# Función de simulación
def simulate_gbm(S0, mu, sigma, T, N, num_simulations):
    dt = T / (N - 1)
    t = np.linspace(0, T, N)
    S = np.zeros((N, num_simulations))
    S[0] = S0
    
    for i in range(1, N):
        W = np.random.normal(0, np.sqrt(dt), num_simulations)
        S[i] = S[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * W)
    
    return t, S

# Simular y graficar
t, S = simulate_gbm(S0, mu, sigma, T, N, 10)
```

### Calibración con Datos Reales

```python
import yfinance as yf

# Descargar datos de Apple
ticker = 'AAPL'
data = yf.download(ticker, start='2023-01-01', end='2024-01-01')
price_series = data['Close']

# Calcular parámetros del GBM
log_returns = np.log(price_series / price_series.shift(1)).dropna()
mu = log_returns.mean() * 252      # Drift anualizado
sigma = log_returns.std() * np.sqrt(252)  # Volatilidad anualizada
```

## Resultados de Ejemplo

### Parámetros Calibrados (AAPL 2023)
- **Precio inicial**: $123.33
- **Drift (μ)**: 0.4422 (44.22% anual)
- **Volatilidad (σ)**: 0.1992 (19.92% anual)
- **Días de datos**: 249

### Optimización de Cartera
- **Cartera óptima encontrada**: [19.24%, 46.59%, 27.28%, 6.89%]
- **Ratio de Sharpe máximo**: Optimizado usando 10,000 simulaciones Monte Carlo
- **Activos correlacionados**: Matriz de correlación 4x4 implementada

## Aplicaciones Académicas

### Conceptos de Finanzas Cuantitativas
- Procesos estocásticos en tiempo continuo
- Calibración de modelos con datos de mercado
- Simulación Monte Carlo para derivados
- Optimización de carteras bajo incertidumbre

### Habilidades de Programación
- Programación científica con NumPy
- Visualización profesional con Matplotlib
- Integración de datos financieros en tiempo real
- Algoritmos de optimización numérica

### Métodos Estadísticos
- Procesos de Wiener y ecuaciones diferenciales estocásticas
- Estimación de parámetros por máxima verosimilitud
- Análisis de series temporales financieras
- Técnicas de reducción de varianza en Monte Carlo

## Marco Teórico

### Ecuaciones Diferenciales Estocásticas (SDE)
- Integración de Itô
- Lema de Itô para transformaciones
- Discretización de Euler-Maruyama

### Modelos de Volatilidad
- Volatilidad constante (Black-Scholes)
- Volatilidad estocástica (Heston, Hull-White)
- Efectos de sonrisa y skew

### Teoría de Carteras Moderna
- Optimización media-varianza extendida
- Incorporación de momentos superiores
- Modelos de factores y correlación dinámica

## Requisitos

```python
numpy>=1.21.0
matplotlib>=3.4.0
yfinance>=0.1.70
pandas>=1.3.0
aleatory>=0.8.0  # Para procesos estocásticos avanzados
scipy>=1.7.0
```

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/usuario/browniano-finanzas.git
cd browniano-finanzas
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecutar el notebook:
```bash
jupyter notebook Browniano.ipynb
```

## Contacto

**Luis Rojas**  
lmrojas99@gmail.com  
www.linkedin.com/in/lmrojasv
