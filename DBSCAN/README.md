# DBSCAN - Implementación desde Cero en Python

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-orange.svg)](https://numpy.org/)

> Implementación educativa completa del algoritmo DBSCAN (Density-Based Spatial Clustering of Applications with Noise) sin usar librerías de clustering especializadas.

---

## Descripción

DBSCAN es un algoritmo de clustering basado en **densidad** que identifica clusters de forma arbitraria y detecta automáticamente outliers. A diferencia de K-Means, **no requiere especificar el número de clusters** y puede encontrar estructuras complejas en los datos.

### ¿Por qué DBSCAN?

 **Detecta formas arbitrarias** - No asume clusters esféricos o convexos  
 **Identifica outliers automáticamente** - Puntos de ruido clasificados explícitamente  
 **No requiere K** - Descubre el número de clusters por sí mismo  
 **Robusto ante ruido** - Outliers no afectan la formación de clusters  

---

## Características

- **Implementación desde cero**: Sin usar `sklearn.cluster.DBSCAN`
- **Validación completa**: Manejo robusto de excepciones
- **Visualización integrada**: Gráficos con matplotlib
- **5 tests representativos**: Casos de uso reales
- **Documentación extensa**: Código comentado y explicado
- **Arquitectura modular**: 6 módulos independientes
- **Propósito educativo**: Ideal para aprender el algoritmo

---

## Requisitos

### Software necesario:
- **Python**: 3.8 o superior
- **pip**: Gestor de paquetes de Python

### Dependencias:
```
numpy >= 1.19.0
matplotlib >= 3.3.0
scikit-learn >= 0.24.0  (solo para datasets de prueba)
```

---

## Instalación

### 1. Clonar/Descargar el proyecto
```bash
git clone https://github.com/SebeNn74/Implementaciones_Sustentacion_IC.git
cd DBSCAN
```

### 2. Crear entorno virtual (recomendado)
```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno
# En Windows:
venv\Scripts\activate

# En Linux/Mac:
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install numpy matplotlib scikit-learn
```
---

## Guía de Ejecución

### Opción 1: Ejecutar Suite Completa de Tests (Recomendado)

```bash
python run.py
```

**Esto ejecutará:**
- 5 tests representativos
- Validación de datos y parámetros
- Clustering en diferentes datasets
- Visualizaciones automáticas
---

## Estructura del Proyecto

```
DBSCAN/
│
├── DataGenerator.py         # Modulo para generación de datasets
├── DataValidator.py         # Modulo para Validación de datos y parametros
├── DBSCAN.py                # Modulo de la implementación completa del algoritmo
├── DBSCANVisualizer.py      # Modulo de visualización de resultados
├── DistanceCalculator.py    # Modulo de cálculo de distancias y vecinos
├── TestSuite.py             # Modulo de Tests
├── run.py                   # Run
└── README.md                # Este archivo
```

### Descripción de Módulos

| Módulo | Responsabilidad | Concepto Teórico |
|--------|----------------|------------------|
| **DataValidator** | Validar datos y parámetros | Garantizar espacio métrico válido |
| **DistanceCalculator** | Calcular distancias y vecinos | Implementar Nε(p) = {q \| d(p,q) ≤ ε} |
| **DBSCAN** | Algoritmo de clustering | Conectividad por densidad |
| **DBSCANVisualizer** | Visualización de resultados | Representación gráfica |
| **DataGenerator** | Crear datasets de prueba | Casos de uso estándar |
| **TestSuite** | Validación y demostración | Verificación de correctitud |

---

## Parámetros del Algoritmo

### Parámetros principales:

#### **eps (epsilon)** - Radio de Vecindad
```python
DBSCAN(eps=0.3, ...)
```

- **Definición**: Radio máximo para considerar vecinos
- **Concepto**: Define Nε(p) = {q | dist(p,q) ≤ eps}
- **Rango típico**: 0.1 - 2.0 (depende de la escala de datos)
- **Efecto**:
  - **eps pequeño**: Más clusters fragmentados, más ruido
  - **eps grande**: Clusters se fusionan, menos ruido

#### **min_samples (MinPts)** - Densidad Mínima
```python
DBSCAN(eps=0.3, min_samples=5)
```
- **Definición**: Mínimo de puntos para formar un cluster denso
- **Concepto**: Punto core si |Nε(p)| ≥ min_samples
- **Rango típico**: 4 - 10 (regla: min_samples ≥ dimensiones + 1)
- **Efecto**:
  - **min_samples bajo**: Más clusters pequeños
  - **min_samples alto**: Clusters más densos, más ruido

---