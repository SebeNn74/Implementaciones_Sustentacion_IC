# 📊 Implementación y Análisis de Linear Discriminant Analysis (LDA)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📝 Descripción

Implementación completa y documentada de **Linear Discriminant Analysis (LDA)** con comparación exhaustiva contra **Principal Component Analysis (PCA)**. Este proyecto incluye validación estadística de supuestos, evaluación con clasificadores, y análisis de eigenvectores discriminantes.

### 🎯 Objetivos del Proyecto

- Implementar LDA desde cero usando scikit-learn
- Comparar técnicas supervisadas (LDA) vs no supervisadas (PCA)
- Validar supuestos estadísticos (normalidad, homocedasticidad)
- Evaluar rendimiento con múltiples clasificadores
- Interpretar eigenvectores discriminantes

---

## 📂 Estructura del Proyecto

```
.
├── Implementacion_LDA_Sustentacion_2_50_IC.ipynb  # Notebook principal
└── README.md                                       # Este archivo
```

---

## 🔧 Instalación y Configuración

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes)
- Jupyter Notebook o VS Code con extensión de Python

### Instalación de Dependencias

```bash
pip install scikit-learn scipy pandas numpy matplotlib seaborn pingouin -q
```

O usando requirements.txt:

```bash
pip install -r requirements.txt
```

### Ejecución del Notebook

```bash
jupyter notebook Implementacion_LDA_Sustentacion_2_50_IC.ipynb
```

O abrir directamente en VS Code.

---

## 📊 Datasets Utilizados

### 1. Wine Dataset 🍷
- **Muestras**: 178
- **Características**: 13 (características químicas)
- **Clases**: 3 (tipos de vino italiano)
- **Propósito**: Demostrar LDA con múltiples clases (visualización 2D)

### 2. Breast Cancer Dataset 🏥
- **Muestras**: 569
- **Características**: 30 (características de tumores)
- **Clases**: 2 (benigno/maligno)
- **Propósito**: Demostrar LDA binario (visualización 1D)

---

## 🧮 Metodología

### 1. Preprocesamiento
- División train/test (70/30) con estratificación
- Estandarización (μ=0, σ=1) usando `StandardScaler`

### 2. Reducción de Dimensionalidad
- **LDA**: Maximiza separación entre clases
  - Wine: 2 componentes (k-1, donde k=3 clases)
  - Breast Cancer: 1 componente (k-1, donde k=2 clases)
- **PCA**: Maximiza varianza total
  - Ambos datasets: 2 componentes para comparación

### 3. Validación Estadística

#### Test de Mardia (Normalidad Multivariada)
Evalúa si los datos siguen distribución normal multivariada.

**Hipótesis:**
- H₀: Los datos siguen una distribución normal multivariada
- H₁: Los datos no siguen una distribución normal multivariada

**Criterio:** p-value > 0.05 → Aceptar normalidad

#### Test de Box's M (Homocedasticidad)
Verifica igualdad de matrices de covarianza entre clases.

**Hipótesis:**
- H₀: Las matrices de covarianza son iguales
- H₁: Las matrices de covarianza son diferentes

**Criterio:** p-value > 0.05 → Aceptar homocedasticidad

### 4. Evaluación con Clasificadores

Se evalúan las proyecciones LDA y PCA usando:

- **SVM (Support Vector Machine)** con kernel RBF
- **Regresión Logística** (baseline lineal)

**Métricas:**
- Accuracy (exactitud)
- Matriz de confusión
- Classification report (precision, recall, f1-score)

---

## 📈 Resultados Esperados

### Wine Dataset (3 clases)

| Método | SVM Accuracy | LR Accuracy | Dimensiones |
|--------|-------------|-------------|-------------|
| **LDA** | ~98-100% | ~97-99% | 2D |
| **PCA** | ~95-97% | ~94-96% | 2D |

**Conclusión:** LDA supera a PCA en tareas de clasificación.

### Breast Cancer Dataset (2 clases)

| Método | SVM Accuracy | LR Accuracy | Dimensiones |
|--------|-------------|-------------|-------------|
| **LDA** | ~96-98% | ~95-97% | 1D |
| **PCA** | ~93-95% | ~92-94% | 2D |

**Conclusión:** LDA con 1 componente puede superar a PCA con 2 componentes.

---

## 🔍 Análisis de Eigenvectores

Los eigenvectores discriminantes revelan las características más importantes:

### Wine Dataset - Top Características (LD1)

1. **Flavonoids** (~0.85) - Mayor peso discriminante
2. **Proline** (~0.24)
3. **Color intensity** (~-0.56) - Dirección opuesta

**Interpretación:** Los flavonoides son el factor químico principal que diferencia entre tipos de vino.

### Breast Cancer Dataset - Top Características (LD1)

Las características de textura y área de los tumores suelen tener los pesos más altos.

---

## 🧪 Validación de Supuestos

### Resultados Típicos

#### Normalidad (Test de Mardia)
- **Wine**: Generalmente cumple normalidad multivariada
- **Breast Cancer**: Puede mostrar desviaciones leves

#### Homocedasticidad (Box's M)
- **Wine**: Matrices de covarianza similares
- **Breast Cancer**: Puede mostrar heterogeneidad

> **Nota:** LDA es robusto a violaciones moderadas de estos supuestos, especialmente con muestras grandes.

---

## 📚 Fundamentos Teóricos

### Linear Discriminant Analysis (LDA)

LDA busca la proyección que maximiza:

$$J(w) = \frac{w^T S_B w}{w^T S_W w}$$

Donde:
- $S_B$ = Matriz de dispersión **between-class** (entre clases)
- $S_W$ = Matriz de dispersión **within-class** (dentro de clases)
- $w$ = Vector de proyección óptimo

### Componentes Discriminantes

Cada componente discriminante es una combinación lineal:

$$LD_i = w_{i1} \cdot x_1 + w_{i2} \cdot x_2 + \ldots + w_{ip} \cdot x_p$$

### Limitaciones de LDA

1. **Máximo k-1 componentes** (k = número de clases)
2. Asume **normalidad multivariada**
3. Asume **homocedasticidad** (covarianzas iguales)
4. **Lineal**: No captura relaciones no lineales

---

## 🚀 Extensiones Posibles

### 1. Quadratic Discriminant Analysis (QDA)
Para datos con matrices de covarianza diferentes por clase.

```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis()
```

### 2. Regularized LDA
Para datos de alta dimensión (p >> n).

```python
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
```

### 3. Kernel LDA
Para fronteras de decisión no lineales.

### 4. Cross-Validation
Validación cruzada k-fold para mayor robustez.

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lda, X, y, cv=5)
```

---

## 🔬 Tecnologías Utilizadas

| Librería | Versión | Propósito |
|----------|---------|-----------|
| **scikit-learn** | ≥1.3 | LDA, PCA, clasificadores |
| **numpy** | ≥1.24 | Operaciones numéricas |
| **pandas** | ≥2.0 | Manipulación de datos |
| **matplotlib** | ≥3.7 | Visualización |
| **seaborn** | ≥0.12 | Gráficos estadísticos |
| **scipy** | ≥1.10 | Tests estadísticos |

---

## 📖 Referencias

### Artículos Fundamentales
1. **Fisher, R.A. (1936)**. "The use of multiple measurements in taxonomic problems". *Annals of Eugenics*, 7(2), 179-188.
   - Artículo original que introduce LDA

2. **Duda, R.O., Hart, P.E., & Stork, D.G. (2001)**. *Pattern Classification* (2nd ed.). Wiley.
   - Capítulo 3: Linear Discriminant Functions

### Libros Recomendados
3. **Hastie, T., Tibshirani, R., & Friedman, J. (2009)**. *The Elements of Statistical Learning*. Springer.
   - Sección 4.3: Linear Discriminant Analysis

4. **James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013)**. *An Introduction to Statistical Learning*. Springer.
   - Capítulo 4: Classification

### Documentación
5. [Scikit-learn LDA Documentation](https://scikit-learn.org/stable/modules/lda_qda.html)
6. [Scipy Statistical Tests](https://docs.scipy.org/doc/scipy/reference/stats.html)

---

## 👥 Autores

- **Milton Nicolas Pirazan Forero** - *Implementación y documentación*

---

## 🔄 Historial de Versiones

- **v1.0.0** (Nov 2025) - Implementación completa con documentación
  - ✅ LDA y PCA implementados
  - ✅ Tests estadísticos (Mardia, Box's M)
  - ✅ Evaluación con SVM y Regresión Logística
  - ✅ Análisis de eigenvectores
  - ✅ Documentación completa en notebook