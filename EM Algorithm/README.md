# Expectation–Maximization (EM) para Modelos de Mezclas Gaussianas (GMM)
Implementación desde cero del algoritmo **Expectation–Maximization (EM)** aplicado a un **Modelo de Mezclas Gaussianas (Gaussian Mixture Model, GMM)**.  
Este proyecto hace parte de la sustentación del cohorte y corresponde a la exposición del tema EM.

---

## 📌 Objetivo
Implementar de manera práctica el algoritmo EM sin utilizar librerías externas de clustering (como `sklearn.mixture.GaussianMixture`) para demostrar comprensión del método, su formulación matemática y su aplicación a problemas reales de agrupamiento.

El proyecto incluye:
- Implementación propia del algoritmo EM.
- Comparación experimental con K-Means.
- Análisis de resultados y comportamiento del algoritmo.
- Código completamente documentado.
- Sustentación teórico-práctica del modelo.

---

## 📚 Descripción del Algoritmo

El **Expectation–Maximization** es un método iterativo para estimar parámetros en modelos con **variables ocultas**. En este proyecto, EM se utiliza para ajustar un modelo de **mezcla de gaussianas**, donde la variable oculta es la pertenencia de cada punto a un cluster.

El ciclo del algoritmo es:

### **1. E-Step (Esperanza):**
Calcular las probabilidades de pertenencia (*responsabilidades*) de cada punto a cada gaussiana del modelo.

### **2. M-Step (Maximización):**
Actualizar los parámetros del modelo:
- Medias  
- Covarianzas  
- Pesos de mezcla  

### **3. Evaluación de la Log-Likelihood**
Medir qué tan bien el modelo explica los datos y detener cuando converge.

---

## 🧠 Modelo Implementado
La mezcla de gaussianas está definida por:

- \( \pi_j \): peso del componente j  
- \( \mu_j \): media del componente j  
- \( \Sigma_j \): matriz de covarianza del componente j  

El modelo general es:

\[
p(x) = \sum_{j=1}^{k} \pi_j \, \mathcal{N}(x \mid \mu_j, \Sigma_j)
\]

La variable oculta \(z_i\) indica el componente al que pertenece cada \(x_i\).
EM estima todos los parámetros maximizando la verosimilitud del modelo.

---

# 🧩 Estructura
📁 EM ALgorithm/
│── em_gmm.py # Implementación completa del algoritmo EM
│── test_EM_K-means.py # Script de pruebas y comparación con K-Means
│── README.md # Este documento
└── data/ # (Opcional) Datos adicionales para pruebas
