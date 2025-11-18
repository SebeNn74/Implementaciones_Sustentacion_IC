"""
Test de comparación entre:
- K-Means (algoritmo determinista de clustering)
- EM para Mezclas Gaussianas (implementado desde cero en EMGaussianMixture)

El objetivo es visualizar y analizar las diferencias
entre un método basado en distancia (K-Means) y el método probabilístico
(EM + GMM), especialmente cuando los clusters tienen distintas varianzas,   ya que K-Means solo funciona bien con clusters
esféricos, mientras que EM puede adaptarse a formas elípticas.
"""
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
# Importamos nuestra implementación EM desde EM_implementacion_GMM.py
from EM_implementacion_GMM import EMGaussianMixture

#                GENERACIÓN DEL DATASET DE PRUEBA

# make_blobs permite crear conjuntos de datos con forma de cluster.
# Es ideal para analizar EM porque se pueden definir varianzas distintas.

# cluster_std controla cuánta variabilidad tiene cada cluster.
# Aquí usamos 3 varianzas diferentes para demostrar que:
# - K-Means asume clusters esféricos → falla si hay varianzas diferentes.
# - EM usa matrices de covarianza → se adapta mejor a formas elípticas.

X, _ = make_blobs(n_samples=500, centers=3, cluster_std=[1.2, 2.8, 0.9], random_state=0)

#                   ENTRENAR ALGORITMO EM

# n_components = número de gaussianas que queremos ajustar al modelo.
em = EMGaussianMixture(n_components=3)
# Entrenamos usando nuestro algoritmo EM implementado a mano
em.fit(X)
# Obtenemos las etiquetas finales según la gaussiana de mayor responsabilidad
labels_em = em.predict(X)

#                     ENTRENAR K-MEANS

# kmeans sirve como comparación. Es el algoritmo de clustering más simple.
# Sin embargo, se basa únicamente en distancia euclidiana;
# esto lo hace inferior cuando los clusters tienen formas no esféricas.

kmeans = KMeans(n_clusters=3, n_init=10)
labels_k = kmeans.fit_predict(X)

#                     VISUALIZACIÓN DE RESULTADOS
plt.figure(figsize=(12,5))

# -------- Gráfica de K-Means --------

plt.subplot(1,2,1)
plt.scatter(X[:,0], X[:,1], c=labels_k)
plt.title("K-means  -clusters esféricos")

# -------- Gráfica de EM --------
plt.subplot(1,2,2)
plt.scatter(X[:,0], X[:,1], c=labels_em)
plt.title("EM (GMM)-clusters elípticos")

plt.show()
