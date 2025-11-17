import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from EM_implementacion_GMM import EMGaussianMixture

# generar datos
X, _ = make_blobs(n_samples=500, centers=3, cluster_std=[1.2, 2.8, 0.9], random_state=0)

# ---- EM ----
em = EMGaussianMixture(n_components=3)
em.fit(X)
labels_em = em.predict(X)

# ---- K-means ----
kmeans = KMeans(n_clusters=3, n_init=10)
labels_k = kmeans.fit_predict(X)

# ---- Graficar ----
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(X[:,0], X[:,1], c=labels_k)
plt.title("K-means")

plt.subplot(1,2,2)
plt.scatter(X[:,0], X[:,1], c=labels_em)
plt.title("EM (GMM desde cero)")

plt.show()
