import numpy as np

from DBSCAN import DBSCAN
from DBSCANVisualizer import DBSCANVisualizer
from DataGenerator import DataGenerator
from DataValidator import DataValidator

def run_tests():
    """Ejecuta Tests."""

    print("=" * 80)
    print("SUITE DE PRUEBAS - DBSCAN")
    print("=" * 80)

    # TEST 1: Clusters no convexos (Ventaja clave de DBSCAN sobre K-Means)
    print("\n[TEST 1] Formas NO CONVEXAS - Dataset Moons")
    print("-" * 80)
    print("Objetivo: Demostrar que DBSCAN detecta clusters con formas arbitrarias")
    X_moons = DataGenerator.make_moons_dataset(n_samples=300, noise=0.1)
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    dbscan.fit(X_moons)
    stats = dbscan.get_statistics()

    print(f"Clusters encontrados: {stats['n_clusters']}")
    print(f"Puntos de ruido: {stats['n_noise']}")
    print(f"Puntos core: {stats['n_core_points']}")
    print(f"Puntos border: {stats['n_border_points']}")
    for cluster, size in stats['cluster_sizes'].items():
        print(f"   {cluster}: {size} puntos")
    print("💡 K-Means fallaría aquí, pero DBSCAN identifica correctamente las lunas")

    DBSCANVisualizer.plot_clusters(X_moons, dbscan.labels_,
                                   dbscan.core_samples_,
                                   "TEST 2: Clusters No Convexos (Moons)")

    # TEST 2: Clusters anidados (Círculos concéntricos)
    print("\n[TEST 2] Clusters ANIDADOS - Dataset Circles")
    print("-" * 80)
    print("Objetivo: Separar círculos concéntricos (imposible para K-Means)")
    X_circles = DataGenerator.make_circles_dataset(n_samples=300, noise=0.05)
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    dbscan.fit(X_circles)
    stats = dbscan.get_statistics()

    print(f"Clusters encontrados: {stats['n_clusters']}")
    print(f"Puntos de ruido: {stats['n_noise']}")
    print("💡 Demuestra la capacidad de DBSCAN para separar estructuras anidadas")

    DBSCANVisualizer.plot_clusters(X_circles, dbscan.labels_,
                                   dbscan.core_samples_,
                                   "TEST 3: Clusters Anidados (Circles)")

    # TEST 3: Detección de ruido y outliers
    print("\n[TEST 3] DETECCIÓN DE RUIDO y Outliers")
    print("-" * 80)
    print("Objetivo: Identificar automáticamente puntos atípicos")

    # Crear dataset con outliers intencionales
    X_blobs = DataGenerator.make_blobs_dataset(n_samples=250, n_centers=3)
    # Agregar outliers artificiales
    outliers = np.random.uniform(-8, 8, (20, 2))
    X_with_noise = np.vstack([X_blobs, outliers])

    dbscan = DBSCAN(eps=0.8, min_samples=5)
    dbscan.fit(X_with_noise)
    stats = dbscan.get_statistics()

    print(f"Clusters encontrados: {stats['n_clusters']}")
    print(f"Puntos de ruido detectados: {stats['n_noise']}")
    print(f"Porcentaje de ruido: {stats['n_noise'] / len(X_with_noise) * 100:.1f}%")
    print("💡 DBSCAN identifica automáticamente outliers sin necesidad de preprocesamiento")

    DBSCANVisualizer.plot_clusters(X_with_noise, dbscan.labels_,
                                   dbscan.core_samples_,
                                   "TEST 4: Detección Automática de Outliers")

    # TEST 4: Análisis de sensibilidad de parámetros
    print("\n[TEST 4] SENSIBILIDAD de Parámetros (eps)")
    print("-" * 80)
    print("Objetivo: Analizar impacto de eps en los resultados")
    X_test = DataGenerator.make_moons_dataset(n_samples=200, noise=0.1)

    print("\n| eps  | Clusters | Ruido | Interpretación")
    print("|------|----------|-------|----------------------------------")

    eps_values = [0.1, 0.2, 0.3, 0.5]
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        dbscan.fit(X_test)
        stats = dbscan.get_statistics()

        if eps == 0.1:
            interp = "eps muy pequeño → muchos puntos como ruido"
        elif eps == 0.2:
            interp = "eps pequeño → puede fragmentar clusters"
        elif eps == 0.3:
            interp = "eps óptimo → clusters bien definidos"
        else:
            interp = "eps grande → puede unir clusters separados"

        print(f"| {eps:.1f}  | {stats['n_clusters']:^8} | {stats['n_noise']:^5} | {interp}")

    print("\n" + "=" * 80)
    print("TESTS COMPLETADOS")
    print("=" * 80)
