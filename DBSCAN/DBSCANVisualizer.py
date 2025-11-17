import numpy as np
import matplotlib.pyplot as plt

class DBSCANVisualizer:
    """Visualiza los resultados del clustering."""

    @staticmethod
    def plot_clusters(X: np.ndarray, labels: np.ndarray,
                      core_samples: np.ndarray = None,
                      title: str = "DBSCAN Clustering") -> None:
        """
        Visualiza los clusters en 2D.

        Args:
            X: Dataset (debe ser 2D)
            labels: Etiquetas de clusters
            core_samples: Máscara de puntos core
            title: Título del gráfico
        """
        if X.shape[1] != 2:
            print("⚠️  Visualización solo disponible para datos 2D")
            return

        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        plt.figure(figsize=(12, 8))

        # Colores para los clusters
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Ruido en negro
                color = 'black'
                marker = 'x'
                size = 50
            else:
                marker = 'o'
                size = 100

            class_mask = labels == label
            xy = X[class_mask]

            kwargs = {
                "c": [color],
                "marker": marker,
                "s": size,
                "alpha": 0.6,
                "linewidth": 0.5,
                "label": f'Cluster {label}' if label != -1 else 'Ruido'
            }

            # Marcadores rellenados → borde negro
            if marker not in {'x', '+', '1', '2', '3', '4', '|', '_'}:
                kwargs["edgecolors"] = "black"

            plt.scatter(xy[:, 0], xy[:, 1], **kwargs)

            # Marcar puntos core
            if core_samples is not None and label != -1:
                core_mask = class_mask & core_samples
                if np.any(core_mask):
                    core_xy = X[core_mask]
                    plt.scatter(core_xy[:, 0], core_xy[:, 1],
                                c=[color],
                                marker='o',
                                s=200,
                                alpha=0.3,
                                edgecolors='red',
                                linewidth=2)

        plt.title(f'{title}\n(Clusters: {n_clusters}, Ruido: {np.sum(labels == -1)})',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
