import numpy as np
from typing import List
from collections import deque

from DataValidator import DataValidator
from DistanceCalculator import DistanceCalculator


class DBSCAN:
    """
    Implementación del algoritmo DBSCAN.

    Attributes:
        eps: Radio de vecindad
        min_samples: Mínimo de puntos para formar un cluster
        labels_: Etiquetas asignadas (-1 para ruido, 0+ para clusters)
        core_samples_: Máscara booleana indicando puntos core
    """

    # Constantes para etiquetas
    NOISE = -1
    UNCLASSIFIED = -2

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        """
        Inicializa DBSCAN.

        Args:
            eps: Radio de vecindad (epsilon)
            min_samples: Mínimo de puntos para formar cluster denso
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.core_samples_ = None
        self._n_clusters = 0

    def fit(self, X: np.ndarray) -> 'DBSCAN':
        """
        Ejecuta el algoritmo DBSCAN sobre los datos.

        Args:
            X: Dataset de forma (n_samples, n_features)

        Returns:
            self

        Raises:
            ValueError: Si los datos o parámetros no son válidos
        """
        # Validaciones
        DataValidator.validate_data(X)
        DataValidator.validate_parameters(self.eps, self.min_samples, len(X))

        n_samples = len(X)

        # Inicializar etiquetas como no clasificadas
        self.labels_ = np.full(n_samples, self.UNCLASSIFIED, dtype=int)
        self.core_samples_ = np.zeros(n_samples, dtype=bool)

        current_cluster = 0

        # Iterar sobre cada punto
        for point_idx in range(n_samples):
            # Si ya fue clasificado, saltar
            if self.labels_[point_idx] != self.UNCLASSIFIED:
                continue

            # Encontrar vecinos
            neighbors = DistanceCalculator.get_neighbors(X, point_idx, self.eps)

            # Si no tiene suficientes vecinos, marcar como ruido (temporalmente)
            if len(neighbors) < self.min_samples:
                self.labels_[point_idx] = self.NOISE
                continue

            # Es un punto core, iniciar nuevo cluster
            self.core_samples_[point_idx] = True
            self._expand_cluster(X, point_idx, neighbors, current_cluster)
            current_cluster += 1

        self._n_clusters = current_cluster
        return self

    def _expand_cluster(self, X: np.ndarray, point_idx: int,
                        neighbors: List[int], cluster_id: int) -> None:
        """
        Expande un cluster usando BFS (Breadth-First Search).

        Args:
            X: Dataset
            point_idx: Índice del punto semilla
            neighbors: Vecinos del punto semilla
            cluster_id: ID del cluster actual
        """
        # Usar una cola para BFS
        seeds = deque(neighbors)
        self.labels_[point_idx] = cluster_id

        while seeds:
            current_point = seeds.popleft()

            # Si era ruido, ahora es border point
            if self.labels_[current_point] == self.NOISE:
                self.labels_[current_point] = cluster_id

            # Si ya estaba clasificado, continuar
            if self.labels_[current_point] != self.UNCLASSIFIED:
                continue

            # Asignar al cluster
            self.labels_[current_point] = cluster_id

            # Encontrar vecinos del punto actual
            current_neighbors = DistanceCalculator.get_neighbors(
                X, current_point, self.eps
            )

            # Si es un core point, agregar sus vecinos a la cola
            if len(current_neighbors) >= self.min_samples:
                self.core_samples_[current_point] = True
                for neighbor in current_neighbors:
                    if self.labels_[neighbor] == self.UNCLASSIFIED:
                        seeds.append(neighbor)

    def get_statistics(self) -> dict:
        """
        Obtiene estadísticas del clustering.

        Returns:
            Diccionario con estadísticas
        """
        if self.labels_ is None:
            raise RuntimeError("Debe ejecutar fit() primero")

        unique_labels = set(self.labels_)
        n_noise = np.sum(self.labels_ == self.NOISE)
        n_core = np.sum(self.core_samples_)
        n_border = len(self.labels_) - n_core - n_noise

        cluster_sizes = {}
        for label in unique_labels:
            if label != self.NOISE:
                cluster_sizes[f"Cluster {label}"] = np.sum(self.labels_ == label)

        return {
            "n_clusters": self._n_clusters,
            "n_noise": n_noise,
            "n_core_points": n_core,
            "n_border_points": n_border,
            "cluster_sizes": cluster_sizes
        }