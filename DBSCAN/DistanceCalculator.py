import numpy as np
from typing import List

class DistanceCalculator:
    """Calcula distancias entre puntos."""

    @staticmethod
    def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Calcula la distancia euclidiana entre dos puntos.
        Args:
            point1: Primer punto
            point2: Segundo punto
        Returns:
            Distancia euclidiana
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))

    @staticmethod
    def get_neighbors(data: np.ndarray, point_idx: int, eps: float) -> List[int]:
        """
        Encuentra todos los vecinos de un punto dentro del radio eps.
        Args:
            data: Dataset completo
            point_idx: Índice del punto a analizar
            eps: Radio de búsqueda
        Returns:
            Lista de índices de vecinos
        """
        neighbors = []
        point = data[point_idx]

        for idx in range(len(data)):
            if DistanceCalculator.euclidean_distance(point, data[idx]) <= eps:
                neighbors.append(idx)

        return neighbors
