import numpy as np

class DataValidator:
    """"Valida los datos de entrada y parámetros del algoritmo."""

    @staticmethod
    def validate_data(data: np.ndarray) -> None:
        """
        Valida que los datos sean correctos.
        Args:
            data: Array numpy con los puntos
        Raises:
            ValueError: Si los datos no son válidos
            TypeError: Si el tipo de dato no es correcto
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Los datos deben ser un numpy array")

        if len(data.shape) != 2:
            raise ValueError("Los datos deben ser una matriz 2D (n_samples, n_features)")

        if data.shape[0] < 1:
            raise ValueError("Debe haber al menos un punto en los datos")

        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError("Los datos contiene valores NaN o Inf")

    @staticmethod
    def validate_parameters(eps: float, min_samples: int, n_points: int) -> None:
        """
        Valida los parámetros del algoritmo.
        Args:
            eps: Radio epsilon
            min_samples: Mínimo de muestras
            n_points: Número de puntos en el dataset
        Raises:
            ValueError: Si los parámetros no son válidos
        """
        if eps <= 0:
            raise ValueError(f"eps debe ser positivo, se recibió: {eps}")

        if min_samples < 1:
            raise ValueError(f"min_samples debe ser >= 1, se recibió: {min_samples}")

        if min_samples > n_points:
            raise ValueError(
                f"min_samples ({min_samples}) no puede ser mayor que "
                f"el número de puntos ({n_points})"
            )
