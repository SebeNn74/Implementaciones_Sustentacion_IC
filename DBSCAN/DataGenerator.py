import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs

class DataGenerator:
    """Genera datasets sintéticos para pruebas usando sklearn."""

    @staticmethod
    def make_moons_dataset(n_samples: int = 300, noise: float = 0.1) -> np.ndarray:
        """
        Genera dataset con forma de lunas usando sklearn.

        Args:
            n_samples: Número de muestras
            noise: Nivel de ruido

        Returns:
            Array numpy con los datos
        """
        X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        return X

    @staticmethod
    def make_circles_dataset(n_samples: int = 300, noise: float = 0.05,
                             factor: float = 0.5) -> np.ndarray:
        """
        Genera dataset con círculos concéntricos usando sklearn.

        Args:
            n_samples: Número de muestras
            noise: Nivel de ruido
            factor: Escala entre círculos interno y externo

        Returns:
            Array numpy con los datos
        """
        X, _ = make_circles(n_samples=n_samples, noise=noise,
                            factor=factor, random_state=42)
        return X

    @staticmethod
    def make_blobs_dataset(n_samples: int = 300, n_centers: int = 3,
                           cluster_std: float = 0.5) -> np.ndarray:
        """
        Genera dataset con clusters en forma de blob usando sklearn.

        Args:
            n_samples: Número de muestras
            n_centers: Número de centros
            cluster_std: Desviación estándar de los clusters

        Returns:
            Array numpy con los datos
        """
        X, _ = make_blobs(n_samples=n_samples, centers=n_centers,
                          cluster_std=cluster_std, random_state=42)
        return X

    @staticmethod
    def make_anisotropic_dataset(n_samples: int = 300) -> np.ndarray:
        """
        Genera dataset con clusters anisotrópicos (elongados).

        Args:
            n_samples: Número de muestras

        Returns:
            Array numpy con los datos
        """
        X, _ = make_blobs(n_samples=n_samples, centers=3, random_state=42)
        # Transformación anisotrópica
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso = np.dot(X, transformation)
        return X_aniso

    @staticmethod
    def make_varied_density_dataset(n_samples: int = 300) -> np.ndarray:
        """
        Genera dataset con clusters de densidades variadas.

        Args:
            n_samples: Número de muestras

        Returns:
            Array numpy con los datos
        """
        X, _ = make_blobs(n_samples=n_samples, centers=3,
                          cluster_std=[1.0, 2.5, 0.5], random_state=42)
        return X