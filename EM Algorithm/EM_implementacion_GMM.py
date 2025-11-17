import numpy as np

class EMGaussianMixture:

    def __init__(self, n_components, max_iter=100, tol=1e-4):
        
        """
        Inicializa el modelo.

        n_components: número de gaussianas (clusters)
        max_iter: máximo número de iteraciones del algoritmo EM
        tol: tolerancia para determinar convergencia
        """ 
        self.k = n_components
        self.max_iter = max_iter
        self.tol = tol

  #                     FUNCIÓN DENSIDAD DE GAUSSIANA
    @staticmethod
    def gaussian_pdf(x, mean, cov):
        """
        Calcula la densidad de probabilidad de una distribución normal multivariada.

        x: punto de datos (vector)
        mean: media de la gaussiana
        cov: matriz de covarianza

        Fórmula:
        N(x | μ, Σ) = exp(-1/2 (x-μ)^T Σ^-1 (x-μ)) / sqrt((2π)^d det(Σ))
        """
        d = mean.shape[0]
        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)
        norm = np.sqrt((2 * np.pi)**d * det)
        diff = x - mean
        return np.exp(-0.5 * diff.T @ inv @ diff) / norm


    #                       ENTRENAMIENTO (ALGORITMO EM)
    def fit(self, X):
        """
        Entrena el modelo usando el algoritmo EM.

        X: datos de entrenamiento, matriz (n_samples, n_features)
        """
        n, d = X.shape
    
        # --- Inicialización ---
        # Inicializamos medias con puntos aleatorios del dataset
        self.means = X[np.random.choice(n, self.k, replace=False)]

        # Las primeras covarianzas son matrices identidad
        self.covs = np.array([np.eye(d) for _ in range(self.k)])
        # Pesos de mezcla iniciales iguales         
        self.weights = np.ones(self.k) / self.k
        log_likelihood_old = 0

        # ------------------ INICIO DEL CICLO EM -----------------------------
        for it in range(self.max_iter):

            # ============ E-STEP ============
            # En el paso de expextation, calculamos las "responsabilidades":
            # r_ij = probabilidad de que el punto i pertenezca al cluster j
            resp = np.zeros((n, self.k))
            for i in range(n):
                for j in range(self.k):
                    # Fórmula completa del paso E:
                    # r_ij = π_j N(x_i | μ_j, Σ_j)
                    resp[i, j] = self.weights[j] * EMGaussianMixture.gaussian_pdf(
                        X[i], self.means[j], self.covs[j]
                    )
            # Normalizamos para que las responsabilidades de cada punto sumen 1
            resp /= resp.sum(axis=1, keepdims=True)

            # ============ M-STEP ============
            # Calculamos N_k = suma de responsabilidades por cluster

            Nk = resp.sum(axis=0)
            # Se actualizan los componente
            for j in range(self.k):
                #               Medias 
                # Nueva media: promedio ponderado por las responsabilidades
                self.means[j] = (resp[:, j].reshape(-1, 1) * X).sum(axis=0) / Nk[j]
                

                #           Covarianzas 
                # Covarianza: sumatoria de (r_ij * (x_i - μ)(x_i - μ)^T)
                diff = X - self.means[j]
                self.covs[j] = (resp[:, j].reshape(-1, 1) * diff).T @ diff / Nk[j]

            # ------------------------ PESOS -------------------------------
            # Peso de cada componente = N_k / total de puntos
            self.weights = Nk / n


            # ================================================================
            #                        LOG-LIKELIHOOD
            # ================================================================
            # Se usa para verificar que el algoritmo converge
            log_likelihood = 0
            for i in range(n):
                s = 0
                for j in range(self.k):
                    s += self.weights[j] * EMGaussianMixture.gaussian_pdf(
                        X[i], self.means[j], self.covs[j]
                    )
                log_likelihood += np.log(s)

            # criterio de parada
            if abs(log_likelihood - log_likelihood_old) < self.tol:
                # Si la mejora es muy pequeña, EM ha convergido                
                break

            log_likelihood_old = log_likelihood

        return self

    #                     PREDICCIÓN (ASIGNAR CLUSTERS)
    def predict(self, X):
        """
        Asigna cada punto al cluster con mayor responsabilidad.

        Devuelve etiquetas (0, 1, ..., k-1)
        """
        n = X.shape[0]
        resp = np.zeros((n, self.k))

        # Se calculan de nuevo las responsabilidades
        for i in range(n):
            for j in range(self.k):
                resp[i, j] = self.weights[j] * EMGaussianMixture.gaussian_pdf(
                    X[i], self.means[j], self.covs[j]
                )
        # Asignación final: cluster con mayor probabilidad
        return np.argmax(resp, axis=1)
