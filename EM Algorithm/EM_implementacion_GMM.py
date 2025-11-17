import numpy as np

class EMGaussianMixture:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.k = n_components
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def gaussian_pdf(x, mean, cov):
        d = mean.shape[0]
        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)
        norm = np.sqrt((2 * np.pi)**d * det)
        diff = x - mean
        return np.exp(-0.5 * diff.T @ inv @ diff) / norm

    def fit(self, X):
        n, d = X.shape
        
        # --- Inicialización ---
        self.means = X[np.random.choice(n, self.k, replace=False)]
        self.covs = np.array([np.eye(d) for _ in range(self.k)])
        self.weights = np.ones(self.k) / self.k

        log_likelihood_old = 0

        for it in range(self.max_iter):

            # ============ E-STEP ============
            resp = np.zeros((n, self.k))
            for i in range(n):
                for j in range(self.k):
                    resp[i, j] = self.weights[j] * EMGaussianMixture.gaussian_pdf(
                        X[i], self.means[j], self.covs[j]
                    )
            resp /= resp.sum(axis=1, keepdims=True)

            # ============ M-STEP ============
            Nk = resp.sum(axis=0)

            for j in range(self.k):
                # medias
                self.means[j] = (resp[:, j].reshape(-1, 1) * X).sum(axis=0) / Nk[j]
                
                # covarianzas
                diff = X - self.means[j]
                self.covs[j] = (resp[:, j].reshape(-1, 1) * diff).T @ diff / Nk[j]

            # pesos
            self.weights = Nk / n

            # ============ Log-Likelihood ============
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
                break

            log_likelihood_old = log_likelihood

        return self

    def predict(self, X):
        n = X.shape[0]
        resp = np.zeros((n, self.k))

        for i in range(n):
            for j in range(self.k):
                resp[i, j] = self.weights[j] * EMGaussianMixture.gaussian_pdf(
                    X[i], self.means[j], self.covs[j]
                )
        return np.argmax(resp, axis=1)
