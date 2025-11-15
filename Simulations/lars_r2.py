from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lars, lars_path

np.random.seed(42)

X, y = load_diabetes(return_X_y=True)

X_64 = np.zeros((442, 64), dtype=np.float64)
X_64[:, :10] = X
X_64[:, 19:64] = np.column_stack([X[:, i]*X[:, j] for i, j in combinations(range(10), 2)])
X_64[:, 10:19] = np.column_stack([X[:, i]**2 for i in range(10) if i != 1])
X_64 -= np.mean(X_64, axis=0)
X_64 /= np.sqrt((X_64**2).sum(axis=0))

y -= np.mean(y)

reg = Lars(n_nonzero_coefs=10, fit_intercept=False)
reg.fit(X_64, y)

beta_true = reg.coef_.copy()
mu = X_64 @ beta_true
e = y - mu

n = len(y)
n_sims = 100
K = 40

pe_matrix = np.zeros((n_sims, K))

for s in range(n_sims):
    e_star = np.random.choice(e, size=n, replace=True)
    y_star = e_star + mu

    alphas, active, coef_path, n_iter = lars_path(
        X_64, y_star, method="lar", max_iter=K, return_path=True, return_n_iter=True)

    for k in range(n_iter):
        beta_hat_k = coef_path[:, k]
        mu_hat_k = X_64 @ beta_hat_k

        pe = 1 - np.sum((mu_hat_k - mu) ** 2) / np.sum(mu ** 2)
        pe_matrix[s, k] = pe


pe_mean = pe_matrix.mean(axis=0)
pe_sd = pe_matrix.std(axis=0)

plt.plot(pe_mean, color="black")
plt.plot(pe_mean + pe_sd, color="black", linestyle="dashed")
plt.plot(pe_mean - pe_sd, color="black", linestyle="dashed")
plt.ylim(0.75, 1.0)

plt.xlabel("Average number of terms")
plt.ylabel("Proportion explained")

plt.show()


