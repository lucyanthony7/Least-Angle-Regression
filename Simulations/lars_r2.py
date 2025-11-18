from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lars, lars_path

np.random.seed(42)

X, y = load_diabetes(return_X_y=True)

# Create quadratic model with 64 predictors
X_64 = np.zeros((442, 64), dtype=np.float64)
X_64[:, :10] = X  # 10 main effects
X_64[:, 19:64] = np.column_stack([X[:, i]*X[:, j] for i, j in combinations(range(10), 2)])  # 45 interactions
X_64[:, 10:19] = np.column_stack([X[:, i]**2 for i in range(10) if i != 1])  # 9 square (skip x_2)
X_64 -= np.mean(X_64, axis=0)  # Centre data
X_64 /= np.sqrt((X_64**2).sum(axis=0))  # Scale data s.t. L2 norm is 1

y -= np.mean(y)

# Run initial regression with 10 steps, these are our "true mean" coefficients
reg = Lars(n_nonzero_coefs=10, fit_intercept=False)
reg.fit(X, y)

beta_true = reg.coef_.copy()
mu = X @ beta_true
e = y - mu

n = len(y)
n_sims = 100
K = 40

pe_matrix = np.zeros((n_sims, K))

#  Run bootstrapping
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

plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)

ax.plot(pe_mean, color="black")
ax.plot(pe_mean + pe_sd, color="black", linestyle="dashed")
ax.plot(pe_mean - pe_sd, color="black", linestyle="dashed")
ax.set_ylim(0.75, 1.0)

ax.set_xlabel("Average number of terms")
ax.set_ylabel("Proportion explained")

plt.savefig("figures/lars_pe.pdf")


