import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

class LARS:


    def __init__(self):
        self.coef_path = None
        self.alphas = None
        self.active_set = None
        self.n_iter = None


    def standardise_data(self, X, y):
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)
        X_std = (X - self.X_mean) / self.X_std

        self.y_mean = y.mean()
        y_centered = y - self.y_mean

        return X_std, y_centered
    

    def compute_correlations(self, X, y):
        return X.T @ (y - self.current_prediction)
    

    def find_max_correlation(self, correlations, active_set):
        abs_correlations = np.abs(correlations)
        abs_correlations[list(active_set)] = -1
        max_idx = np.argmax(abs_correlations)
        max_corr = correlations[max_idx]
        return max_idx, max_corr
    

    def compute_equiangular_vector(self, X, active_set, signs):
        if not active_set:
            return None, 0
        
        X_A = X[:, list(active_set)] * signs[list(active_set)]

        G_A = X_A.T @ X_A
        G_A_inv = np.linalg.inv(G_A)
        ones_A = np.ones(len(active_set))
        A_A = 1.0 / np.sqrt(ones_A.T @ G_A_inv @ ones_A)
        w_A = A_A * (G_A_inv @ ones_A)
        u_A = X_A @ w_A

        return u_A, A_A
    

    def compute_step_size(self, X, correlations, active_set, u_A, A_A, max_corr):
        if u_A is None:
            return abs(max_corr)
        
        a = X.T @ u_A
        min_gamma = float('inf')
        new_var = None

        for j in range(X.shape[1]):
            if j in active_set:
                continue

            gamma1 = (max_corr - correlations[j]) / (A_A - a[j])
            gamma2 = (max_corr + correlations[j]) / (A_A + a[j])

            for gamma in [gamma1, gamma2]:
                if gamma > 1e-10 and gamma < min_gamma:
                    min_gamma = gamma
                    new_var = j

        return min_gamma
    

    def fit(self, X, y, max_steps=None):
        n_samples, n_features = X.shape
        if max_steps is None:
            max_steps = n_features

        X_std, y_centered = self.standardise_data(X, y)

        self.current_prediction = np.zeros(n_samples)
        active_set = set()
        signs = np.zeros(n_features)

        coef_path = [np.zeros(n_features)]
        alphas = [np.inf]

        for step in range(max_steps):
            correlations = self.compute_correlations(X_std, y_centered)

            max_idx, max_corr = self.find_max_correlation(correlations, active_set)

            if abs(max_corr) < 1e-10:
                break

            active_set.add(max_idx)
            signs[max_idx] = np.sign(max_corr)

            u_A, A_A = self.compute_equiangular_vector(X_std, active_set, signs)

            gamma = self.compute_step_size(X_std, correlations, active_set, u_A, A_A, max_corr)

            if u_A is not None:
                self.current_prediction += gamma * u_A
            else:
                self.current_prediction += gamma * X_std[:, max_idx] * signs[max_idx]

            current_coef = np.zeros(n_features)
            for idx in active_set:
                current_coef[idx] = signs[idx] * np.linalg.lstsq(X_std[:, list(active_set)] * signs[list(active_set)],
                                                                 self.current_prediction, rcond=None)[0][list(active_set).index(idx)]
            coef_path.append(current_coef.copy())
            alphas.append(max_corr)

        self.coef_path = np.array(coef_path)
        self.alphas = np.array(alphas)
        self.active_set = active_set
        self.n_iter = len(coef_path) - 1

        return self
    

    def predict(self, X):
        X_std = (X - self.X_mean) / self.X_std
        return X_std @ self.coef_path[-1] + self.y_mean
    
    def plot_coef_path(self):
        if self.coef_path is None:
            raise ValueError("Model must be fitted first")
        
        plt.figure(figsize=(10, 6))
        n_steps = len(self.coef_path)
        
        for feature in range(self.coef_path.shape[1]):
            plt.plot(range(n_steps), self.coef_path[:, feature], 
                    label=f'Feature {feature}', linewidth=2)
        
        plt.xlabel('Step')
        plt.ylabel('Coefficient Value')
        plt.title('LARS Coefficient Paths')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def diabetes_example():
    np.random.seed(123)
    n_samples, n_features = 100, 6
    
    X = np.random.randn(n_samples, n_features)
    X[:, 2] = 0.7 * X[:, 0] + 0.3 * X[:, 2]
    X[:, 4] = 0.5 * X[:, 1] + 0.5 * X[:, 4]
    
    true_beta = np.array([1.5, 0, 2.0, 0, -1.0, 0])
    y = X @ true_beta + np.random.randn(n_samples) * 0.5
    
    print("=== Diabetes-style Example ===")
    print("True coefficients:", true_beta)
    
    lars = LARS()
    lars.fit(X, y)
    
    print("\nLARS coefficient path:")
    for step, coef in enumerate(lars.coef_path):
        print(f"Step {step}: {coef}")
    
    lars.plot_coef_path()
    
    final_coef = lars.coef_path[-1]
    print(f"\nSparsity recovery:")
    print(f"True zeros: {np.where(true_beta == 0)[0]}")
    print(f"Estimated zeros (|coef| < 0.1): {np.where(np.abs(final_coef) < 0.1)[0]}")


diabetes_example()