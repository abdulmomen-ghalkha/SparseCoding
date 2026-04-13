import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import Lasso


X_emb = np.load("./data/" + "mnist_embeddings.npy")
y = np.load("./data/" + "mnist_labels.npy")

print("Loaded embeddings:", X_emb.shape)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# 3. FIXED DICTIONARY + LASSO
# ======================================================

d = X_emb.shape[1]
n_atoms = 128

D_fixed = np.random.randn(d, n_atoms)
D_fixed /= np.linalg.norm(D_fixed, axis=0, keepdims=True)

def sparse_code_lasso(X, D, alpha=0.05):
    codes = []
    for x in X[:500]:
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
        lasso.fit(D, x)
        codes.append(lasso.coef_)
    return np.array(codes)

R_lasso = sparse_code_lasso(X_emb, D_fixed)
print("LASSO codes:", R_lasso.shape)

# ======================================================
# 4. K-SVD STYLE
# ======================================================

def ksvd_update(X, D, R, n_iter=3):
    for _ in range(n_iter):
        R = np.linalg.pinv(D) @ X.T
        R[np.abs(R) < 0.1] = 0

        for k in range(D.shape[1]):
            idx = np.where(R[k, :] != 0)[0]
            if len(idx) == 0:
                continue

            E = X[idx].T - D @ R[:, idx] + np.outer(D[:, k], R[k, idx])
            U, S, Vt = np.linalg.svd(E, full_matrices=False)
            D[:, k] = U[:, 0]
            R[k, idx] = S[0] * Vt[0, :]

    return D, R

R_init = np.random.randn(n_atoms, X_emb.shape[0])
D_ksvd, R_ksvd = ksvd_update(X_emb[:1000], D_fixed.copy(), R_init)
print("K-SVD done")

# ======================================================
# 5. SGD-BASED DICTIONARY LEARNING
# ======================================================

class SparseCodingModel(nn.Module):
    def __init__(self, d, n_atoms):
        super().__init__()
        self.D = nn.Parameter(torch.randn(d, n_atoms))

    def forward(self, R):
        return self.D @ R

model_sgd = SparseCodingModel(d, n_atoms).to(device)
optimizer = optim.Adam(model_sgd.parameters(), lr=1e-1)

X_torch = torch.tensor(X_emb[:1000], dtype=torch.float32).to(device)
R = torch.randn(n_atoms, 1000, requires_grad=True, device=device)

for epoch in range(200):
    optimizer.zero_grad()

    recon = model_sgd(R)
    loss = ((recon.T - X_torch) ** 2).mean() + 0.1 * torch.norm(R, 1)

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        model_sgd.D /= model_sgd.D.norm(dim=0, keepdim=True)

    if epoch % 10 == 0:
        print(f"SGD Epoch {epoch}, Loss: {loss.item():.4f}")

# ======================================================
# 6. LAGRANGE MULTIPLIER METHOD
# ======================================================

def lagrange_dictionary_update(X, R, lambdas):
    RRt = R @ R.T
    Lambda = np.diag(lambdas)
    inv = np.linalg.inv(RRt + Lambda)
    D = (X.T @ R.T) @ inv
    return D.T

lambdas = np.ones(n_atoms) * 0.1
R_sample = R_lasso.T
D_lagrange = lagrange_dictionary_update(X_emb[:500], R_sample, lambdas)

print("Lagrange dictionary shape:", D_lagrange.shape)

# ======================================================
# DONE
# ======================================================
print("Pipeline: Supervised encoder -> embeddings -> sparse coding")



# ======================================================
# 7. EVALUATION METRICS
# ======================================================

def compute_metrics(X, D, R, name=""):
    """
    X: (N, d)
    D: (d, n_atoms)
    R: (N, n_atoms)
    """
    # Reconstruction
    X_hat = R @ D.T  # (N, d)

    # Reconstruction error per sample
    
    
    errors = (
        np.linalg.norm(X - X_hat, axis=1) /
        np.linalg.norm(X, axis=1)
    ) * 100


    # Sparsity (L0)
    sparsity = np.sum(np.abs(R) > 1e-4, axis=1)

    # Sparsity ratio
    sparsity_ratio = np.mean(np.abs(R) > 1e-4)

    print(f"\n===== {name} =====")
    print(f"Avg reconstruction error: {errors.mean():.4f}")
    print(f"Std reconstruction error: {errors.std():.4f}")
    print(f"Avg sparsity (#nonzeros): {sparsity.mean():.2f}")
    print(f"Min/Max sparsity: {sparsity.min()} / {sparsity.max()}")
    print(f"Sparsity ratio: {sparsity_ratio:.4f}")

    return errors, sparsity


# ======================================================
# LASSO EVALUATION
# ======================================================

errors_lasso, sparsity_lasso = compute_metrics(
    X_emb[:500],
    D_fixed,
    R_lasso,
    name="LASSO"
)


# ======================================================
# K-SVD EVALUATION
# ======================================================

# R_ksvd is (n_atoms, N) → transpose
R_ksvd_T = R_ksvd.T

errors_ksvd, sparsity_ksvd = compute_metrics(
    X_emb[:1000],
    D_ksvd,
    R_ksvd_T,
    name="K-SVD"
)


# ======================================================
# SGD EVALUATION
# ======================================================

with torch.no_grad():
    D_sgd = model_sgd.D.cpu().numpy()
    R_sgd = R.detach().cpu().numpy().T  # (N, n_atoms)

errors_sgd, sparsity_sgd = compute_metrics(
    X_emb[:1000],
    D_sgd,
    R_sgd,
    name="SGD"
)


# ======================================================
# LAGRANGE EVALUATION
# ======================================================

# Uses LASSO codes with new dictionary
errors_lagrange, sparsity_lagrange = compute_metrics(
    X_emb[:500],
    D_lagrange,
    R_lasso,
    name="LAGRANGE"
)


# ======================================================
# SUMMARY TABLE
# ======================================================

print("\n========= SUMMARY =========")
print(f"LASSO     | Error: {errors_lasso.mean():.4f} | Sparsity: {sparsity_lasso.mean():.2f}")
print(f"K-SVD     | Error: {errors_ksvd.mean():.4f} | Sparsity: {sparsity_ksvd.mean():.2f}")
print(f"SGD       | Error: {errors_sgd.mean():.4f} | Sparsity: {sparsity_sgd.mean():.2f}")
print(f"LAGRANGE  | Error: {errors_lagrange.mean():.4f} | Sparsity: {sparsity_lagrange.mean():.2f}")