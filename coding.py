import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit

# ======================================================
# 1. LOAD DATA
# ======================================================

X_emb = np.load("./data/mnist_embeddings.npy")  # (N, d)
X_emb = X_emb[:2000].T  # -> (d, N)

d, N = X_emb.shape
n_atoms = 256

print("Data shape:", X_emb.shape)

# ======================================================
# 2. INITIAL DICTIONARY
# ======================================================

def normalize_columns(D):
    return D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-8)

D_init = normalize_columns(np.random.randn(d, n_atoms))

# ======================================================
# 3. LASSO SPARSE CODING (CORRECT)
# ======================================================

def compute_lasso_codes(D, X, alpha=0.1):
    K = D.shape[1]
    N = X.shape[1]
    R = np.zeros((K, N))

    for i in range(N):
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
        model.fit(D, X[:, i])
        R[:, i] = model.coef_

    return R

R_lasso = compute_lasso_codes(D_init, X_emb)
D_lasso = D_init.copy()

# ======================================================
# 4. LEARNED DICTIONARY (ISTA + SGD)
# ======================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

class SparseCodingModel(nn.Module):
    def __init__(self, d, K):
        super().__init__()
        self.D = nn.Parameter(torch.randn(d, K))

    def forward(self, R):
        return self.D @ R

def soft_threshold(z, alpha):
    return torch.sign(z) * torch.relu(torch.abs(z) - alpha)

model = SparseCodingModel(d, n_atoms).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

X_torch = torch.tensor(X_emb, dtype=torch.float32).to(device)
R = torch.randn(n_atoms, N, device=device)

lr_R = 1e-3

for epoch in range(500):

    # --- ISTA update for R ---
    with torch.no_grad():
        for _ in range(20):
            recon = model.D @ R
            grad_R = - model.D.T @ (X_torch - recon)  # FIXED SIGN
            R = soft_threshold(R - lr_R * grad_R, 0.1)

    # --- Update D ---
    optimizer.zero_grad()
    recon = model.D @ R.detach()
    loss = ((X_torch - recon) ** 2).mean()
    loss.backward()
    optimizer.step()

    # Normalize D
    with torch.no_grad():
        model.D[:] = model.D / (model.D.norm(dim=0, keepdim=True) + 1e-8)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

D_learned = model.D.detach().cpu().numpy()
R_learned = R.detach().cpu().numpy()

# ======================================================
# 5. K-SVD (CORRECT VERSION)
# ======================================================

def ksvd(X, D, n_iter=5, sparsity=64):
    d, N = X.shape
    K = D.shape[1]

    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)

    for it in range(n_iter):
        print(f"K-SVD Iteration {it}")

        # --- Sparse coding (OMP) ---
        R = np.zeros((K, N))
        for i in range(N):
            omp.fit(D, X[:, i])
            R[:, i] = omp.coef_

        # --- Dictionary update ---
        for k in range(K):
            idx = np.where(R[k, :] != 0)[0]
            if len(idx) == 0:
                continue

            Rk = R.copy()
            Rk[k, :] = 0

            E = X[:, idx] - D @ Rk[:, idx]

            U, S, Vt = np.linalg.svd(E, full_matrices=False)

            D[:, k] = U[:, 0]
            R[k, idx] = S[0] * Vt[0, :]

        D = normalize_columns(D)

    return D, R

D_ksvd, R_ksvd = ksvd(X_emb, D_init.copy())

# ======================================================
# 6. LAGRANGE DICTIONARY UPDATE
# ======================================================

def lagrange_dictionary_update(X, R, lambdas):
    RRt = R @ R.T
    Lambda = np.diag(lambdas)

    inv = np.linalg.pinv(RRt + Lambda)  # more stable
    D = (X @ R.T) @ inv

    return normalize_columns(D)

lambdas = np.ones(n_atoms) * 0.1
D_lagrange = lagrange_dictionary_update(X_emb, R_lasso, lambdas)

# ======================================================
# 7. METRICS (FIXED)
# ======================================================

def compute_metrics(X, D, R, name=""):
    X_hat = D @ R  # (d, N)

    errors = (
        np.linalg.norm(X - X_hat, axis=0) /
        np.linalg.norm(X, axis=0)
    ) * 100

    sparsity = np.sum(np.abs(R) > 1e-4, axis=0)
    sparsity_ratio = np.mean(np.abs(R) > 1e-4)

    print(f"\n===== {name} =====")
    print(f"Avg error: {errors.mean():.4f}")
    print(f"Std error: {errors.std():.4f}")
    print(f"Avg sparsity: {sparsity.mean():.2f}")
    print(f"Sparsity ratio: {sparsity_ratio:.4f}")

    return errors, sparsity

# ======================================================
# 8. EVALUATION
# ======================================================

errors_lasso, _ = compute_metrics(X_emb, D_lasso, R_lasso, "LASSO")
errors_learned, _ = compute_metrics(X_emb, D_learned, R_learned, "LEARNED")
errors_ksvd, _ = compute_metrics(X_emb, D_ksvd, R_ksvd, "K-SVD")
errors_lagrange, _ = compute_metrics(X_emb, D_lagrange, R_lasso, "LAGRANGE")

# ======================================================
# 9. SUMMARY
# ======================================================

print("\n===== SUMMARY =====")
print(f"LASSO     : {errors_lasso.mean():.4f}")
print(f"LEARNED   : {errors_learned.mean():.4f}")
print(f"K-SVD     : {errors_ksvd.mean():.4f}")
print(f"LAGRANGE  : {errors_lagrange.mean():.4f}")