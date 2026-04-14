import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import Lasso


def update_atom(k, X, D, R):
    idx = np.where(R[k, :] != 0)[0]
    if len(idx) == 0:
        return None, None, k

    # Compute residual without atom k
    R_except = R.copy()
    R_except[k, :] = 0

    E = X[:, idx] - D @ R_except[:, idx]

    # SVD
    U, S, Vt = np.linalg.svd(E, full_matrices=False)

    new_atom = U[:, 0]
    new_coeffs = S[0] * Vt[0, :]

    return new_atom, new_coeffs, k



X_emb = np.load("./data/" + "mnist_embeddings.npy")
X_emb=X_emb.T[:, :500]
y = np.load("./data/" + "mnist_labels.npy")

print("Loaded embeddings:", X_emb.shape)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

class SparseCodingModel(nn.Module):
    def __init__(self, d, n_atoms):
        super().__init__()
        self.D = nn.Parameter(torch.randn(d, n_atoms))

    def forward(self, R):
        return self.D @ R




# ======================================================
# 3. FIXED DICTIONARY + LASSO
# ======================================================

d = X_emb.shape[0]
n_atoms = 256
N = X_emb.shape[1]

D_fixed = np.random.randn(d, n_atoms)
D_fixed /= np.linalg.norm(D_fixed, axis=0, keepdims=True)

model = Lasso(alpha=0.1)

model.fit(X=D_fixed, y=X_emb)

R_lasso = model.coef_.T
D_lasso = D_fixed.copy()
print("LASSO codes:", R_lasso.shape)


# ======================================================
# 3.b Learned DICTIONARY + LASSO
# ======================================================
model_lasso = SparseCodingModel(d, n_atoms).to(device)
optimizer_D = optim.Adam(model_lasso.parameters(), lr=1e-3)
lr_R = 1e-3


X_torch = torch.tensor(X_emb, dtype=torch.float32).to(device)
R_llasso = torch.randn(n_atoms, N, requires_grad=True, device=device)



def soft_threshold(z, alpha):
    return torch.sign(z) * torch.relu(torch.abs(z) - alpha)

# Joint Training Loop
for epoch in range(100):
    # --- Step 1: Update R (Sparse Coding) ---
    # We do a few "inner" iterations of ISTA
    with torch.no_grad():
        for _ in range(5):
            recon = model_lasso.D @ R_llasso
            grad_R = model_lasso.D.T @ (X_torch - recon)
            R_llasso = soft_threshold(R_llasso - lr_R * grad_R, 0.01)

    # --- Step 2: Update D (Dictionary Update) ---
    optimizer_D.zero_grad()
    recon = model_lasso.D @ R_llasso.detach() # Fix R
    loss_D = ((X_torch - recon) ** 2).mean()
    loss_D.backward()
    optimizer_D.step()

    # --- Step 3: Constrain D ---
    with torch.no_grad():
        model_lasso.D /= model_lasso.D.norm(dim=0, keepdim=True)
        
print(model_lasso.D.shape, R_llasso.shape)
# ======================================================
# 4. K-SVD STYLE
# ======================================================

def ksvd_update(X, D, R, n_iter=3, n_threads=8):
    d, N = X.shape
    K = D.shape[1]

    for it in range(n_iter):
        print("Iteration:", it)

        # Sparse coding
        R = np.linalg.pinv(D) @ X
        R[np.abs(R) < 0.1] = 0

        # Parallel dictionary update
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [
                executor.submit(update_atom, k, X, D, R)
                for k in range(K)
            ]

            for f in futures:
                new_atom, new_coeffs, k = f.result()
                if new_atom is None:
                    continue
                D[:, k] = new_atom
                idx = np.where(R[k, :] != 0)[0]
                R[k, idx] = new_coeffs

    return D, R

R_init = np.random.randn(n_atoms, N)
D_ksvd, R_ksvd = ksvd_update(X_emb, D_fixed.copy(), R_init)
print(f"Shape for KSVD: {D_ksvd.shape}, {R_ksvd.shape}")
print("K-SVD done")

# ======================================================
# 5. SGD-BASED DICTIONARY LEARNING
# ======================================================


model_sgd = SparseCodingModel(d, n_atoms).to(device)
optimizer = optim.Adam(model_sgd.parameters(), lr=1e-3)

X_torch = torch.tensor(X_emb, dtype=torch.float32).to(device)
R = torch.randn(n_atoms, N, requires_grad=True, device=device)

for epoch in range(30):
    optimizer.zero_grad()

    recon = model_sgd(R)
    loss = ((X_torch - recon) ** 2).mean() + 0.1 * torch.norm(R, 1)

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
    print(RRt.shape, Lambda.shape)
    inv = np.linalg.inv(RRt + Lambda)
    D = (X @ R.T) @ inv
    return D

lambdas = np.ones(n_atoms) * 0.1
R_sample = R_lasso.copy()
D_lagrange = lagrange_dictionary_update(X_emb, R_sample, lambdas)

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
    X_hat = D @ R  # (N, d)

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


errors_llasso, sparsity_llasso = compute_metrics(
    X_emb[:500],
    model_lasso.D.detach().cpu().numpy(),
    R_llasso.cpu().numpy(),
    name="Learned LASSO"
)

# ======================================================
# K-SVD EVALUATION
# ======================================================

# R_ksvd is (n_atoms, N) → transpose
R_ksvd_T = R_ksvd

errors_ksvd, sparsity_ksvd = compute_metrics(
    X_emb[:1000],
    D_ksvd,
    R_ksvd,
    name="K-SVD"
)


# ======================================================
# SGD EVALUATION
# ======================================================

with torch.no_grad():
    D_sgd = model_sgd.D.cpu().numpy()
    R_sgd = R.detach().cpu().numpy()  # (N, n_atoms)


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
print(f"Learned LASSO     | Error: {errors_llasso.mean():.4f} | Sparsity: {sparsity_llasso.mean():.2f}")
print(f"K-SVD     | Error: {errors_ksvd.mean():.4f} | Sparsity: {sparsity_ksvd.mean():.2f}")
print(f"SGD       | Error: {errors_sgd.mean():.4f} | Sparsity: {sparsity_sgd.mean():.2f}")
print(f"LAGRANGE  | Error: {errors_lagrange.mean():.4f} | Sparsity: {sparsity_lagrange.mean():.2f}")