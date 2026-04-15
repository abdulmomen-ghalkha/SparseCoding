import torch
import numpy as np


# =========================================================
# Utilities
# =========================================================
def normalize_columns(D):
    return D / (torch.norm(D, dim=0, keepdim=True) + 1e-12)


def hard_group_topk(Z, k):
    """
    Keep k rows with largest L2 norm (group sparsity)
    """
    row_norms = torch.norm(Z, dim=1)
    idx = torch.argsort(row_norms, descending=True)[:k]

    Z_new = torch.zeros_like(Z)
    Z_new[idx, :] = Z[idx, :]
    return Z_new


# =========================================================
# Single-user ADMM Dictionary Learning
# =========================================================
class SingleUserADMMDict:

    def __init__(self, d, K, k_sparsity, rho=1.0, admm_iters=30, device=None):

        self.d = d
        self.K = K
        self.k = k_sparsity
        self.rho = rho
        self.admm_iters = admm_iters

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Dictionary initialization
        D = torch.randn(d, K, device=self.device)
        self.D = normalize_columns(D)


    # =====================================================
    # ADMM sparse coding (group k-sparsity)
    # =====================================================
    def admm_sparse_coding(self, X):

        d, n = X.shape
        X = X.to(self.device)
        S = torch.zeros(self.K, n, device=self.device)
        Z = torch.zeros_like(S).to(self.device)
        U = torch.zeros_like(S).to(self.device)

        DtD = self.D.T @ self.D
        DtX = self.D.T @ X

        A = DtD + self.rho * torch.eye(self.K, device=self.device)

        # numerical stability
        A_inv = torch.linalg.inv(A + 1e-8 * torch.eye(self.K, device=self.device))

        for _ in range(self.admm_iters):

            # ---- S update (quadratic solve)
            S = A_inv @ (DtX + self.rho * (Z - U))

            # ---- Z update (EXACT group sparsity projection)
            Z = hard_group_topk(S + U, self.k)

            # ---- dual update
            U = U + S - Z

        return Z


    # =====================================================
    # Dictionary update (stable MOD / least squares)
    # =====================================================
    def update_dictionary(self, X, S):

        # S S^T + regularization
        X = X.to(self.device)
        S = S.to(self.device)
        SS_T = S @ S.T
        SS_T = SS_T + 1e-6 * torch.eye(self.K, device=self.device)

        # MOD update
        D = X @ S.T @ torch.linalg.inv(SS_T)

        self.D = normalize_columns(D)


    # =====================================================
    # Training loop
    # =====================================================
    def fit(self, X, outer_iters=10):

        losses = []
        X = X.to(self.device)

        for t in range(outer_iters):

            # ---- sparse coding
            S = self.admm_sparse_coding(X)

            # ---- dictionary update
            self.update_dictionary(X, S)

            # ---- reconstruction loss
            recon = torch.norm(X - self.D @ S, p='fro')

            losses.append(recon.item())

            print(f"iter {t}: loss={recon.item():.6f}")

        return self.D, S, losses


    # =====================================================
    # Transform new data
    # =====================================================
    def transform(self, X):

        """
        Sparse coding using final learned dictionary
        """

        return self.admm_sparse_coding(X)


# =========================================================
# MULTI-USER DICTIONARY LEARNING
# =========================================================
class MultiUserADMMDict:

    def __init__(self, d, K, k_sparsity, rho=1.0, admm_iters=20, device=None):

        self.d = d
        self.K = K
        self.k = k_sparsity
        self.rho = rho
        self.admm_iters = admm_iters

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # shared dictionary
        D = torch.randn(d, K, device=self.device)
        self.D = normalize_columns(D)


    # =====================================================
    # LOCAL ADMM (each user)
    # =====================================================
    def admm_sparse_coding(self, X, D):

        d, n = X.shape

        S = torch.zeros(self.K, n, device=self.device)
        Z = torch.zeros_like(S)
        U = torch.zeros_like(S)

        DtD = D.T @ D
        DtX = D.T @ X

        A = DtD + self.rho * torch.eye(self.K, device=self.device)
        A_inv = torch.linalg.inv(A + 1e-8 * torch.eye(self.K, device=self.device))

        for _ in range(self.admm_iters):

            # S-update
            S = A_inv @ (DtX + self.rho * (Z - U))

            # Z-update (group sparsity)
            Z = hard_group_topk(S + U, self.k)

            # dual update
            U = U + S - Z

        return Z


    # =====================================================
    # SERVER STEP (dictionary update)
    # =====================================================
    def update_dictionary(self, X_list, S_list):

        """
        X_list: list of X_i
        S_list: list of S_i
        """

        num_users = len(X_list)

        A = torch.zeros((self.K, self.K), device=self.device)
        B = torch.zeros((self.d, self.K), device=self.device)

        for X_i, S_i in zip(X_list, S_list):

            A += S_i @ S_i.T
            B += X_i @ S_i.T

        # regularization
        A += 1e-6 * torch.eye(self.K, device=self.device)

        D = B @ torch.linalg.inv(A)

        self.D = normalize_columns(D)


    # =====================================================
    # TRAINING LOOP (FEDERATED STYLE)
    # =====================================================
    def fit(self, X_dict, outer_iters=10):

        """
        X_dict: {user_id: X_i}
        """

        losses = []

        for t in range(outer_iters):

            S_list = []
            X_list = []

            # -----------------------------
            # LOCAL UPDATE (parallelizable)
            # -----------------------------
            for user_id, X_i in X_dict.items():

                if isinstance(X_i, np.ndarray):
                    X_i = torch.tensor(X_i, device=self.device).float()
                else:
                    X_i = X_i.to(self.device)

                S_i = self.admm_sparse_coding(X_i, self.D)

                S_list.append(S_i)
                X_list.append(X_i)

            # -----------------------------
            # GLOBAL UPDATE (server)
            # -----------------------------
            self.update_dictionary(X_list, S_list)

            # -----------------------------
            # LOSS
            # -----------------------------
            recon = 0.0
            for X_i, S_i in zip(X_list, S_list):
                recon += torch.norm(X_i - self.D @ S_i, 'fro') ** 2

            losses.append(recon.item())

            print(f"iter {t}: loss={recon.item():.6f}")

        return self.D, S_list, losses


    # =====================================================
    # TRANSFORM
    # =====================================================
    def transform(self, X):

        return self.admm_sparse_coding(X, self.D)
    



def orthogonal_procrustes(A, B):
    """
    Solve: min_O || O A - B ||_F^2  s.t. O^T O = I
    """
    M = B @ A.T
    U, _, Vt = torch.linalg.svd(M)
    return U @ Vt


# =========================================================
# MAIN CLASS
# =========================================================
class MultiUserSheafDict:

    def __init__(
        self,
        d,
        K,
        edges,
        k_sparsity,
        rho=1.0,
        mu=0.1,
        admm_iters=20,
        update_O_every=True,
        device=None,
    ):

        self.d = d
        self.K = K
        self.edges = edges
        self.k = k_sparsity
        self.rho = rho
        self.mu = mu
        self.admm_iters = admm_iters
        self.update_O_every = update_O_every

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Shared dictionary
        D = torch.randn(d, K, device=self.device)
        self.D = normalize_columns(D)

        # Edge transport maps
        self.O = {}
        for (u, v) in self.edges:
            self.O[(u, v)] = torch.eye(d, device=self.device)

    # =====================================================
    # LOCAL ADMM (per user)
    # =====================================================
    def admm_sparse_coding(self, X):

        d, n = X.shape

        S = torch.zeros(self.K, n, device=self.device)
        Z = torch.zeros_like(S)
        U = torch.zeros_like(S)

        DtD = self.D.T @ self.D
        DtX = self.D.T @ X

        A = DtD + self.rho * torch.eye(self.K, device=self.device)
        A_inv = torch.linalg.inv(A + 1e-8 * torch.eye(self.K, device=self.device))

        for _ in range(self.admm_iters):

            # S-update
            S = A_inv @ (DtX + self.rho * (Z - U))

            # Z-update (group sparsity)
            Z = hard_group_topk(S + U, self.k)

            # Dual update
            U = U + S - Z

        return Z

    # =====================================================
    # DICTIONARY UPDATE (GLOBAL)
    # =====================================================
    def update_dictionary(self, X_list, S_list):

        A = torch.zeros((self.K, self.K), device=self.device)
        B = torch.zeros((self.d, self.K), device=self.device)

        # Reconstruction term
        for X_i, S_i in zip(X_list, S_list):
            A += S_i @ S_i.T
            B += X_i @ S_i.T

        # Sheaf coupling
        for (u, v) in self.edges:

            S_u = S_list[u]
            S_v = S_list[v]
            O_uv = self.O[(u, v)]

            A += self.mu * (S_u @ S_u.T + S_v @ S_v.T)

            B += self.mu * (
                O_uv @ (self.D @ S_u) @ S_u.T +
                (self.D @ S_v) @ S_v.T
            )

        A += 1e-6 * torch.eye(self.K, device=self.device)

        D = B @ torch.linalg.inv(A)
        self.D = normalize_columns(D)

    # =====================================================
    # UPDATE O (ALL EDGES)
    # =====================================================
    def update_O_all(self, S_list):

        for (u, v) in self.edges:

            S_u = S_list[u]
            S_v = S_list[v]

            A = self.D @ S_u
            B = self.D @ S_v

            self.O[(u, v)] = orthogonal_procrustes(A, B)

    # =====================================================
    # TRAINING LOOP
    # =====================================================
    def fit(self, X_dict, outer_iters=10):

        losses = []

        for t in range(outer_iters):

            S_list = []
            X_list = []

            # -------- LOCAL STEP --------
            for user_id, X_i in X_dict.items():

                if isinstance(X_i, np.ndarray):
                    X_i = torch.tensor(X_i, device=self.device).float()
                else:
                    X_i = X_i.to(self.device)

                S_i = self.admm_sparse_coding(X_i)

                S_list.append(S_i)
                X_list.append(X_i)

            # -------- GLOBAL UPDATE --------
            self.update_dictionary(X_list, S_list)

            # -------- O UPDATE --------
            if self.update_O_every:
                self.update_O_all(S_list)

            # -------- LOSS --------
            recon = 0.0
            sheaf_loss = 0.0

            for i, (X_i, S_i) in enumerate(zip(X_list, S_list)):
                recon += torch.norm(X_i - self.D @ S_i, 'fro') ** 2

            for (u, v) in self.edges:
                A = self.O[(u, v)] @ (self.D @ S_list[u])
                B = self.D @ S_list[v]
                sheaf_loss += torch.norm(A - B, 'fro') ** 2

            total_loss = recon + self.mu * sheaf_loss
            losses.append(total_loss.item())

            print(f"Iter {t}: Recon={recon.item():.4f}, Sheaf={sheaf_loss.item():.4f}")

        # Final O update (if delayed)
        if not self.update_O_every:
            self.update_O_all(S_list)

        return self.D, S_list, self.O, losses

    # =====================================================
    # TRANSFORM
    # =====================================================
    def transform(self, X):
        return self.admm_sparse_coding(X)