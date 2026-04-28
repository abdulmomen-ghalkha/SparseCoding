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
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    O = U @ Vh
    return O

""" 
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
 """    




# =====================================================
# Main class
# =====================================================

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
        dict_step=1.0,
        parallel_updates=True,
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
        self.dict_step = dict_step
        self.parallel_updates = parallel_updates
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        D = torch.randn(d, K, device=self.device)
        self.D = normalize_columns(D)

        self.O = {}
        for (u, v) in self.edges:
            self.O[(u, v)] = torch.eye(d, device=self.device)

        self.user_to_idx = None
        self.idx_to_user = None
        self.neighbors = None
        self.degree = None

    # -----------------------------------------------------------------
    def admm_sparse_coding(self, X, neigh_info):
        n = X.shape[1]
        S = torch.zeros(self.K, n, device=self.device)
        Z = torch.zeros_like(S)
        U = torch.zeros_like(S)

        DtD = self.D.T @ self.D
        DtX = self.D.T @ X
        I = torch.eye(self.K, device=self.device)

        deg_i = len(neigh_info)
        A = (1.0 + 2.0 * self.mu * deg_i) * DtD + self.rho * I
        B = DtX.clone()
        for (_, O_ij, S_j) in neigh_info:
            B += 2.0 * self.mu * (self.D.T @ O_ij.T @ self.D @ S_j)

        A_inv = torch.linalg.inv(A + 1e-8 * I)
        for _ in range(self.admm_iters):
            S = A_inv @ (B + self.rho * (Z - U))
            Z = hard_group_topk(S + U, self.k)
            U = U + S - Z
        return Z

    # -----------------------------------------------------------------
    def update_dictionary(self, X_list, S_list):
        A = torch.zeros((self.K, self.K), device=self.device)
        B = torch.zeros((self.d, self.K), device=self.device)

        for X_i, S_i in zip(X_list, S_list):
            A += S_i @ S_i.T
            B += X_i @ S_i.T

        for (u, v) in self.edges:
            i = self.user_to_idx[u]
            j = self.user_to_idx[v]
            S_i = S_list[i]
            S_j = S_list[j]
            O_uv = self.O[(u, v)]

            DSi = self.D @ S_i
            DSj = self.D @ S_j

            A += 2.0 * self.mu * (S_i @ S_i.T + S_j @ S_j.T)
            B += 2.0 * self.mu * (O_uv.T @ DSj @ S_i.T + O_uv @ DSi @ S_j.T)

        A += 1e-8 * torch.eye(self.K, device=self.device)
        D_new = B @ torch.linalg.inv(A)
        D_new = normalize_columns(D_new)
        self.D = (1 - self.dict_step) * self.D + self.dict_step * D_new
        self.D = normalize_columns(self.D)

    # -----------------------------------------------------------------
    def update_O_all(self, S_list):
        for (u, v) in self.edges:
            i = self.user_to_idx[u]
            j = self.user_to_idx[v]
            A = self.D @ S_list[i]
            B = self.D @ S_list[j]
            self.O[(u, v)] = orthogonal_procrustes(A, B)

    # -----------------------------------------------------------------
    def fit(self, X_dict, outer_iters=10):
        # -------- Indexing --------
        self.user_to_idx = {u: i for i, u in enumerate(X_dict.keys())}
        self.idx_to_user = list(X_dict.keys())
        N = len(X_dict)

        # -------- Data tensors --------
        X_list = [None] * N
        for user, idx in self.user_to_idx.items():
            X = X_dict[user]
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, device=self.device).float()
            else:
                X = X.to(self.device).float()
            X_list[idx] = X

        # -------- Build neighbors & degrees (using indices) --------
        self.neighbors = {i: [] for i in range(N)}
        self.degree = {i: 0 for i in range(N)}
        for (u, v) in self.edges:
            if u not in self.user_to_idx or v not in self.user_to_idx:
                continue
            i, j = self.user_to_idx[u], self.user_to_idx[v]
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)
            self.degree[i] += 1
            self.degree[j] += 1

        # -------- Initial S --------
        S_list = [torch.zeros(self.K, X_list[i].shape[1], device=self.device) for i in range(N)]
        losses = []

        for t in range(outer_iters):
            # -------- Update S_i --------
            if self.parallel_updates:
                # Jacobi: use previous S_list for all neighbours
                prev_S = S_list
                new_S = [None] * N
                for i in range(N):
                    neigh_info = []
                    for j in self.neighbors[i]:
                        user_i = self.idx_to_user[i]
                        user_j = self.idx_to_user[j]
                        if (user_i, user_j) in self.O:
                            O_ij = self.O[(user_i, user_j)]
                        else:
                            O_ij = self.O[(user_j, user_i)].T
                        neigh_info.append((j, O_ij, prev_S[j]))
                    new_S[i] = self.admm_sparse_coding(X_list[i], neigh_info)
                S_list = new_S
            else:
                # Gauss-Seidel: update in place
                for i in range(N):
                    neigh_info = []
                    for j in self.neighbors[i]:
                        user_i = self.idx_to_user[i]
                        user_j = self.idx_to_user[j]
                        if (user_i, user_j) in self.O:
                            O_ij = self.O[(user_i, user_j)]
                        else:
                            O_ij = self.O[(user_j, user_i)].T
                        neigh_info.append((j, O_ij, S_list[j]))
                    S_list[i] = self.admm_sparse_coding(X_list[i], neigh_info)

            # -------- Dictionary update --------
            self.update_dictionary(X_list, S_list)

            # -------- O update --------
            if self.update_O_every:
                self.update_O_all(S_list)

            # -------- Loss --------
            recon = sum(torch.norm(X_list[i] - self.D @ S_list[i], 'fro')**2 for i in range(N))
            sheaf = 0.0
            for (u, v) in self.edges:
                i = self.user_to_idx[u]
                j = self.user_to_idx[v]
                O_uv = self.O[(u, v)]
                sheaf += torch.norm(O_uv @ (self.D @ S_list[i]) - self.D @ S_list[j], 'fro')**2
            total = recon + self.mu * sheaf
            losses.append(total.item())
            print(f"Iter {t:3d}: recon={recon.item():.4f} sheaf={sheaf.item():.4f} total={total.item():.4f}")

        if not self.update_O_every:
            self.update_O_all(S_list)

        return self.D, S_list, self.O, losses

    # -----------------------------------------------------------------
    def transform(self, X, neigh_info=None):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, device=self.device).float()
        else:
            X = X.to(self.device).float()
        if neigh_info is None:
            neigh_info = []
        return self.admm_sparse_coding(X, neigh_info)
    



class MultiUserSheafFixedDict:

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
        with torch.no_grad():

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
    



## -------------------------------------------------


import torch

# =========================================================
# Helpers
# =========================================================
def normalize_columns(D):
    return D / (torch.norm(D, dim=0, keepdim=True) + 1e-8)


def hard_group_topk(S, k):
    row_norms = torch.norm(S, dim=1)
    idx = torch.topk(row_norms, k=k).indices

    Z = torch.zeros_like(S)
    Z[idx] = S[idx]
    return Z


# =========================================================
# Single-user SCA (NO inner ADMM loops)
# =========================================================
class SingleUserSCADict:

    def __init__(self, d, K, k_sparsity, rho=1.0, alpha=1.0, gamma=0.5, beta=0.5, device=None):

        self.d = d
        self.K = K
        self.k = k_sparsity
        self.rho = rho
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize variables
        D = torch.randn(d, K, device=self.device)
        self.D = normalize_columns(D)

        self.P = self.D.clone()
        self.U = torch.zeros_like(self.D)

    # =====================================================
    # S update (one-shot, from closed form)
    # =====================================================
    def update_S(self, X, S, Z, V):

        DtD = self.D.T @ self.D
        DtX = self.D.T @ X

        Q = DtD + self.rho * torch.eye(self.K, device=self.device)
        R = DtX + self.rho * (Z - V) # Possible error (transpose)

        Q_inv = torch.linalg.inv(Q + 1e-8 * torch.eye(self.K, device=self.device))
        S_tilde = Q_inv @ R

        # SCA smoothing
        S_new = S + self.alpha * (S_tilde - S)

        return S_new

    # =====================================================
    # Z update (projection)
    # =====================================================
    def update_Z(self, S, V):
        return hard_group_topk(S + V, self.k)

    # =====================================================
    # D update (from Eq. DA = B)
    # =====================================================
    def update_D(self, X, S):

        SS_T = S @ S.T
        A = SS_T + self.rho * torch.eye(self.K, device=self.device)

        B = X @ S.T + self.rho * (self.P - self.U)

        A_inv = torch.linalg.inv(A + 1e-8 * torch.eye(self.K, device=self.device))
        D_tilde = B @ A_inv

        # SCA smoothing
        D_new = self.D + self.alpha * (D_tilde - self.D)

        return D_new

    # =====================================================
    # P update (projection onto OB)
    # =====================================================
    def update_P(self, D):

        H = D + self.U
        return normalize_columns(H)

    # =====================================================
    # Dual updates
    # =====================================================
    def update_duals(self, D, P, S, Z, V):

        self.U = self.U + D - P
        V = V + S - Z

        return V
    

    def update_alpha(self, iter, alpha, gamma, beta):
        gamma = gamma
        beta = iter * beta
        alpha = (alpha * gamma) / (1 + beta)
        return alpha, gamma, beta
    # =====================================================
    # Training loop
    # =====================================================
    def fit(self, X, outer_iters=20):

        X = X.to(self.device)
        d, n = X.shape

        # Initialize S, Z, V
        S = torch.zeros(self.K, n, device=self.device)
        Z = torch.zeros_like(S)
        V = torch.zeros_like(S)

        losses = []

        for t in range(1, outer_iters):

            # ---- S update
            S_new = self.update_S(X, S, Z, V)

            # ---- D update
            D_new = self.update_D(X, S)
            S = S_new
            self.D = D_new


            # ---- Z update
            Z = self.update_Z(S, V)


            # ---- P update
            P_new = self.update_P(D_new)

            # ---- dual updates
            V = self.update_duals(D_new, P_new, S, Z, V)

            # assign updates
            self.P = P_new
            # ---- loss
            recon = torch.norm(X - self.D @ S, p='fro')
            losses.append(recon.item())

            print(f"iter {t}: loss={recon.item():.6f}")

            self.alpha, self.gamma, self.beta = self.update_alpha(t, self.alpha, self.gamma, self.beta)
            print(self.alpha,self.gamma, self.beta)
        S = hard_group_topk(S, self.k)
        return self.D, S, losses

    # =====================================================
    # Transform
    # =====================================================
    def transform(self, X):

        X = X.to(self.device)

        S = torch.zeros(self.K, X.shape[1], device=self.device)
        Z = torch.zeros_like(S)
        V = torch.zeros_like(S)

        for _ in range(10):
            S = self.update_S(X, S, Z, V)
            Z = self.update_Z(S, V)
            V = V + S - Z

        return Z
    


# =========================================================
# Multi-user SCA Dictionary Learning (FIXED)
# =========================================================
class MultiUserSCADict:

    def __init__(self, d, K, k_sparsity, edges, mu=0.1,
                 rho=1.0, alpha=1.0, gamma=0.5, beta=0.5, num_fp_iters=10, tol=1e-4, gamma_D=1e-2, device=None):

        self.d = d
        self.K = K
        self.k = k_sparsity
        self.edges = edges
        self.mu = mu
        self.rho = rho
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.num_fp_iters = num_fp_iters
        self.tol = tol
        self.gamma_D = gamma_D

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ---- shared dictionary
        D = torch.randn(d, K, device=self.device)
        self.D = normalize_columns(D)

        self.P = self.D.clone()
        self.U = torch.zeros_like(self.D)

        # ---- initialize O_uv
        self.O = {(u, v): torch.eye(d, device=self.device) for (u, v) in edges}

    # =====================================================
    # S update (CORRECT SCA + sheaf + ADMM form)
    # =====================================================
    def update_S(self, i, X_dict, S_dict, Z_dict, V_dict):

        X_i = X_dict[i]
        S_i = S_dict[i]
        Z_i = Z_dict[i]
        V_i = V_dict[i]

        DtD = self.D.T @ self.D
        DtX = self.D.T @ X_i

        # degree
        delta_i = sum([1 for (u, v) in self.edges if u == i or v == i])

        Q = (1 + 2 * self.mu * delta_i) * DtD + self.rho * torch.eye(self.K, device=self.device)

        R = DtX + self.rho * (Z_i - V_i)

        # ---- sheaf terms (CORRECT)
        #for (u, v) in self.edges:
        #    if u == i:
        #        R += 2 * self.mu * self.D.T @ self.O[(u, v)].T @ self.D @ S_dict[v]
        #    elif v == i:
        #        R += 2 * self.mu * self.D.T @ self.O[(u, v)] @ self.D @ S_dict[u]

        Q_inv = torch.linalg.inv(Q + 1e-8 * torch.eye(self.K, device=self.device))
        S_tilde = Q_inv @ R

        return S_i + self.alpha * (S_tilde - S_i)

    # =====================================================
    # D update (FIXED-POINT + EARLY STOPPING)
    # =====================================================
    def update_D(self, X_dict, S_dict, num_fp_iters=10, tol=1e-4):

        # ---- build A
        A = torch.zeros((self.K, self.K), device=self.device)

        for i in X_dict:
            S = S_dict[i]
            A += S @ S.T

        for (u, v) in self.edges:
            A += 2 * self.mu * (S_dict[u] @ S_dict[u].T + S_dict[v] @ S_dict[v].T)

        A += self.rho * torch.eye(self.K, device=self.device)

        # ---- build B
        B = torch.zeros((self.d, self.K), device=self.device)

        for i in X_dict:
            B += X_dict[i] @ S_dict[i].T

        B += self.rho * (self.P - self.U) + 2 * self.gamma_D * self.D @ torch.linalg.inv(self.D.T @ self.D)

        A_inv = torch.linalg.inv(A + 1e-8 * torch.eye(self.K, device=self.device))

        # ---- FIXED-POINT ITERATION
        D_fp = self.D.clone()

        for k in range(num_fp_iters):

            D_prev = D_fp.clone()

            sheaf_term = torch.zeros_like(D_fp)

            for (u, v) in self.edges:
                Su = S_dict[u]
                Sv = S_dict[v]
                # Can be precomputed to acceleraed the training
                sheaf_term += (
                    self.O[(u, v)].T @ D_fp @ Sv @ Su.T +
                    self.O[(u, v)] @ D_fp @ Su @ Sv.T
                )

            # ---- update
            #D_fp = (B + 2 * self.mu * 0 * sheaf_term) @ A_inv
            D_fp = B @ A_inv
            # ---- convergence check
            diff = torch.norm(D_fp - D_prev, p='fro')
            #print(diff)
            if diff < tol:
                # optional debug
                print(f"D fixed-point converged at iter {k}, diff={diff.item():.6e}")
                break

        # ---- SCA smoothing
        D_new = self.D + self.alpha * (D_fp - self.D)

        return D_new

    # =====================================================
    # O update (Procrustes)
    # =====================================================
    def update_O(self, S_dict):

        for (u, v) in self.edges:

            A = self.D @ S_dict[u]
            B = self.D @ S_dict[v]

            M = B @ A.T
            U, _, Vt = torch.linalg.svd(M)

            self.O[(u, v)] = U @ Vt

    # =====================================================
    # Z update (GROUP SPARSITY on S^T)
    # =====================================================
    def update_Z(self, S, V):
        return hard_group_topk(S + V, self.k)

    # =====================================================
    # P update
    # =====================================================
    def update_P(self, D):
        return normalize_columns(D + self.U)

    # =====================================================
    # Dual updates (FIXED TRANSPOSE)
    # =====================================================
    def update_duals(self, D, P, S_dict, Z_dict, V_dict):

        self.U = self.U + D - P

        for i in S_dict:
            V_dict[i] = V_dict[i] + S_dict[i] - Z_dict[i]

        return V_dict

    # =====================================================
    # Step-size schedule
    # =====================================================
    def update_alpha(self, t):
        self.beta = t * self.beta
        self.alpha = (self.alpha * self.gamma) / (1 + self.beta)

    # =====================================================
    # Training loop (torch.no_grad)
    # =====================================================
    def fit(self, X_dict, outer_iters=20):

        X_dict = {i: X.to(self.device) for i, X in X_dict.items()}

        # ---- initialize
        S_dict = {i: torch.zeros(self.K, X.shape[1], device=self.device) for i, X in X_dict.items()}
        S_dict_new = {i: torch.zeros(self.K, X.shape[1], device=self.device) for i, X in X_dict.items()}
        Z_dict = {i: torch.zeros_like(S_dict[i]) for i in X_dict}
        V_dict = {i: torch.zeros_like(S_dict[i]) for i in X_dict}

        losses = []

        for t in range(1, outer_iters):

            with torch.no_grad():

                # ---- S updates
                for i in X_dict:
                    S_dict_new[i] = self.update_S(i, X_dict, S_dict, Z_dict, V_dict)

                # ---- D update
                D_new = self.update_D(X_dict, S_dict, num_fp_iters=self.num_fp_iters, tol=self.tol)

                for i in X_dict:
                    S_dict[i] = S_dict_new[i].detach().clone()
                
                # ---- O updates
                #self.update_O(S_dict)

                # ---- Z updates
                for i in X_dict:
                    Z_dict[i] = self.update_Z(S_dict[i], V_dict[i])

                # ---- P update
                P_new = self.update_P(D_new)

                # ---- dual updates
                V_dict = self.update_duals(D_new, P_new, S_dict, Z_dict, V_dict)

                # assign
                self.D = D_new
                self.P = P_new

                # ---- loss
                loss = 0
                for i in X_dict:
                    loss += torch.norm(X_dict[i] - self.D @ S_dict[i], p='fro')**2

                for (u, v) in self.edges:
                    loss += 0 * self.mu * torch.norm(
                        self.O[(u, v)] @ self.D @ S_dict[u] - self.D @ S_dict[v], p='fro'
                    )**2

                losses.append(loss.item())

                print(f"iter {t}: loss={loss.item():.6f}")

                # step size update
                self.update_alpha(t)
        self.update_O(S_dict)
        for i in S_dict:
            S_dict[i] = hard_group_topk(S_dict[i], self.k)
        return self.D, S_dict, losses




# =========================================================
# Multi-user Sheaf SCA Dictionary Learning (FULL MODEL)
# =========================================================
class MultiUserSheafSCADict:

    def __init__(self, d, K, k_sparsity, edges, mu=0.1,
                 rho=1.0, alpha=1.0, gamma=0.5, beta=0.5, num_fp_iters=10, tol=1e-4, device=None):

        self.d = d
        self.K = K
        self.k = k_sparsity
        self.edges = edges
        self.mu = mu
        self.rho = rho
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.num_fp_iters = num_fp_iters
        self.tol = tol

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ---- shared dictionary
        D = torch.randn(d, K, device=self.device)
        self.D = normalize_columns(D)

        self.P = self.D.clone()
        self.U = torch.zeros_like(self.D)

        # ---- initialize O_uv
        self.O = {(u, v): torch.eye(d, device=self.device) for (u, v) in edges}

    # =====================================================
    # S update (CORRECT SCA + sheaf + ADMM form)
    # =====================================================
    def update_S(self, i, X_dict, S_dict, Z_dict, V_dict):

        X_i = X_dict[i]
        S_i = S_dict[i]
        Z_i = Z_dict[i]
        V_i = V_dict[i]

        DtD = self.D.T @ self.D
        DtX = self.D.T @ X_i

        # degree
        delta_i = sum([1 for (u, v) in self.edges if u == i or v == i])

        Q = (1 + 2 * self.mu * delta_i) * DtD + self.rho * torch.eye(self.K, device=self.device)

        R = DtX + self.rho * (Z_i - V_i)

        # ---- sheaf terms (CORRECT)
        for (u, v) in self.edges:
            if u == i:
                R += 2 * self.mu * self.D.T @ self.O[(u, v)].T @ self.D @ S_dict[v]
            elif v == i:
                R += 2 * self.mu * self.D.T @ self.O[(u, v)] @ self.D @ S_dict[u]

        Q_inv = torch.linalg.inv(Q + 1e-8 * torch.eye(self.K, device=self.device))
        S_tilde = Q_inv @ R

        return S_i + self.alpha * (S_tilde - S_i)

    # =====================================================
    # D update (FIXED-POINT + EARLY STOPPING)
    # =====================================================
    def update_D(self, X_dict, S_dict, num_fp_iters=10, tol=1e-4):

        # ---- build A
        A = torch.zeros((self.K, self.K), device=self.device)

        for i in X_dict:
            S = S_dict[i]
            A += S @ S.T

        for (u, v) in self.edges:
            A += 2 * self.mu * (S_dict[u] @ S_dict[u].T + S_dict[v] @ S_dict[v].T)

        A += self.rho * torch.eye(self.K, device=self.device)

        # ---- build B
        B = torch.zeros((self.d, self.K), device=self.device)

        for i in X_dict:
            B += X_dict[i] @ S_dict[i].T

        B += self.rho * (self.P - self.U)

        A_inv = torch.linalg.inv(A + 1e-8 * torch.eye(self.K, device=self.device))

        # ---- FIXED-POINT ITERATION
        D_fp = self.D.clone()

        for k in range(num_fp_iters):

            D_prev = D_fp.clone()

            sheaf_term = torch.zeros_like(D_fp)

            for (u, v) in self.edges:
                Su = S_dict[u]
                Sv = S_dict[v]
                # Can be precomputed to acceleraed the training
                sheaf_term += (
                    self.O[(u, v)].T @ D_fp @ Sv @ Su.T +
                    self.O[(u, v)] @ D_fp @ Su @ Sv.T
                )

            # ---- update
            D_fp = (B + 2 * self.mu * sheaf_term) @ A_inv
            #D_fp = B @ A_inv
            # ---- convergence check
            diff = torch.norm(D_fp - D_prev, p='fro')
            #print(diff)
            if diff < tol:
                # optional debug
                print(f"D fixed-point converged at iter {k}, diff={diff.item():.6e}")
                break

        # ---- SCA smoothing
        D_new = self.D + self.alpha * (D_fp - self.D)

        return D_new

    # =====================================================
    # O update (Procrustes)
    # =====================================================
    def update_O(self, S_dict):

        for (u, v) in self.edges:

            A = self.D @ S_dict[u]
            B = self.D @ S_dict[v]

            M = B @ A.T
            U, _, Vt = torch.linalg.svd(M)

            self.O[(u, v)] = U @ Vt

    # =====================================================
    # Z update (GROUP SPARSITY on S^T)
    # =====================================================
    def update_Z(self, S, V):
        return hard_group_topk(S + V, self.k)

    # =====================================================
    # P update
    # =====================================================
    def update_P(self, D):
        return normalize_columns(D + self.U)

    # =====================================================
    # Dual updates (FIXED TRANSPOSE)
    # =====================================================
    def update_duals(self, D, P, S_dict, Z_dict, V_dict):

        self.U = self.U + D - P

        for i in S_dict:
            V_dict[i] = V_dict[i] + S_dict[i] - Z_dict[i]

        return V_dict

    # =====================================================
    # Step-size schedule
    # =====================================================
    def update_alpha(self, t):
        self.beta = t * self.beta
        self.alpha = (self.alpha * self.gamma) / (1 + self.beta)

    # =====================================================
    # Training loop (torch.no_grad)
    # =====================================================
    def fit(self, X_dict, outer_iters=20):

        X_dict = {i: X.to(self.device) for i, X in X_dict.items()}

        # ---- initialize
        S_dict = {i: torch.zeros(self.K, X.shape[1], device=self.device) for i, X in X_dict.items()}
        S_dict_new = {i: torch.zeros(self.K, X.shape[1], device=self.device) for i, X in X_dict.items()}
        Z_dict = {i: torch.zeros_like(S_dict[i]) for i in X_dict}
        V_dict = {i: torch.zeros_like(S_dict[i]) for i in X_dict}

        losses = []

        for t in range(1, outer_iters):

            with torch.no_grad():

                # ---- S updates
                for i in X_dict:
                    S_dict_new[i] = self.update_S(i, X_dict, S_dict, Z_dict, V_dict)

                # ---- D update
                D_new = self.update_D(X_dict, S_dict, num_fp_iters=self.num_fp_iters, tol=self.tol)

                for i in X_dict:
                    S_dict[i] = S_dict_new[i].detach().clone()
                
                # ---- O updates
                self.update_O(S_dict)

                # ---- Z updates
                for i in X_dict:
                    Z_dict[i] = self.update_Z(S_dict[i], V_dict[i])

                # ---- P update
                P_new = self.update_P(D_new)

                # ---- dual updates
                V_dict = self.update_duals(D_new, P_new, S_dict, Z_dict, V_dict)

                # assign
                self.D = D_new
                self.P = P_new

                # ---- loss
                loss = 0
                for i in X_dict:
                    loss += torch.norm(X_dict[i] - self.D @ S_dict[i], p='fro')**2

                for (u, v) in self.edges:
                    loss += self.mu * torch.norm(
                        self.O[(u, v)] @ self.D @ S_dict[u] - self.D @ S_dict[v], p='fro'
                    )**2

                losses.append(loss.item())

                print(f"iter {t}: loss={loss.item():.6f}")

                # step size update
                self.update_alpha(t)
        self.update_O(S_dict)
        for i in S_dict:
            S_dict[i] = hard_group_topk(S_dict[i], self.k)
        return self.D, S_dict, losses




def compute_sheaf_loss(model, S_dict, X_dict=None, device=None, verbose=True):
    """
    Compute sheaf consistency loss and per-edge errors.

    Args:
        model: trained MultiUserSCADict
        S_dict: dict of sparse codes {i: S_i}
        X_dict: (optional) dict of data, only used for device alignment
        device: torch device
        verbose: print per-edge errors

    Returns:
        sheaf_loss (tensor), edge_errors (dict)
    """

    device = device or model.device
    sheaf_loss = torch.tensor(0.0, device=device)

    edge_errors = {}

    with torch.no_grad():

        for (u, v) in model.edges:

            S_u = S_dict[u].to(device)
            S_v = S_dict[v].to(device)

            O_uv = model.O[(u, v)]

            # Reconstructions
            DSu = model.D @ S_u
            DSv = model.D @ S_v

            # Sheaf residual
            diff = O_uv @ DSu - DSv

            edge_loss = torch.norm(diff, p='fro')**2
            sheaf_loss += edge_loss

            # relative error (%)
            denom = torch.norm(DSv, p='fro')**2 + 1e-8
            percentage_error = (edge_loss / denom) * 100

            edge_errors[(u, v)] = percentage_error.item()

            if verbose:
                print(f"Edge {(u, v)}: error = {percentage_error.item():.4f}%")

    if verbose:
        print(f"Total Sheaf loss: {sheaf_loss.item():.4f}")

    return sheaf_loss, edge_errors