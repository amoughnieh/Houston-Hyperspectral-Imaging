from used_packages import *
from utils import tensor, set_seed

def RPCA_image_decomp(image, lam=0.0022, rho=1.5, tol=1e-4, max_iter=None, verbose=False, seed=42):

    '''
    This algorithm is based on Algorithm 1 from the paper
    "Emmanuel J Candès, Xiaodong Li, Yi Ma, and John Wright - Robust principal component analysis - JACM, 2011,
    which is based on the original work of Lin et al. - Fast convex optimization algorithms for exact
recovery of a corrupted low-rank matrix - 2009.
    '''
    set_seed(seed)
    M = tensor(image)
    Y = torch.zeros_like(M)
    L = torch.zeros_like(Y)
    S = torch.zeros_like(L)
    m, n = M.shape

    lam = lam
    norm2 = torch.linalg.vector_norm(M)
    mu = 1.25 / norm2
    mu_bar = mu * 1e7
    rho = rho
    d_norm = torch.linalg.vector_norm(M)
    iter = 0
    total_svd = 0
    converged = False
    tol = tol
    stopCriterion = 1
    t = []

    start_time = time.perf_counter()
    while not converged:
        iter += 1
        S0 = M - L + Y / mu
        S = torch.sign(S0) * torch.maximum(torch.abs(S0) - lam / mu, torch.tensor(0.0, device=S0.device))
        L0 = M - S + Y / mu
        u, s, vh = torch.linalg.svd(L0, full_matrices=False)
        svp = (s > 1 / mu).sum()
        t.append(svp)
        L = (u[:, :svp] * (s[:svp] - 1 / mu)) @ vh[:svp]
        total_svd += 1
        Z = M - L - S
        Y = Y + mu * Z
        mu = min(mu * rho, mu_bar)

        stopCriterion = torch.linalg.norm(Z) / d_norm

        if verbose:
            s_cpu = np.array(s.cpu()).reshape(-1, 1)
            s_scaled = s_cpu / s_cpu.sum()
            s_cumsum = s_scaled.cumsum()[:svp]
            print(f'Epoch {iter}, Top {t[-1].item()} Principle Component(s) - {s_cumsum[-1] * 100:.2f}% explained variance - error = {stopCriterion :.4e}')

        if stopCriterion < tol or iter == max_iter:
            converged = True

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    return L, S

#%%
