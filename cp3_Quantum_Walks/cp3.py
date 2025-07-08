import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- 1. 建立 Liouville 矩陣 M for a reflecting chain of length N ---
def build_liouville_chain(N):
    M = np.zeros((N, N), dtype=float)
    for i in range(N):
        # 如果有左邊鄰居，就加上一個 -1
        if i > 0:
            M[i, i-1] = -1.0
        # 如果有右邊鄰居，就加上一個 -1
        if i < N-1:
            M[i, i+1] = -1.0
        # 對角元素 = 連出去的邊數
        M[i, i] = -M[i].sum()
    return M

# --- 2. 古典隨機漫步：解 dP/dt = -M P ---
def solve_classical(M, P0, t_eval):
    def rhs(t, P):
        return -M.dot(P)
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), P0,
                    t_eval=t_eval, method="RK45",
                    atol=1e-9, rtol=1e-6)
    return sol.y  # shape = (N, len(t_eval))

# --- 3. 量子漫步：ψ(t) = exp(-i M t) ψ(0))
#    為了快速，採用 M 的正交對角化 M = V diag(w) Vᵀ ---
def solve_quantum(M, psi0, t_eval):
    # 對角化
    w, V = np.linalg.eigh(M)    # w: eigenvalues, V: orthonormal eigenvectors
    # 初始在第 1 個基底 e1 的投影係數 c_k = ⟨v_k | e1⟩ = V[0,k]
    c = V[0, :].copy()  
    # 對每個時間點算 exp(-i w t)
    exp_wt = np.exp(-1j * np.outer(w, t_eval))  # shape = (N, len(t_eval))
    # 在 eigenbasis 上的係數
    coeffs = c[:, None] * exp_wt                # (N, T)
    # 轉回原始 basis
    psi_t = V.dot(coeffs)                       # (N, T)
    return psi_t

# --- 主程式 ---
if __name__ == "__main__":
    N = 32
    M = build_liouville_chain(N)

    # 起始分布：全量質點都在節點 1（python index 0）
    P0     = np.zeros(N);    P0[0]   = 1.0
    psi0   = np.zeros(N, dtype=complex); psi0[0] = 1.0

    # 時間點：0 ~ 1000 等距取 501 點
    t_eval = np.linspace(0, 1000, 501)

    # --- (a) 古典解 ---
    P_t = solve_classical(M, P0, t_eval)  # shape (N, T)
    # 計算 ⟨x^2⟩，假設節點 n 的座標是 x = n−1
    positions = np.arange(N)              # [0,1,2,...,N-1]
    msd_classical = (P_t * positions[:, None]**2).sum(axis=0)

    # --- (b) 量子解 ---
    psi_t = solve_quantum(M, psi0, t_eval)  # shape (N, T)
    prob_t = np.abs(psi_t)**2              # 每個節點的機率
    msd_quantum = (prob_t * positions[:, None]**2).sum(axis=0)

    # --- 繪圖比較 ---
    plt.figure(figsize=(7,5))
    plt.plot(t_eval, msd_classical, label="Classical ⟨x²⟩", lw=2)
    plt.plot(t_eval, msd_quantum,   label="Quantum ⟨x²⟩",   lw=2)
    plt.xlabel("Time $t$")
    plt.ylabel(r"Mean squared displacement $\langle x^2\rangle$")
    plt.title("Classical vs Quantum Continuous‐time Walk on N=32 Chain")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
