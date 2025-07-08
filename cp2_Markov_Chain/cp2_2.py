import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def build_liouville_matrix():
    # 節點順序 [c1, c2, c3, c4, c5]，
    # 無向邊速率 γ=1 → A[i,j]=1 if 有邊 else 0
    A = np.array([
        [0,1,1,1,0],    # c1
        [1,0,1,1,1],    # c2
        [1,1,0,1,1],    # c3
        [1,1,1,0,1],    # c4
        [0,1,1,1,0],    # c5
    ], dtype=float)
    D = np.diag(A.sum(axis=1))
    return D - A      # Liouville 矩陣 M = D - A

def rhs(t, P, M):
    return -M.dot(P)  # dP/dt = -M P

if __name__ == "__main__":
    M = build_liouville_matrix()
    P0 = np.zeros(5);  P0[0] = 1.0   # 一開始都在 c1

    # --- 取點策略 ---
    # t∈[0,1]：1001 個點（能清楚看 0~1 的過度）
    # t∈[1,1000]：再取 300 個點
    t1 = np.linspace(0, 1, 1001)
    t2 = np.linspace(1, 1000, 300)
    t_eval = np.concatenate((t1, t2[1:]))  # 避免重複 t=1

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, M),
        t_span=(0, 1000),
        y0=P0,
        t_eval=t_eval,
        method="RK45",
        atol=1e-10,
        rtol=1e-8
    )

    plt.figure(figsize=(8,5))
    for i in range(5):
        plt.plot(sol.t, sol.y[i],
                 label=f'$P_{{{i+1}}}(t)$')
    plt.axhline(0.2, color='k', ls='--', lw=1, label='Uniform $1/5$')
    plt.xlim(0, 1)   # 先 zoom 到 0~1 秒之內
    plt.xlabel('Time $t$')
    plt.ylabel('Probability $P_n(t)$')
    plt.title('Transient behavior for $t\\in[0,1]$')
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #  0~1000 ：
    plt.figure(figsize=(8,5))
    for i in range(5):
        plt.plot(sol.t, sol.y[i], label=f'$P_{{{i+1}}}(t)$')
    plt.axhline(0.2, color='k', ls='--', lw=1, label='Uniform $1/5$')
    plt.xlim(0, 1000)
    plt.xlabel('Time $t$')
    plt.ylabel('Probability $P_n(t)$')
    plt.title('Full evolution up to $t=10^3$')
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
