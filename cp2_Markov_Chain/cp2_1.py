import numpy as np

# 1. 定義 transition matrix P
#    P[i,j] = P(state i -> state j)
#    行 i 代表「從 c_{i+1} 出發」，列 j 代表「到 c_{j+1}」。
P = np.array([
    # c1      c2      c3      c4      c5
    [0.16,   0.42,   0.06,   0.29,   0.07],    # from c1
    [0.23,   0.05,   0.41,   0.19,   0.12],    # from c2
    [0.06,   0.41,   0.06,   0.26,   0.21],    # from c3
    [0.31,   0.13,   0.06,   0.31,   0.19],    # from c4
    [0.03,   0.16,   0.26,   0.16,   0.39],    # from c5
])


# 2. Power method 迭代
def stationary_via_power_method(P, tol=1e-12, max_iter=10000):
    n = P.shape[0]
    # 初始猜測：均勻分布
    pi = np.ones(n) / n

    for it in range(max_iter):
        pi_next = pi @ P       # 下一個分布
        pi_next /= pi_next.sum()   # renormalize

        if np.linalg.norm(pi_next - pi, 1) < tol:
            print(f"Converged after {it+1} iterations.")
            return pi_next

        pi = pi_next

    raise RuntimeError("Power method did not converge")

if __name__ == "__main__":
    pi = stationary_via_power_method(P)
    # 列印出來，並檢查 sum(pi)=1
    print("Stationary distribution π (c1..c5):")
    for i, p in enumerate(pi, 1):
        print(f"  π[c{i}] = {p:.6f}")
    print("sum =", pi.sum())
