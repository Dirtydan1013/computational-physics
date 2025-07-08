import numpy as np
import matplotlib.pyplot as plt

def simulate_2d_random_walk(M, T):
    """
    模擬 M 個行走者，在方格點上做 T 步 2D 隨機漫步。
    回傳各時間點的平均平方位移 <r^2>(t)，shape=(T+1,)
    """
    # 所有行走者初始都在原點 (0,0)
    pos = np.zeros((M, 2), dtype=int)
    r2 = np.zeros(T+1)
    r2[0] = 0.0

    for t in range(1, T+1):
        # 隨機選 0,1,2,3 分別對應上、下、左、右
        steps = np.random.randint(0, 4, size=M)
        dx = (steps == 3).astype(int) - (steps == 2).astype(int)
        dy = (steps == 0).astype(int) - (steps == 1).astype(int)
        pos[:, 0] += dx
        pos[:, 1] += dy
        # 計算所有行走者的 r^2 = x^2 + y^2，並取平均
        r2[t] = np.mean(pos[:, 0]**2 + pos[:, 1]**2)

    return r2

if __name__ == "__main__":
    M = 20000    # 行走者數量
    T = 1000     # 步數
    r2 = simulate_2d_random_walk(M, T)
    t = np.arange(T+1)

    plt.figure(figsize=(6,4))
    plt.plot(t, r2, lw=2)
    plt.xlabel("step $t$")
    plt.ylabel(r"$\langle r^2\rangle$")
    plt.title("2D mean squared net displacement of random walk")
    plt.grid(True)
    plt.show()
