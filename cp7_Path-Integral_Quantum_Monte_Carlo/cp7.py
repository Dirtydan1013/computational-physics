import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# ---------------------------
# 參數設定
beta     = 4.0    # inverse temperature
N        = 100    # 虛時間切片數
eps      = beta / N
n_sweeps = 50000  # Monte Carlo sweep 數量
delta    = 1.0    # 提案步長 scale
# ---------------------------

# 初始路徑：全切片都設為 0
x = np.zeros(N)

# Action：S = sum_k [ (x_{k+1}-x_k)^2/(2 eps) + eps*(x_k^2)/2 ]
def local_action(xm, xk, xp):
    """切片 k 的局部 action"""
    # kinetic: ( (x_k - x_{k-1})^2 + (x_{k+1}-x_k)^2 )/(2 eps)
    # potential: eps * 0.5 * x_k^2
    return ( (xk-xm)**2 + (xp-xk)**2 )/(2*eps) + 0.5*eps*(xk**2)

# 蒙地卡羅 sweep
for sweep in trange(n_sweeps):
    for k in range(N):
        # 隨機挑一個切片 k 提案 xk -> xk'
        xk_old = x[k]
        xm = x[(k-1)%N]
        xp = x[(k+1)%N]
        S_old = local_action(xm, xk_old, xp)

        xk_new = xk_old + np.random.uniform(-delta,delta)
        S_new = local_action(xm, xk_new, xp)

        # Metropolis 接受準則
        if np.random.rand() < np.exp(-(S_new - S_old)):
            x[k] = xk_new

# 取樣：在路徑上再跑一段，蒐集 x[0] 和 x[N//2]
n_samples = 20000
samples   = np.zeros((n_samples, 2))

for i in trange(n_samples):
    # 做一次 sweep
    for k in range(N):
        xk_old = x[k]
        xm = x[(k-1)%N]; xp = x[(k+1)%N]
        S_old = local_action(xm, xk_old, xp)
        xk_new = xk_old + np.random.uniform(-delta,delta)
        S_new = local_action(xm, xk_new, xp)
        if np.random.rand() < np.exp(-(S_new - S_old)):
            x[k] = xk_new
    # 蒐集樣本
    samples[i,0] = x[0]
    samples[i,1] = x[N//2]

# ---------------------------
# 畫直方圖並疊上解析解
xs = np.linspace(-3, 3, 500)
sigma2 = 1.0 / np.tanh(beta/2)
norm   = np.sqrt(1/(2*np.pi*sigma2))
f_analytic = norm * np.exp(- xs**2 / (2*sigma2))

fig, axs = plt.subplots(1,2, figsize=(10,4))
for idx, ax in enumerate(axs):
    ax.hist(samples[:,idx], bins=60, density=True,
            alpha=0.6, label=f'sample $x_{{{ [0,N//2][idx] }}}$')
    ax.plot(xs, f_analytic, 'k--', label='analytic')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$P(x)$')
    ax.legend()
    ax.set_title(f'Slice {["0","N/2"][idx]}')

plt.tight_layout()
plt.show()
