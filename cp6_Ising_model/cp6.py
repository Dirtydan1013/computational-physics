import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit

# -------------------------------
# 用 Numba JIT 加速關鍵函式
# -------------------------------
@njit
def metropolis_sweep_numba(spin, beta):
    L = spin.shape[0]
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        s = spin[i, j]
        # 週期邊界四鄰居和
        nb = (spin[(i + 1) % L, j] +
              spin[(i - 1) % L, j] +
              spin[i, (j + 1) % L] +
              spin[i, (j - 1) % L])
        dE = 2 * s * nb
        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spin[i, j] = -s

@njit
def measure_E_M_numba(spin):
    L = spin.shape[0]
    E = 0
    # 只算右和下方向可避免重複
    for i in range(L):
        for j in range(L):
            s = spin[i, j]
            E -= s * spin[(i+1)%L, j]
            E -= s * spin[i, (j+1)%L]
    M = 0
    for i in range(L):
        for j in range(L):
            M += spin[i, j]
    return E, M

# -------------------------------
# 其餘不變
# -------------------------------
def run_ising(L, T, eq_sweeps, meas_sweeps):
    beta = 1.0 / T
    spin = np.random.choice(np.int8([-1, 1]), size=(L, L))
    # 熱平衡
    for _ in range(eq_sweeps):
        metropolis_sweep_numba(spin, beta)
    # 量測
    E_acc = E2_acc = Mabs_acc = M2_acc = M4_acc = 0.0
    for _ in range(meas_sweeps):
        metropolis_sweep_numba(spin, beta)
        E, M = measure_E_M_numba(spin)
        E_acc    += E
        E2_acc   += E * E
        Mabs_acc += abs(M)
        M2_acc   += M * M
        M4_acc   += M**4
    norm = float(meas_sweeps)
    return (E_acc/norm, E2_acc/norm,
            Mabs_acc/norm, M2_acc/norm, M4_acc/norm)

if __name__ == "__main__":
    np.random.seed(0)
    L_list = [4, 8, 16]
    T_list = np.linspace(0.5, 5.0, 40)

    results = {L: {'mag': [], 'cv': [], 'chi': [], 'UL': []}
               for L in L_list}

    # 嵌套 tqdm，分別顯示 L 與 T 的進度
    for L in tqdm(L_list, desc="Lattice L"):
        N = L * L
        eq_sweeps = 200 * L      # 減少 sweep 次數做快速示範
        meas_sweeps = 500 * L

        for T in tqdm(T_list, desc=f"T for L={L}", leave=False):
            E, E2, Mabs, M2, M4 = run_ising(L, T, eq_sweeps, meas_sweeps)
            C   = (E2 - E*E) / (T*T)
            chi = (M2 - Mabs*Mabs) / T
            UL  = 1.0 - (M4 / (3.0 * M2*M2))
            mag = np.sqrt(M2) / N

            r = results[L]
            r['mag'].append(mag)
            r['cv'].append(C/N)
            r['chi'].append(chi/N)
            r['UL'].append(UL)

    # 繪圖同原本
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_mag, ax_cv, ax_chi, ax_UL = axes.ravel()
    for L in L_list:
        ax_mag.plot(T_list, results[L]['mag'], label=f"L={L}")
        ax_cv .plot(T_list, results[L]['cv'],  label=f"L={L}")
        ax_chi.plot(T_list, results[L]['chi'], label=f"L={L}")
        ax_UL .plot(T_list, results[L]['UL'],  label=f"L={L}")

    ax_mag.set_ylabel(r"$\sqrt{\langle M^2\rangle}/N$")
    ax_cv .set_ylabel(r"$C/N$")
    ax_chi.set_ylabel(r"$\chi/N$")
    ax_UL .set_ylabel(r"$U_L$")

    for ax in (ax_mag, ax_cv, ax_chi, ax_UL):
        ax.legend(); ax.set_xlabel("Temperature $T$"); ax.grid(True)

    ax_mag.set_title("Magnetization")
    ax_cv .set_title("Heat capacity per site")
    ax_chi.set_title("Susceptibility per site")
    ax_UL .set_title("Binder cumulant")

    plt.tight_layout()
    plt.show()
