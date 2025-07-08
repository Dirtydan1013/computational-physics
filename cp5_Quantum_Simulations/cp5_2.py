import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Pauli
X = np.array([[0,1],[1,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

def kronN(mats):
    """多個矩陣做 Kronecker product"""
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

# --- 建 H1, H2, M_op ---
L = 4
# H1 = sum Z_j Z_{j+1}
H1 = np.zeros((2**L,2**L), dtype=complex)
for j in range(L):
    ops = [I2]*L
    ops[j] = Z
    ops[(j+1)%L] = Z
    H1 += kronN(ops)

# H2 = - sum X_j
H2 = np.zeros_like(H1)
for j in range(L):
    ops = [I2]*L
    ops[j] = X
    H2 -= kronN(ops)

H = H1 + H2

# M_z operator
M_op = np.zeros_like(H)
for j in range(L):
    ops = [I2]*L
    ops[j] = Z
    M_op += kronN(ops)

# 初始態 |0000>
psi0 = np.zeros(2**L, dtype=complex)
psi0[0] = 1.0

# 時間設置
t_max = 10
dt = 0.1
steps = int(t_max/dt)+1
t_list = np.linspace(0, t_max, steps)

# 預先算好局部演化子
U1 = expm(-1j * H1 * dt)
U2 = expm(-1j * H2 * dt)
U_exact_full = expm(-1j * H * dt)  # 也可以一步步疊乘或直接 expm(-i H t) 分 t

# 儲存 Mz
Mz_trot = np.zeros(steps, dtype=float)
Mz_ex  = np.zeros(steps, dtype=float)

# Trotter 演化
psi_t = psi0.copy()
for idx in range(steps):
    t = t_list[idx]
    Mz_trot[idx] = np.real(psi_t.conj() @ (M_op @ psi_t))
    # 下一步
    psi_t = U2 @ (U1 @ psi_t)

# Exact 演化：直接對每個 t 用 expm(-i H t)
for idx, t in enumerate(t_list):
    Ue = expm(-1j * H * t)
    psi_e = Ue @ psi0
    Mz_ex[idx] = np.real(psi_e.conj() @ (M_op @ psi_e))

# 繪圖比較
plt.figure(figsize=(7,4))
plt.plot(t_list, Mz_trot, label='Trotter–Suzuki')
plt.plot(t_list, Mz_ex,  '--', label='Exact')
plt.xlabel('Time $t$')
plt.ylabel(r'$M_z(t)$')
plt.title('Transverse‐field Ising Chain ($L=4$)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
