import numpy as np
import matplotlib.pyplot as plt

# Pauli matrices
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)

gamma = 0.5
theta = 0.3*np.pi

# 初始態 psi0
psi0 = np.array([np.cos(theta/2), np.sin(theta/2)], dtype=complex)

# H 與演化算符
H = gamma * Z

def U(t):
    # H 已對角化 => U = exp(-i*gamma*t * Z) = diag(e^{-iγt}, e^{+iγt})
    return np.diag([np.exp(-1j*gamma*t), np.exp(+1j*gamma*t)])

# 要記錄的時間
t_list = np.linspace(0, 20, 501)

# 儲存 expectation values
expX = np.zeros_like(t_list, dtype=float)
expY = np.zeros_like(t_list, dtype=float)
expZ = np.zeros_like(t_list, dtype=float)

for idx, t in enumerate(t_list):
    psi_t = U(t) @ psi0
    expX[idx] = np.real(psi_t.conj() @ (X @ psi_t))
    expY[idx] = np.real(psi_t.conj() @ (Y @ psi_t))
    expZ[idx] = np.real(psi_t.conj() @ (Z @ psi_t))

# 繪圖
plt.figure(figsize=(7,4))
plt.plot(t_list, expX, label=r'$\langle X\rangle$')
plt.plot(t_list, expY, label=r'$\langle Y\rangle$')
plt.plot(t_list, expZ, label=r'$\langle Z\rangle$')
plt.xlabel('Time $t$')
plt.ylabel('Expectation value')
plt.title('Spin‐1/2 Precession in a $z$‐field')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
