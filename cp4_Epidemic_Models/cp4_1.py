import numpy as np
import matplotlib.pyplot as plt

def sir_rhs(y, beta, gamma):
    S, I, R = y
    dS = -beta * S * I
    dI =  beta * S * I - gamma * I
    dR =  gamma * I
    return np.array([dS, dI, dR])

def rk4_step(f, y, dt, *args):
    k1 = f(y,             *args)
    k2 = f(y + dt/2*k1,   *args)
    k3 = f(y + dt/2*k2,   *args)
    k4 = f(y + dt*k3,     *args)
    return y + dt*(k1 + 2*k2 + 2*k3 + k4)/6

if __name__ == "__main__":
    # 參數
    beta, gamma = 0.4, 0.05
    I0 = 1/5000
    R0 = 0.0
    S0 = 1 - I0 - R0
    y = np.array([S0, I0, R0])

    t_max = 200.0
    dt = 0.1
    steps = int(t_max/dt) + 1

    # 儲存陣列
    t_arr = np.linspace(0, t_max, steps)
    S = np.zeros(steps)
    I = np.zeros(steps)
    R = np.zeros(steps)

    # 初值
    S[0], I[0], R[0] = y

    # RK4 主迴圈
    for n in range(1, steps):
        y = rk4_step(sir_rhs, y, dt, beta, gamma)
        S[n], I[n], R[n] = y

    # 繪圖
    plt.figure(figsize=(8,5))
    plt.plot(t_arr, S, label="S(t)")
    plt.plot(t_arr, I, label="I(t)")
    plt.plot(t_arr, R, label="R(t)")
    plt.xlabel("Time $t$")
    plt.ylabel("Fraction")
    plt.title("SIR Model (β=0.4, γ=0.05) by RK4")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
