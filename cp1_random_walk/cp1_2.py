import numpy as np
import matplotlib.pyplot as plt

def simulate_first_passage(N, T_max):
    """
    Simulate N independent 1D symmetric random walks on a lattice.
    Record the first-return time to the origin for each walk.
    Return F[t] = probability of first return at step t, for t=0..T_max.
    """
    F = np.zeros(T_max+1)
    for i in range(N):
        position = 0
        for t in range(1, T_max+1):
            step = np.random.choice([-1, 1])
            position += step
            if position == 0:
                F[t] += 1
                break
        # if it never returns within T_max steps, we ignore it
    return F / N

if __name__ == "__main__":
    # simulation parameters
    N = 200_000    # number of independent walks
    T_max = 2000   # maximum number of steps per walk

    # run simulation
    F = simulate_first_passage(N, T_max)
    t = np.arange(T_max+1)

    # perform log–log fit on the nonzero part
    mask = (t > 0) & (F > 0)
    log_t = np.log(t[mask])
    log_F = np.log(F[mask])
    alpha, intercept = np.polyfit(log_t, log_F, 1)
    print(f"Fitted first-passage exponent α ≈ {alpha:.3f}")

    # plot
    plt.figure(figsize=(6,4))
    plt.loglog(t[mask], F[mask], '.', label='Simulation data')
    plt.loglog(t[mask], np.exp(intercept)*t[mask]**alpha, '-',
               label=f'$\\propto t^{{{alpha:.2f}}}$')
    plt.xlabel("Time steps $t$")
    plt.ylabel("First‐passage probability $F(t)$")
    plt.title("1D Random Walk First‐Passage Probability")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()
