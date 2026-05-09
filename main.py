import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
matplotlib.use("Agg")

SIGMA_BOB=0.4
SIGMA_EVE=0.5
SIGMA_STEP=0.05
SIGMA_MAX_STEPS=20
SIGM_MAXEVE=SIGMA_EVE + SIGMA_STEP * SIGMA_MAX_STEPS
PLOT_DIR="plots"
EVE_COLOR="#0f766e"
BOB_COLOR="#e76f51"
FIG_COLOR="#f8fafc"
AX_COLOR="#ffffff"
GRID_COLOR="#cbd5e1"

K=[2, 4]
N=10000
RNG=np.random.default_rng()

# --- Z/kZ functionality ---
def sample_msg(ZkZ_msg_set):    
    rand_idx = np.random.randint(len(ZkZ_msg_set))
    return ZkZ_msg_set[rand_idx]

def sample_randvec(tuple_size):
    # limit here is a constant, although maybe need to be random?    
    limit    = 100
    rand_int = np.random.randint(-limit, limit, size=tuple_size)
    return rand_int

def sample_noise(x, sigma):
    noise = RNG.normal(0.0, sigma, size=x.shape)
    return noise

def bob_zkz_decode(yB, k, ZkZ_msg_set):


    best_m = None
    best_d = float("inf")

    for m in ZkZ_msg_set:
        m = np.array(m)
        x_hat = m + k * np.rint((yB - m) / k)
        d = np.linalg.norm(yB - x_hat)
        if d < best_d:
            best_d = d
            best_m = m

    return best_m

# Eve's decode algorithm:
# 1) yE_norm = obtain norm from yE noisy vector
# 2) for every possible message in Z/kZ, perform the same normalisation
# 3) find the closest normalised Z/kZ coset to yE_norm
# 4) best_m = message associated with the corresponding closest coset is the one to return
def eve_zkz_decode(yE, k, ZkZ_msg_set):
    yE_norm = yE / k

    best_m = None
    best_d = float("inf")

    for m in ZkZ_msg_set:
        shift = np.array(m) / k
        nearest_int = np.rint(yE_norm - shift)
        d = np.linalg.norm((yE_norm - shift) - nearest_int)

        if d < best_d:
            best_d = d
            best_m = m

    return best_m

# --- E8 functionality ---
HAMMING_G = np.array([
    [1, 0, 0, 0, 0, 1, 1, 1],
    [0, 1, 0, 0, 1, 0, 1, 1],
    [0, 0, 1, 0, 1, 1, 0, 1],
    [0, 0, 0, 1, 1, 1, 1, 0],
], dtype=int)

def all_hamming_codewords() -> np.ndarray:
    codewords = []
    for i in range(16):
        bits = np.array([(i >> j) & 1 for j in range(4)], dtype=int)
        cw = (bits @ HAMMING_G) % 2
        codewords.append(cw)
    return np.array(codewords, dtype=int)

ALL_E8_MSG_SET=all_hamming_codewords()
del all_hamming_codewords

def sample_e8_msg(E8_msg_set):
    idx = np.random.randint(len(E8_msg_set))
    return E8_msg_set[idx]

def bob_e8_decode(yB, k, E8_msg_set):
    best_m = None
    best_d = float("inf")

    for m in E8_msg_set:
        m = np.array(m)
        x_hat = m + k * np.rint((yB - m) / k)
        d = np.linalg.norm(yB - x_hat)
        if d < best_d:
            best_d = d
            best_m = m
    return best_m

def eve_e8_decode(yE, k, E8_msg_set):
    yE_norm = yE / k

    best_m = None
    best_d = float("inf")

    for m in E8_msg_set:
        shift = np.array(m) / k
        nearest_int = np.rint(yE_norm - shift)
        d = np.linalg.norm((yE_norm - shift) - nearest_int)

        if d < best_d:
            best_d = d
            best_m = m
    return best_m


def simulate(
        title,
        k,
        bob_msg_set,
        eve_msg_set,
        sample_msg_fn,
        bob_decode_fn,
        eve_decode_fn
):
    b_total = 0
    e_rates = []
    for step in range(SIGMA_MAX_STEPS):
        sigma_eve = SIGMA_EVE + (step * SIGMA_STEP)
        b_corr = 0
        e_corr = 0

        for i in range(N):
            m = sample_msg_fn(bob_msg_set)
            z = sample_randvec(len(m))
            x = m + (k * z)

            eB = sample_noise(x, SIGMA_BOB)
            eE = sample_noise(x, sigma_eve)

            yB = x + eB
            yE = x + eE

            mB = bob_decode_fn(yB, k, bob_msg_set)
            mE = eve_decode_fn(yE, k, eve_msg_set)

            b_corr = b_corr + np.array_equal(m, mB)
            e_corr = e_corr + np.array_equal(m, mE)

        b_rate   = (b_corr / N) * 100
        e_rate   = (e_corr / N) * 100
        b_total  = b_total + b_rate
        e_rates.append(e_rate)

        print(f"-- Monte Carlo simulation results for {title} --")
        print(f"-- Eve Sigma {sigma_eve:.2f} --")
        print(f"Bob success rate       {b_corr}/{N} ({b_rate:.2f}%)")
        print(f"Eve success rate       {e_corr}/{N} ({e_rate:.2f}%)")
        print()
        print('===================================================================\n')
    b_avg = b_total / SIGMA_MAX_STEPS
    return b_avg, e_rates

# Variable Legend
# m  - randomly sampled message from available Z/kZ set
# x  - the encoded msg by Alice
# eB - Bob's Gaussian noise vector
# eE - Eve's Gaussian noise vector
# yB - Bob's received message with noise
# yE - Eve's received message with noise
# mB - Bob's decoded message
# mE - Eve's decoded message
if __name__ == "__main__":    
    os.makedirs(PLOT_DIR, exist_ok=True)
    sigma_values = [round(SIGMA_EVE + (step * SIGMA_STEP), 2) for step in range(SIGMA_MAX_STEPS)]
    simulation_results = []

    #### Z/2Z implementation ####
    Z2Z_msg_set = [(a, b) for a in range(2) for b in range(2)]
    b_avg, e_rates = simulate(
        title         = "Z/2Z",
        k             = 2,
        bob_msg_set   = Z2Z_msg_set,
        eve_msg_set   = Z2Z_msg_set,
        sample_msg_fn = sample_msg,
        bob_decode_fn = bob_zkz_decode,
        eve_decode_fn = eve_zkz_decode
    )
    simulation_results.append((r"$\mathbb{Z}^{2}/2\mathbb{Z}^{2}$", "z2z.png", b_avg, e_rates))

    #### Z/4Z implementation ####
    Z4Z_msg_set = [(a, b) for a in range(4) for b in range(4)]
    b_avg, e_rates = simulate(
        title         = "Z/4Z",
        k             = 4,
        bob_msg_set   = Z4Z_msg_set,
        eve_msg_set   = Z4Z_msg_set,
        sample_msg_fn = sample_msg,
        bob_decode_fn = bob_zkz_decode,
        eve_decode_fn = eve_zkz_decode
    )
    simulation_results.append((r"$\mathbb{Z}^{2}/4\mathbb{Z}^{2}$", "z4z.png", b_avg, e_rates))

    #### repetitive Z/2Z implementation ####
    rZ2Z_msg_set = [(a, a) for a in range(2)]
    b_avg, e_rates = simulate(
        title         = "repetitive Z/2Z",
        k             = 2,
        bob_msg_set   = rZ2Z_msg_set,
        eve_msg_set   = rZ2Z_msg_set,
        sample_msg_fn = sample_msg,
        bob_decode_fn = bob_zkz_decode,
        eve_decode_fn = eve_zkz_decode
    )
    simulation_results.append((r"Repetition code in $\mathbb{Z}^{2}/2\mathbb{Z}^{2}$", "repetitive_z2z.png", b_avg, e_rates))

    #### repetitive Z/4Z implementation ####
    rZ4Z_msg_set = [(a, a) for a in range(4)]
    b_avg, e_rates = simulate(
        title         = "repetitive Z/4Z",
        k             = 4,
        bob_msg_set   = rZ4Z_msg_set,
        eve_msg_set   = rZ4Z_msg_set,
        sample_msg_fn = sample_msg,
        bob_decode_fn = bob_zkz_decode,
        eve_decode_fn = eve_zkz_decode
    )
    simulation_results.append((r"Repetition code in $\mathbb{Z}^{2}/4\mathbb{Z}^{2}$", "repetitive_z4z.png", b_avg, e_rates))
    #### E8 implementation ####
    b_avg, e_rates = simulate(
        title         = "E8",
        k             = 2,
        bob_msg_set   = ALL_E8_MSG_SET,
        eve_msg_set   = ALL_E8_MSG_SET,
        sample_msg_fn = sample_e8_msg,
        bob_decode_fn = bob_e8_decode,
        eve_decode_fn = eve_e8_decode
    )
    simulation_results.append((r"$\sqrt{2}E_8/2\mathbb{Z}^{8}$", "e8.png", b_avg, e_rates))

    #### repetitive E8 implementation ####
    REPET_E8_MSG_SET = [
        [0] * 8,
        [1] * 8,
    ]
    
    b_avg, e_rates = simulate(
        title         = "repetitive E8",
        k             = 2,
        bob_msg_set   = REPET_E8_MSG_SET,
        eve_msg_set   = REPET_E8_MSG_SET,
        sample_msg_fn = sample_e8_msg,
        bob_decode_fn = bob_e8_decode,
        eve_decode_fn = eve_e8_decode
    )
    simulation_results.append((r"Repetition code in $\sqrt{2}E_8/2\mathbb{Z}^{8}$", "repetitive_e8.png", b_avg, e_rates))

    #### doubly-even E8 implementation ####
    DEVEN_E8_MSG_SET=[
        [0] * 8,
        [1] * 8,
        [1] * 4 + [0] * 4,
        [0] * 4 + [1] * 4
    ]

    b_avg, e_rates = simulate(
        title         = "doubly-even E8",
        k             = 2,
        bob_msg_set   = DEVEN_E8_MSG_SET,
        eve_msg_set   = DEVEN_E8_MSG_SET,
        sample_msg_fn = sample_e8_msg,
        bob_decode_fn = bob_e8_decode,
        eve_decode_fn = eve_e8_decode
    )
    simulation_results.append((r"Doubly-even code in $\sqrt{2}E_8/2\mathbb{Z}^{8}$", "doubly_even_e8.png", b_avg, e_rates))

    #### Bob is E8, Eve is repetitive E8
    b_avg, e_rates = simulate(
        title         = "Bob is E8 and Eve is repetitive E8",
        k             = 2,
        bob_msg_set   = ALL_E8_MSG_SET,
        eve_msg_set   = REPET_E8_MSG_SET,
        sample_msg_fn = sample_e8_msg,
        bob_decode_fn = bob_e8_decode,
        eve_decode_fn = eve_e8_decode
    )
    simulation_results.append((r"Bob: $\sqrt{2}E_8/2\mathbb{Z}^{8}$; Eve: repetition code", "bob_e8_eve_repetitive_e8.png", b_avg, e_rates))

    for title, filename, b_avg, e_rates in simulation_results:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor=FIG_COLOR)
        ax.set_facecolor(AX_COLOR)
        ax.plot(
            sigma_values,
            e_rates,
            color=EVE_COLOR,
            marker="o",
            markerfacecolor=AX_COLOR,
            markeredgecolor=EVE_COLOR,
            markeredgewidth=2,
            linewidth=2.5,
            label="Eve success rate"
        )
        ax.plot(
            sigma_values,
            [b_avg] * len(sigma_values),
            color=BOB_COLOR,
            linestyle="--",
            linewidth=2.5,
            label=f"Bob average ({b_avg:.2f}%)"
        )
        ax.set_title(title)
        ax.set_xlabel("Eve sigma")
        ax.set_ylabel("Success rate (%)")
        ax.set_ylim(0, 100)
        ax.set_xticks(sigma_values)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, color=GRID_COLOR, alpha=0.7)
        ax.legend(frameon=True, facecolor=AX_COLOR, edgecolor=GRID_COLOR)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150, facecolor=FIG_COLOR)
        plt.close()

    exit(0)
