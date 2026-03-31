import numpy as np

NUM_TRIALS = 10000

# start value for Bob's sigma
SIGMA_BOB_START      = 0.2
SIGMA_START_DIFF     = 0.1
SIGMA_ACCUM          = 0.05
SIGMA_ACCUM_NUM      = 3
SIGMA_BOB_START_MAX  = 0.3

Z2_K_RANGE = 2
E8_K_RANGE = 2

rng = np.random.default_rng()

# Coset definitions for Z^2 / 2Z^2
cosets = {
    (0, 0): np.array([0, 0]),
    (0, 1): np.array([0, 1]),
    (1, 0): np.array([1, 0]),
    (1, 1): np.array([1, 1]),
}

E8_BASIS = np.array([
    [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0,  1.0,-1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0,  0.0, 1.0,-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0,  0.0, 0.0, 1.0,-1.0, 0.0, 0.0],
    [0.0, 0.0,  0.0, 0.0, 0.0, 1.0,-1.0, 0.0],
    [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 1.0,-1.0],
    [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    [0.5, 0.5,  0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
], dtype=float)

E8_BASIS_INV = np.linalg.inv(E8_BASIS)

def rnd_msg(lattice="z2"):
    if lattice == "e8":
        return tuple(rng.integers(0, 2, size=8))
    return tuple(rng.integers(0, 2, size=2))

def encode(msg, lattice="z2", k_range=2) -> np.ndarray:
    if lattice == "e8":
        # Cosets of E8 / 2E8:
        m = np.array(msg, dtype=int)
        k = rng.integers(-k_range, k_range + 1, size=8)
        x = (m + 2 * k) @ E8_BASIS
        return x.astype(float)

    c = cosets[msg]
    k = rng.integers(-k_range, k_range + 1, size=2)
    x = c + 2 * k
    return x.astype(float)

def awgn_channel(x, sigma) -> np.ndarray:
    noise = rng.normal(0.0, sigma, size=x.shape)
    return x + noise

def nearest_z2_point(y) -> np.ndarray:
    return np.rint(y).astype(int)

def nearest_d8_point(y) -> np.ndarray:
    z = np.rint(y).astype(int)

    # D8 = integer vectors with even coordinate sum
    if np.sum(z) % 2 != 0:
        idx = np.argmax(np.abs(y - z))
        z[idx] += 1 if y[idx] >= z[idx] else -1

    return z.astype(float)

def nearest_e8_point(y) -> np.ndarray:
    # E8 = D8 union (D8 + (1/2,...,1/2))
    z0 = nearest_d8_point(y)
    z1 = nearest_d8_point(y - 0.5) + 0.5

    if np.sum((y - z0) ** 2) <= np.sum((y - z1) ** 2):
        return z0
    return z1

def coset_decode(y, lattice="z2"):
    if lattice == "e8":
        z_hat     = nearest_e8_point(y)
        coeff_hat = np.rint(z_hat @ E8_BASIS_INV).astype(int)
        msg_hat   = tuple((coeff_hat % 2).astype(int))
        return msg_hat, z_hat

    z_hat   = nearest_z2_point(y)
    msg_hat = tuple((z_hat % 2).astype(int))
    return msg_hat, z_hat

def __simulate(num_trials, sigma_bob, sigma_eve, lattice="z2") -> tuple[float, float]:
    bob_correct = 0
    eve_correct = 0
    k_range     = E8_K_RANGE if lattice == "e8" else Z2_K_RANGE

    for _ in range(num_trials):
        msg = rnd_msg(lattice=lattice)
        x   = encode(msg, lattice=lattice, k_range=k_range)

        y_bob = awgn_channel(x, sigma_bob)
        y_eve = awgn_channel(x, sigma_eve)

        bob_msg_hat, _ = coset_decode(y_bob, lattice=lattice)
        eve_msg_hat, _ = coset_decode(y_eve, lattice=lattice)

        if bob_msg_hat == msg:
            bob_correct += 1
        if eve_msg_hat == msg:
            eve_correct += 1

    bob_rate = bob_correct / num_trials
    eve_rate = eve_correct / num_trials

    return bob_rate, eve_rate

def simulate(
        num_trials,
        bob_sigma,
        start_diff,
        accum,
        accum_num,
        bob_start_max
):
    lattices = ["z2", "e8"]
    while bob_sigma < bob_start_max:
        eve_sigma      = bob_sigma + start_diff
        eve_start_max  = eve_sigma + (float(accum_num) * accum)
            
        while eve_sigma < eve_start_max:
            print(f"--- REPORT BOB SIGMA {bob_sigma:.2f}; EVE SIGMA: {eve_sigma:.2f}")
            for lattice in lattices:
                bob_rate, eve_rate = __simulate(
                    num_trials,
                    bob_sigma,
                    eve_sigma,
                    lattice=lattice
                )
                print(f"-- Lattice {lattice}")
                print(f"Bob rate:  {bob_rate:.2f}")
                print(f"Eve rate:  {eve_rate:.2f}\n")
            eve_sigma += accum
        bob_sigma += accum
        print("-------------------------------------------")


if __name__ == "__main__":
    simulate(
        NUM_TRIALS,
        SIGMA_BOB_START,
        SIGMA_START_DIFF,
        SIGMA_ACCUM,
        SIGMA_ACCUM_NUM,
        SIGMA_BOB_START_MAX
    )
    exit(0)
