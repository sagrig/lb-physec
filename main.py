import numpy as np

NUM_TRIALS = 10000

# start value for Bob's sigma
SIGMA_BOB_START      = 0.2
SIGMA_START_DIFF     = 0.1
SIGMA_ACCUM          = 0.05
SIGMA_ACCUM_NUM      = 3
SIGMA_BOB_START_MAX  = 0.3

rng = np.random.default_rng()

# Coset definitions for Z^2/2Z^2
cosets = {
    (0, 0): np.array([0, 0]),
    (0, 1): np.array([0, 1]),
    (1, 0): np.array([1, 0]),
    (1, 1): np.array([1, 1]),
}

def rnd_msg() -> tuple[int, int]:
    return tuple(rng.integers(0, 2, size=2))

def encode(msg, k_range=2) -> np.ndarray:
    c = cosets[msg]
    k = rng.integers(-k_range, k_range + 1, size=2)
    x = c + 2 * k
    return x.astype(float)

def awgn_channel(x, sigma) -> np.ndarray:
    noise = rng.normal(0.0, sigma, size=x.shape)
    return x + noise

def nearest_z2_point(y) -> np.ndarray:
    return np.rint(y).astype(int)

def coset_decode(y) -> tuple[tuple[int, int], np.ndarray]:
    z_hat   = nearest_z2_point(y)
    msg_hat = tuple((z_hat % 2).astype(int))
    return msg_hat, z_hat

def __simulate(num_trials, sigma_bob, sigma_eve) -> tuple[float, float]:
    bob_correct = 0
    eve_correct = 0

    for _ in range(num_trials):
        msg = rnd_msg()
        x   = encode(msg)

        y_bob = awgn_channel(x, sigma_bob)
        y_eve = awgn_channel(x, sigma_eve)

        bob_msg_hat, _ = coset_decode(y_bob)
        eve_msg_hat, _ = coset_decode(y_eve)

        if bob_msg_hat == msg:
            bob_correct += 1
        if eve_msg_hat == msg:
            eve_correct += 1

    bob_rate = bob_correct / num_trials
    eve_rate = eve_correct / num_trials

    return bob_rate, eve_rate

def simulate(
        num_trials,
        bob_start,
        start_diff,
        accum,
        accum_num,
        bob_start_max
):
    while bob_start < bob_start_max:
        eve_start      = bob_start + start_diff
        eve_start_max  = eve_start + (float(accum_num) * accum)

        while eve_start < eve_start_max:
            bob_rate, eve_rate = __simulate(num_trials, bob_start, eve_start)
            print(f"-- REPORT BOB SIGMA {bob_start:.2f}; EVE SIGMA: {eve_start:.2f}")
            print(f"Bob rate:  {bob_rate:.2f}")
            print(f"Eve rate:  {eve_rate:.2f}\n")
            eve_start += accum
        bob_start     += accum
    return 0, 0

if __name__ == "__main__":
    bob_rate, eve_rate = simulate(
        NUM_TRIALS,
        SIGMA_BOB_START,
        SIGMA_START_DIFF,
        SIGMA_ACCUM,
        SIGMA_ACCUM_NUM,
        SIGMA_BOB_START_MAX)
    exit(0)
