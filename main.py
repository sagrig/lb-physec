import numpy as np

NUM_TRIALS = 10000

# TODO: try and compare different values of sigma instead of fixed
SIGMA_BOB  = 0.2   # Bob noise
SIGMA_EVE  = 0.4   # Eve noise std

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

def simulate(num_trials, sigma_bob, sigma_eve) -> tuple[float, float]:
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


if __name__ == "__main__":
    bob_rate, eve_rate = simulate(NUM_TRIALS, SIGMA_BOB, SIGMA_EVE)

    print("Basic lattice wiretap simulation")
    print(f"Trials:                          {NUM_TRIALS}")
    print(f"Bob sigma:                       {SIGMA_BOB}")
    print(f"Eve sigma:                       {SIGMA_EVE}")
    print(f"Bob correct coset decoding rate: {bob_rate:.4f}")
    print(f"Eve correct coset decoding rate: {eve_rate:.4f}")
    exit(0)
