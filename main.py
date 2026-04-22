import numpy as np

#####################################################################
# PRE-SIMULATION CONSTANTS:
# HAMMING_G – generator matrix of Hamming (8, 4, 4) code
#####################################################################
HAMMING_G = np.array([
    [1, 0, 0, 0, 0, 1, 1, 1],
    [0, 1, 0, 0, 1, 0, 1, 1],
    [0, 0, 1, 0, 1, 1, 0, 1],
    [0, 0, 0, 1, 1, 1, 1, 0],
], dtype=int)

#####################################################################
# HELPER FUNCTIONS FOR CONSTANT DEFINITIONS
# all_hamming_codewords() – return all 2^4 = 16 codewords of the
#                           Hamming (8, 4, 4) code each of length 8
#####################################################################
def all_hamming_codewords() -> np.ndarray:
    codewords = []
    for i in range(16):
        bits = np.array([(i >> j) & 1 for j in range(4)], dtype=int)
        cw = (bits @ HAMMING_G) % 2
        codewords.append(cw)
    return np.array(codewords, dtype=int)

######################################################################
# SIMULATION CONSTANTS:
#
# NUM_TRIALS           – number of Monte Carlo simulation trials
# SIGMA_BOB_START      - initial Bob noise standard deviation
# SIGMA_START_DIFF     - initial offset between Bob and Eve sigmas
# SIGMA_ACCUM          - sigma step used in the simulation loops
# SIGMA_ACCUM_NUM      - number of sigma increments tested for each
#                        Bob sigma
# SIGMA_BOB_START_MAX  - upper bound for Bob sigma in the
#                        outer simulation loop
# K_VALUES             - list of tested k values
# RAND_RANGE           - max. random coefficient used to choose coset
# E8_HAMMING_BASIS     - basis matrix for sqrt(2) E8
# E8_HAMMING_BASIS_INV - inverse of E8_HAMMING_BASIS for recovering
#                        integer coeffs after decode
# HAMMING_CODEWORDS    – all 16 binary codewords of (8, 4, 4) obtained
#                        from all_hamming_codewords()
#####################################################################
NUM_TRIALS           = 10000 
SIGMA_BOB_START      = 0.2
SIGMA_START_DIFF     = 0.1
SIGMA_ACCUM          = 0.10
SIGMA_ACCUM_NUM      = 2
SIGMA_BOB_START_MAX  = 0.4
K_VALUES             = [2, 4, 8]
RAND_RANGE           = 2
E8_HAMMING_BASIS     = np.vstack([
    HAMMING_G,
    2 * np.eye(8, dtype=int)[4:]
]).astype(float)
E8_HAMMING_BASIS_INV = np.linalg.inv(E8_HAMMING_BASIS)
HAMMING_CODEWORDS    = all_hamming_codewords()
SQRT2                = np.sqrt(2.0)
RNG                  = np.random.default_rng()

del HAMMING_G
del all_hamming_codewords

def lattice_dim(lattice="z2"):
    if lattice == "e8":
        return 8
    return 2

def make_scales(k, dim) -> np.ndarray:
    exponents = np.zeros(dim, dtype=int)

    for i in range(k):
        exponents[i % dim] += 1

    return (2 ** exponents).astype(int)

def rnd_msg(lattice="z2", k=2):
    dim    = lattice_dim(lattice)
    scales = make_scales(k, dim)

    msg = [RNG.integers(0, scales[i]) for i in range(dim)]
    return tuple(msg)

def encode(msg, lattice="z2", k=2, rand_range=2) -> np.ndarray:
    dim    = lattice_dim(lattice)
    scales = make_scales(k, dim)

    m = np.array(msg, dtype=int)
    r = RNG.integers(-rand_range, rand_range + 1, size=dim)

    coeff = m + scales * r

    if lattice == "e8":
        x = (coeff @ E8_HAMMING_BASIS) / SQRT2
        return x.astype(float)

    return coeff.astype(float)

def awgn_channel(x, sigma) -> np.ndarray:
    noise = RNG.normal(0.0, sigma, size=x.shape)
    return x + noise

def nearest_z2_point(y) -> np.ndarray:
    return np.rint(y).astype(int)

def nearest_e8_point(y) -> np.ndarray:
    target = SQRT2 * y

    best_u    = None
    best_dist = np.inf

    for c in HAMMING_CODEWORDS:
        z = np.rint((target - c) / 2.0).astype(int)
        u = 2 * z + c

        dist = np.sum((target - u) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_u    = u
    return best_u.astype(float) / SQRT2


def coset_decode(y, lattice="z2", k=2):
    dim    = lattice_dim(lattice)
    scales = make_scales(k, dim)

    if lattice == "e8":
        z_hat = nearest_e8_point(y)
        u_hat = np.rint(SQRT2 * z_hat).astype(int)

        coeff_hat = np.rint(u_hat @ E8_HAMMING_BASIS_INV).astype(int)

        msg_hat = tuple((coeff_hat % scales).astype(int))
        return msg_hat, z_hat

    z_hat   = nearest_z2_point(y)
    msg_hat = tuple((z_hat % scales).astype(int))
    return msg_hat, z_hat


def __simulate(num_trials, sigma_bob, sigma_eve, lattice="z2", k=2, rand_range=2) -> tuple[float, float]:
    bob_correct = 0
    eve_correct = 0

    for _ in range(num_trials):
        msg = rnd_msg(lattice=lattice, k=k)
        x   = encode(msg, lattice=lattice, k=k, rand_range=rand_range)

        y_bob = awgn_channel(x, sigma_bob)
        y_eve = awgn_channel(x, sigma_eve)

        bob_msg_hat, _ = coset_decode(y_bob, lattice=lattice, k=k)
        eve_msg_hat, _ = coset_decode(y_eve, lattice=lattice, k=k)

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
        bob_start_max,
        k_values,
        rand_range
):
    lattices = ["z2", "e8"]

    while bob_sigma < bob_start_max:
        eve_sigma      = bob_sigma + start_diff
        eve_start_max  = eve_sigma + (float(accum_num) * accum)

        while eve_sigma < eve_start_max:
            print(f"--- REPORT BOB SIGMA {bob_sigma:.2f}; EVE SIGMA: {eve_sigma:.2f}")

            for lattice in lattices:
                for k in k_values:
                    bob_rate, eve_rate = __simulate(
                        num_trials,
                        bob_sigma,
                        eve_sigma,
                        lattice=lattice,
                        k=k,
                        rand_range=rand_range
                    )
                    print(f"-- Lattice {lattice}; k: {k}")
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
        SIGMA_BOB_START_MAX,
        K_VALUES,
        RAND_RANGE
    )
    exit(0)
