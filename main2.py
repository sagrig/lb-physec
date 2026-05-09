import numpy as np

SIGMA_BOB=0.4
SIGMA_EVE=0.8
K=[2, 3, 4]
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
    yB_hat = np.rint(yB).astype(int)
    yB_mod = np.mod(yB_hat, k)

    best_m = None
    best_d = float("inf")

    for m in ZkZ_msg_set:
        d = np.linalg.norm(yB_mod - np.array(m))
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
    yB_hat = np.rint(yB).astype(int)
    yB_mod = np.mod(yB_hat, k)

    best_m = None
    best_d = float("inf")

    for m in E8_msg_set:
        d = np.sum(yB_mod != np.array(m))
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
    # Z/kZ implementation
    for k in K:        
        ZkZ_msg_set = [(a, b) for a in range(k) for b in range(k)]
        b_corr = 0
        e_corr  = 0

        for i in range(N):
            m  = sample_msg(ZkZ_msg_set)
            z  = sample_randvec(len(m))
            x  = m + (k * z)

            eB = sample_noise(x, SIGMA_BOB)
            eE = sample_noise(x, SIGMA_EVE)

            yB = x + eB
            yE = x + eE

            mB = bob_zkz_decode(yB, k, ZkZ_msg_set)
            mE = eve_zkz_decode(yE, k, ZkZ_msg_set)

            b_corr = b_corr + (m == mB)
            e_corr = e_corr + (m == mE)

        b_rate = (b_corr / N) * 100
        e_rate = (e_corr / N) * 100

        print(f"-- Monte Carlo simulation results for Z/{k}Z: --")
        print(f"Bob success rate       {b_corr}/{N} ({b_rate:.2f}%)")
        print(f"Eve success rate       {e_corr}/{N} ({e_rate:.2f}%)")
        print()
    print('============================================================\n')
    # repetitive Z/kZ implementation
    for k in K:        
        ZkZ_msg_set = [(a, a) for a in range(k)]
        b_corr = 0
        e_corr = 0

        for i in range(N):
            m  = sample_msg(ZkZ_msg_set)
            z  = sample_randvec(len(m))
            x  = m + (k * z)

            eB = sample_noise(x, SIGMA_BOB)
            eE = sample_noise(x, SIGMA_EVE)

            yB = x + eB
            yE = x + eE

            mB = bob_zkz_decode(yB, k, ZkZ_msg_set)
            mE = eve_zkz_decode(yE, k, ZkZ_msg_set)

            b_corr = b_corr + (m == mB)
            e_corr = e_corr + (m == mE)

        b_rate = (b_corr / N) * 100
        e_rate = (e_corr / N) * 100

        print(f"-- Monte Carlo simulation results for repetitive Z/{k}Z: --")
        print(f"Bob success rate       {b_corr}/{N} ({b_rate:.2f}%)")
        print(f"Eve success rate       {e_corr}/{N} ({e_rate:.2f}%)")
        print()
    print('============================================================\n')        
    # E8 implementation
    BOB_E8_MSG_SET=ALL_E8_MSG_SET
    EVE_E8_MSG_SET=ALL_E8_MSG_SET
    
    for k in [2]:
        b_corr = 0
        e_corr = 0

        for i in range(N):
            m = sample_e8_msg(BOB_E8_MSG_SET)
    
            z = sample_randvec(len(m))
            x = m + (k * z)

            eB = sample_noise(x, SIGMA_BOB)
            eE = sample_noise(x, SIGMA_EVE)

            yB = x + eB
            yE = x + eE

            mB = bob_e8_decode(yB, k, BOB_E8_MSG_SET)
            mE = eve_e8_decode(yE, k, EVE_E8_MSG_SET)

            b_corr = b_corr + np.array_equal(m, mB)
            e_corr = e_corr + np.array_equal(m, mE)
            
        b_rate = (b_corr / N) * 100
        e_rate = (e_corr / N) * 100
        print(f"-- Monte Carlo simulation results for E8 (k={k}): --")
        print(f"Bob success rate       {b_corr}/{N} ({b_rate:.2f}%)")
        print(f"Eve success rate       {e_corr}/{N} ({e_rate:.2f}%)")
        print()

    print('============================================================\n')        
    # repetitive E8 implementation
    REPET_E8_MSG_SET = [
        [0] * 8,
        [1] * 8,
    ]
    BOB_E8_MSG_SET = REPET_E8_MSG_SET
    EVE_E8_MSG_SET = REPET_E8_MSG_SET
    
    for k in [2]:
        b_corr = 0
        e_corr = 0

        for i in range(N):
            m = sample_e8_msg(BOB_E8_MSG_SET)
            
            z = sample_randvec(len(m))
            x = m + (k * z)

            eB = sample_noise(x, SIGMA_BOB)
            eE = sample_noise(x, SIGMA_EVE)

            yB = x + eB
            yE = x + eE

            mB = bob_e8_decode(yB, k, BOB_E8_MSG_SET)
            mE = eve_e8_decode(yE, k, EVE_E8_MSG_SET)

            b_corr = b_corr + np.array_equal(m, mB)
            e_corr = e_corr + np.array_equal(m, mE)
            
        b_rate = (b_corr / N) * 100
        e_rate = (e_corr / N) * 100
        print(f"-- Monte Carlo simulation results for repetitive E8 (k={k}): --")
        print(f"Bob success rate       {b_corr}/{N} ({b_rate:.2f}%)")
        print(f"Eve success rate       {e_corr}/{N} ({e_rate:.2f}%)")
        print()
        
    print('============================================================\n')        
    # doubly-even E8 implementation
    DEVEN_E8_MSG_SET=[
        [0] * 8,
        [1] * 8,
        [1] * 4 + [0] * 4,
        [0] * 4 + [1] * 4
    ]
    BOB_E8_MSG_SET = DEVEN_E8_MSG_SET
    EVE_E8_MSG_SET = DEVEN_E8_MSG_SET
    
    for k in [2]:
        b_corr = 0
        e_corr = 0

        for i in range(N):
            m = sample_e8_msg(BOB_E8_MSG_SET)
            
            z = sample_randvec(len(m))
            x = m + (k * z)

            eB = sample_noise(x, SIGMA_BOB)
            eE = sample_noise(x, SIGMA_EVE)

            yB = x + eB
            yE = x + eE

            mB = bob_e8_decode(yB, k, BOB_E8_MSG_SET)
            mE = eve_e8_decode(yE, k, EVE_E8_MSG_SET)

            b_corr = b_corr + np.array_equal(m, mB)
            e_corr = e_corr + np.array_equal(m, mE)
            
        b_rate = (b_corr / N) * 100
        e_rate = (e_corr / N) * 100
        print(f"-- Monte Carlo simulation results for doubly-even E8 (k={k}): --")
        print(f"Bob success rate       {b_corr}/{N} ({b_rate:.2f}%)")
        print(f"Eve success rate       {e_corr}/{N} ({e_rate:.2f}%)")
        print()

    print('============================================================\n')        
    # E8 for Bob, repetitive E8 for Eve
    BOB_E8_MSG_SET = ALL_E8_MSG_SET
    EVE_E8_MSG_SET = REPET_E8_MSG_SET
    for k in [2]:
        b_corr = 0
        e_corr = 0

        for i in range(N):
            m = sample_e8_msg(BOB_E8_MSG_SET)
            
            z = sample_randvec(len(m))
            x = m + (k * z)

            eB = sample_noise(x, SIGMA_BOB)
            eE = sample_noise(x, SIGMA_EVE)

            yB = x + eB
            yE = x + eE

            mB = bob_e8_decode(yB, k, BOB_E8_MSG_SET)
            mE = eve_e8_decode(yE, k, EVE_E8_MSG_SET)

            b_corr = b_corr + np.array_equal(m, mB)
            e_corr = e_corr + np.array_equal(m, mE)
            
        b_rate = (b_corr / N) * 100
        e_rate = (e_corr / N) * 100
        print(f"-- Monte Carlo simulation results for E8 for Bob and repetitive for Eve (k={k}): --")
        print(f"Bob success rate       {b_corr}/{N} ({b_rate:.2f}%)")
        print(f"Eve success rate       {e_corr}/{N} ({e_rate:.2f}%)")
        print()
        
        
    # print("-- An example of vectors in computation --")
    # print(f"Picked message            m={m}")
    # print(f"Random Vector             z={z}")
    # print(f"Encoded message           x={x}")
    # print(f"Eve Error vector          eB={eB}")    
    # print(f"Eve Error vector          eE={eE}")
    # print(f"Eve Error vector          eE={eB}")
    # print(f"Eve Recv message          yE={yE}")    
    # print(f"Eve Decode message        mE={mE}")    
    exit(0)
