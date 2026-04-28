import numpy as np

SIGMA_BOB=0.2
SIGMA_EVE=0.3
K=[2, 3, 4]
N=3000
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

def bob_decode_old(yB, k):
    yB_hat = np.rint(yB).astype(int)
    return tuple((yB_hat % k).astype(int))

def bob_decode_new(yB, k, ZkZ_msg_set):
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
def eve_decode(yE, k, ZkZ_msg_set):
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

E8_MSG_SET=all_hamming_codewords()
del all_hamming_codewords

def sample_e8_msg():
    idx = np.random.randint(len(E8_MSG_SET))
    return E8_MSG_SET[idx]

def eve_e8_decode(yE, k):
    yE_norm = yE / k

    best_m = None
    best_d = float("inf")

    for m in E8_MSG_SET:
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
    for k in K:        
        ZkZ_msg_set = [(a, b) for a in range(k) for b in range(k)]
        bo_corr = 0
        bn_corr = 0
        e_corr  = 0

        for i in range(N):
            m  = sample_msg(ZkZ_msg_set)
            z  = sample_randvec(len(m))
            x  = m + (k * z)

            eBo = sample_noise(x, SIGMA_BOB)
            eBn = sample_noise(x, SIGMA_BOB)
            eE = sample_noise(x, SIGMA_EVE)

            yBo = x + eBo
            yBn = x + eBn
            yE = x + eE

            mBo = bob_decode_old(yBo, k)
            mBn = bob_decode_new(yBn, k, ZkZ_msg_set)
            mE = eve_decode(yE, k, ZkZ_msg_set)

            bo_corr = bo_corr + (m == mBo)
            bn_corr = bn_corr + (m == mBn)
            e_corr = e_corr + (m == mE)

        bo_rate = (bo_corr / N) * 100
        bn_rate = (bn_corr / N) * 100
        e_rate  = (e_corr / N) * 100

        print(f"-- Monte Carlo simulation results for Z/{k}Z: --")
        print(f"Bob (old) success rate {bo_corr}/{N} ({bo_rate:.2f}%)")
        print(f"Bob (new) success rate {bn_corr}/{N} ({bn_rate:.2f}%)")
        print(f"Eve success rate       {e_corr}/{N} ({e_rate:.2f}%)")
        print()

    # E8 implementation
    k = 2
    
    m = sample_e8_msg()
    z = sample_randvec(len(m))
    x = m + (k * z)

    eB = sample_noise(x, SIGMA_BOB)
    eE = sample_noise(x, SIGMA_EVE)

    yB = x + eB
    yE = x + eE

    # mB = ???
    mE = eve_e8_decode(yE, k)

    
    print("-- An example of vectors in computation --")
    print(f"Picked message            m={m}")
    print(f"Random Vector             z={z}")
    print(f"Encoded message           x={x}")
    print(f"Eve Error vector          eB={eB}")    
    print(f"Eve Error vector          eE={eE}")
    print(f"Eve Error vector          eE={eB}")
    print(f"Eve Recv message          yE={yE}")    
    print(f"Eve Decode message        mE={mE}")    
    exit(0)
