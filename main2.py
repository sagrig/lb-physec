import numpy as np

SIGMA_BOB=0.2
SIGMA_EVE=0.25
K=2
RNG=np.random.default_rng()

ZkZ_MSG_SET = [(a, b) for a in range(K) for b in range(K)]

def sample_msg():    
    rand_idx = np.random.randint(len(ZkZ_MSG_SET))
    return ZkZ_MSG_SET[rand_idx]

def sample_randvec(tuple_size):
    # limit here is a constant, although maybe need to be random?    
    limit    = 100
    rand_int = np.random.randint(-limit, limit, size=tuple_size)
    return rand_int

def sample_noise(x, sigma):
    noise = RNG.normal(0.0, sigma, size=x.shape)
    return noise

def bob_decode_old(yB):
    yB_hat = np.rint(yB).astype(int)
    return tuple((yB_hat % K).astype(int))

def bob_decode_new(yB):
    yB_hat = np.rint(yB).astype(int)
    yB_mod = np.mod(yB_hat, K)

    best_m = None
    best_d = float("inf")

    for m in ZkZ_MSG_SET:
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
def eve_decode(yE):
    yE_norm = yE / K

    best_m = None
    best_d = float("inf")

    for m in ZkZ_MSG_SET:
        shift = np.array(m) / K
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
    bo_corr = 0
    bn_corr = 0
    e_corr  = 0
    N = 10000
    for i in range(N):
        m  = sample_msg()
        z  = sample_randvec(len(m))
        x  = m + (K * z)

        eBo = sample_noise(x, SIGMA_BOB)
        eBn = sample_noise(x, SIGMA_BOB)
        eE = sample_noise(x, SIGMA_EVE)

        yBo = x + eBo
        yBn = x + eBn
        yE = x + eE

        mBo = bob_decode_old(yBo)
        mBn = bob_decode_new(yBn)
        mE = eve_decode(yE)

        bo_corr = bo_corr + (m == mBo)
        bn_corr = bn_corr + (m == mBn)
        e_corr = e_corr + (m == mE)

    bo_rate = (bo_corr / N) * 100
    bn_rate = (bn_corr / N) * 100
    e_rate  = (e_corr / N) * 100

    print("-- Monte Carlo simulation results for Z/{K}Z: --")
    print(f"Bob (old) success rate {bo_corr}/{N} ({bo_rate:.2f}%)")
    print(f"Bob (new) success rate {bn_corr}/{N} ({bn_rate:.2f}%)")
    print(f"Eve success rate       {e_corr}/{N} ({e_rate:.2f}%)")

    print()
    print("-- An example of vectors in computation --")
    print(f"Picked message            m={m}")
    print(f"Random Vector             z={z}")
    print(f"Encoded message           x={x}")
    print(f"Bob (old) Error vector    eB(old)={eBo}")
    print(f"Bob (new) Error vector    eB(new)={eBn}")
    print(f"Eve Error vector          eE={eE}")
    print(f"Bob (old) Recv message    yB(old)={yBo}")
    print(f"Bob (new) Recv message    yB(new)={yBn}")
    print(f"Eve Recv message          yE={yE}")    
    print(f"Bob (old) Decode message  mB(old)={mBo}")    
    print(f"Bob (new) Decode message  mB(new)={mBn}")
    print(f"Eve Decode message        mE={mE}")    
    exit(0)
