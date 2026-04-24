import numpy as np

SIGMA_BOB=0.2
SIGMA_EVE=0.3
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

# Bob's decode algorithm:
# 1) yB_hat = Approximate the yB noisy vector to the closest integers
# 2) mB = yB_hat (mod K)
#
# mB is the decoded message
def bob_decode(yB):
    yB_hat = np.rint(yB).astype(int)
    mB     = tuple((yB_hat % K).astype(int))
    return mB

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
        shift       = np.array(m) / K
        nearest_int = np.rint(yE_norm - shift)
        d           = np.linalg.norm((yE_norm - shift) - nearest_int)

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
    m  = sample_msg()
    z  = sample_randvec(len(m))
    x  = m + (K * z)

    eB = sample_noise(x, SIGMA_BOB)
    eE = sample_noise(x, SIGMA_EVE)

    yB = x + eB
    yE = x + eE

    mB = bob_decode(yB)
    mE = eve_decode(yE)
    
    print(f"Picked message      m={m}")
    print(f"Random Vector       z={z}")
    print(f"Encoded message     x={x}")
    print(f"Bob Error vector    eB={eB}")
    print(f"Eve Error vector    eE={eE}")
    print(f"Bob Recv message    yB={yB}")
    print(f"Eve Recv message    yE={yE}")    
    print(f"Bob Decode message  mB={mB}")    
    print(f"Eve Decode message  mE={mE}")    
    exit(0)
