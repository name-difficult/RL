import numpy as np
from typing import Tuple, Dict, Any, Optional

# -----------------------------
# Ring: R_p = Z_p[x] / (x^n + 1)
# Representation: ring element = length-n int vector (coeffs mod p)
# Multiplication: negacyclic convolution (x^n == -1)
# -----------------------------

def ring_mul_xn1(a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    """
    Multiply in R_p = Z_p[x]/(x^n+1).
    a,b: shape (n,), int64
    return: shape (n,), int64 mod p
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if a.ndim != 1 or b.ndim != 1 or a.shape[0] != b.shape[0]:
        raise ValueError("ring_mul_xn1 expects 1D arrays of same length.")
    n = a.shape[0]

    # schoolbook convolution in Z
    c = np.zeros(2 * n - 1, dtype=np.int64)
    for i in range(n):
        ai = int(a[i])
        if ai == 0:
            continue
        for j in range(n):
            c[i + j] += ai * int(b[j])

    # reduce mod (x^n + 1): x^n == -1
    res = np.zeros(n, dtype=np.int64)
    res[:n] = c[:n]
    for k in range(n, 2 * n - 1):
        res[k - n] -= c[k]

    return (res % p).astype(np.int64)

def ring_vec_mul_scalar(A: np.ndarray, a: np.ndarray, q: int) -> np.ndarray:
    """
    A ∈ R_q^m as coefficient matrix shape (n, m) (each column is a poly).
    a ∈ R_q as coefficient vector shape (n,).
    Return A*a ∈ R_q^m as shape (n, m).
    """
    q64 = np.int64(q)
    A = np.asarray(A, dtype=np.int64) % q64
    a = np.asarray(a, dtype=np.int64) % q64

    n, m = A.shape
    if a.ndim != 1 or a.shape[0] != n:
        raise ValueError(f"a must have shape (n,) with n={n}, got {a.shape}")

    out = np.empty_like(A, dtype=np.int64)
    for j in range(m):
        out[:, j] = ring_mul_xn1(A[:, j], a, q)  # 环乘
    return out

def msg_to_matrix(n: int, msg: str) -> Tuple[np.ndarray, int]:
    """
    Convert a message string into an n × l binary matrix and return l.

    Each column corresponds to one character:
      - start from an n-dimensional zero vector
      - replace the first 8 entries with the 8-bit ASCII code (MSB -> LSB)

    Args:
        n (int): dimension of each column vector (n >= 8)
        msg (str): ASCII message

    Returns:
        msg_matrix (np.ndarray): shape (n, l), dtype int64
        l (int): number of characters (columns)
    """
    if n < 8:
        raise ValueError("n must be at least 8.")
    if not msg:
        raise ValueError("msg must be a non-empty string.")

    l = len(msg)
    msg_matrix = np.zeros((n, l), dtype=np.int64)

    for j, ch in enumerate(msg):
        ascii_val = ord(ch)
        if ascii_val >= 256:
            raise ValueError(f"Non-ASCII character detected: {ch!r}")

        # 8-bit binary (MSB -> LSB)
        bits = [(ascii_val >> i) & 1 for i in range(7, -1, -1)]
        msg_matrix[:8, j] = bits

    return msg_matrix, l

def msg_matrix_to_msg_str(msg_matrix: np.ndarray) -> str:
    """
    Inverse of msg_to_matrix.

    Convert an n × l binary matrix back to a string by reading
    the first 8 bits of each column as an ASCII code (MSB -> LSB).

    Args:
        msg_matrix (np.ndarray): shape (n, l), entries in {0,1}

    Returns:
        msg (str): decoded ASCII string
    """
    if msg_matrix.ndim != 2:
        raise ValueError("msg_matrix must be a 2D array.")

    n, l = msg_matrix.shape
    if n < 8:
        raise ValueError("msg_matrix must have at least 8 rows.")

    msg_chars = []

    for j in range(l):
        bits = msg_matrix[:8, j]

        # sanity check (optional but recommended)
        if not np.all((bits == 0) | (bits == 1)):
            raise ValueError(f"Non-binary entry detected in column {j}.")

        # reconstruct ASCII value (MSB -> LSB)
        ascii_val = 0
        for bit in bits:
            ascii_val = (ascii_val << 1) | int(bit)

        if ascii_val >= 256:
            raise ValueError(f"Invalid ASCII value {ascii_val} in column {j}.")

        msg_chars.append(chr(ascii_val))

    return "".join(msg_chars)

def sample_rlwe_error(shape, q: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """
    RLWE-style small error:
      Y ~ N(0, (alpha/sqrt(2π))^2)
      e = round(q * Y)  (integer, centered around 0)

    Returned in Z_q for storage, but fundamentally 'small' in centered form.
    """
    if q <= 1:
        raise ValueError("q must be >= 2.")
    if not (alpha > 0):
        raise ValueError("alpha must be > 0.")
    if rng is None:
        rng = np.random.default_rng()

    sigma = alpha / np.sqrt(2.0 * np.pi)   # std of Y
    y = rng.normal(loc=0.0, scale=sigma, size=shape)
    e = np.rint(q * y).astype(np.int64)    # small integers (can be negative)
    return e % q


def encrypt_msg_matrix(
        msg_matrix: np.ndarray,
        q: int,
        *,
        alpha: float,
) -> np.ndarray:
    """
    Symmetric "message + noise" encryption (as you specified):

      1) generate noise according to the paper definition (Φ_α -> ϕ on Z_q)
      2) c = msg_matrix * floor(q/2) + noise   (mod q)

    Args:
      msg_matrix: binary matrix (0/1), shape (n, l)
      q: modulus
      alpha: parameter in Φ_α (must be > 0)
      rng: optional numpy Generator

    Returns:
      c: ciphertext matrix in Z_q, shape (n, l), dtype int64
    """
    rng = np.random.default_rng()

    M = np.asarray(msg_matrix, dtype=np.int64)
    if M.ndim != 2:
        raise ValueError("msg_matrix must be a 2D array.")
    if not np.all((M == 0) | (M == 1)):
        raise ValueError("msg_matrix entries must be 0/1.")
    if q <= 1:
        raise ValueError("q must be >= 2.")

    Delta = q // 2
    noise = sample_rlwe_error(M.shape, q=q, alpha=alpha, rng=rng)

    c = (M * Delta + noise) % q
    return c

def _centered_mod(a: np.ndarray, q: int) -> np.ndarray:
    """Map to (-q/2, q/2]."""
    a = a % q
    half = q // 2
    return np.where(a > half, a - q, a).astype(np.int64)


def decrypt_msg_matrix(c: np.ndarray, q: int) -> np.ndarray:
    """
    Decision with threshold q/4 on the centered value:
      if |center(c)| < q/4 -> 0
      else                 -> 1
    """
    C = np.asarray(c, dtype=np.int64)
    if C.ndim != 2:
        raise ValueError("c must be 2D.")
    if q <= 1:
        raise ValueError("q must be >= 2.")

    thr = q / 4.0
    Cc = _centered_mod(C, q)
    return (np.abs(Cc) >= thr).astype(np.int64)


def keyGen_sym_sk(
        *,
        n: int,
        p: int,
        eta: int,
        rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    KeyGen (symmetric RLWE masking, Semantic-1):
      - Define ring R_p = Z_p[x]/(x^n+1)
      - Sample:
          VK0 ∈ Z_p^*
          u ∈ R_p^eta
          a ∈ R_p
          s ∈ Z_p^*
      - Compute secret mask key (vector in R_p^eta):
          sk = VK0 · u · a · s
        interpreted as component-wise ring multiplication:
          sk_lambda = (VK0*s) * (u_lambda * a)  in R_p, for lambda=1..eta

    Returns:
      sk: np.ndarray shape (eta, n), coefficients mod p
      ctx: dict containing sampled (VK0, u, a, s) for debugging/verification
    """
    if p <= 2:
        raise ValueError("p must be >= 3.")
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n must be a power of 2 (as in your text).")
    if eta <= 0:
        raise ValueError("eta must be positive.")
    if rng is None:
        rng = np.random.default_rng()

    # Sample scalars
    VK0 = int(rng.integers(1, p))  # in Z_p^*
    s   = int(rng.integers(1, p))  # in Z_p^*

    # Sample ring elements (uniform coefficients mod p)
    a = rng.integers(0, p, size=n, dtype=np.int64)          # a ∈ R_p
    u = rng.integers(0, p, size=(eta, n), dtype=np.int64)   # u ∈ R_p^eta

    # Compute sk ∈ R_p^eta
    k_scalar = (VK0 * s) % p
    sk = np.zeros((eta, n), dtype=np.int64)
    for lam in range(eta):
        ua = ring_mul_xn1(u[lam], a, p)              # u_lam * a ∈ R_p
        sk[lam] = (ua * k_scalar) % p               # scalar multiply in R_p

    ctx = {"n": n, "p": p, "eta": eta, "VK0": VK0, "s": s, "a": a, "u": u}
    return sk, ctx

# Example usage
if __name__ == "__main__":
    n = 256
    q = 17921

    msg = "hello"
    msg_matrix, l = msg_to_matrix(n, msg)
    c = encrypt_msg_matrix(msg_matrix, q, alpha = 0.01)
    msg_matrix_1 = decrypt_msg_matrix(c, q)
    msg_1 = msg_matrix_to_msg_str(msg_matrix_1)
    print(msg_1)


