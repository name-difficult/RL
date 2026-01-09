import numpy as np
from numba import njit, prange

# 可选：用 sympy 找 primitive root，简单可靠
try:
    from sympy import isprime, primitive_root
except Exception:  # 允许你在没有 sympy 时自行替换
    isprime = None
    primitive_root = None

# -----------------------------
# 1) 全局缓存：按 (n, q) 缓存 NTT 计划
# -----------------------------
_NTT_CACHE = {}

def _bit_reverse_indices(N: int) -> np.ndarray:
    bits = (N - 1).bit_length()
    rev = np.empty(N, dtype=np.int64)
    for i in range(N):
        x = i
        r = 0
        for _ in range(bits):
            r = (r << 1) | (x & 1)
            x >>= 1
        rev[i] = r
    return rev

def _build_twiddles_concat(N: int, q: int, rootN: int):
    """
    Build twiddles in a concatenated 1D array plus offsets per stage.
    Stage lengths: 2,4,...,N. For each stage len, store half=len/2 twiddles:
      w[j] = root^(j * N/len) mod q  (iteratively built)
    Returns:
      tw: int64 array of all twiddles concatenated
      off: int64 array offsets of length (num_stages+1)
    """
    tw_list = []
    offsets = [0]
    length = 2
    while length <= N:
        half = length // 2
        wlen = pow(rootN, N // length, q)
        w = np.empty(half, dtype=np.int64)
        w[0] = 1
        for i in range(1, half):
            w[i] = (w[i - 1] * wlen) % q
        tw_list.append(w)
        offsets.append(offsets[-1] + half)
        length *= 2

    tw = np.concatenate(tw_list).astype(np.int64)
    off = np.asarray(offsets, dtype=np.int64)
    return tw, off

def _get_ntt_plan(n: int, q: int):
    """
    Return cached plan for (n,q). Plan is for length N=2n NTT (negacyclic mul).
    Requires:
      - n power of two
      - q prime
      - (q-1) divisible by 2n
    """
    key = (int(n), int(q))
    if key in _NTT_CACHE:
        return _NTT_CACHE[key]

    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n must be a positive power of two.")
    N = 2 * n
    if (N & (N - 1)) != 0:
        raise ValueError("2n must be power of two (n must be power of two).")

    if (q - 1) % N != 0:
        raise ValueError(f"Need (q-1) divisible by 2n={N} for length-2n NTT.")

    if isprime is None or primitive_root is None:
        raise RuntimeError("sympy not available; install sympy or provide your own primitive_root/isprime.")

    if not bool(isprime(q)):
        raise ValueError("q must be prime for this NTT implementation.")

    g = int(primitive_root(q))
    rootN = pow(g, (q - 1) // N, q)           # primitive N-th root
    inv_rootN = pow(rootN, q - 2, q)          # inverse root
    invN = pow(N, q - 2, q)                   # N^{-1} mod q

    rev = _bit_reverse_indices(N)

    tw_fwd, off = _build_twiddles_concat(N, q, rootN)
    tw_inv, _   = _build_twiddles_concat(N, q, inv_rootN)

    plan = {
        "n": int(n),
        "N": int(N),
        "q": int(q),
        "rootN": int(rootN),
        "inv_rootN": int(inv_rootN),
        "invN": int(invN),
        "rev": rev,
        "tw_fwd": tw_fwd,
        "tw_inv": tw_inv,
        "off": off,
    }
    _NTT_CACHE[key] = plan
    return plan

# -----------------------------
# 2) Numba 内核：1D NTT / iNTT（twiddle 用 concat+offset）
# -----------------------------
@njit(cache=True)
def _ntt_inplace_1d(a, q, rev, tw, off):
    """
    Forward NTT in-place on 1D array a (length N).
    tw: concatenated twiddles
    off: offsets per stage
    """
    N = a.shape[0]

    # bit-reversal permutation
    tmp = a.copy()
    for i in range(N):
        a[i] = tmp[rev[i]]

    length = 2
    stage = 0
    while length <= N:
        half = length // 2
        base = off[stage]  # twiddles for this stage start at tw[base:base+half]

        for start in range(0, N, length):
            for j in range(half):
                u = a[start + j]
                v = a[start + j + half]
                t = (v * tw[base + j]) % q
                a[start + j] = (u + t) % q
                a[start + j + half] = (u - t) % q

        stage += 1
        length *= 2

@njit(cache=True)
def _intt_inplace_1d(a, q, rev, tw_inv, off, invN):
    """
    Inverse NTT in-place on 1D array a (length N), then scale by invN.
    """
    _ntt_inplace_1d(a, q, rev, tw_inv, off)

    N = a.shape[0]
    for i in range(N):
        a[i] = (a[i] * invN) % q

@njit(cache=True)
def _pointwise_mul_inplace(a, b, q):
    N = a.shape[0]
    for i in range(N):
        a[i] = (a[i] * b[i]) % q

# -----------------------------
# 3) 单次 negacyclic 乘法：ring_mul_xn1 的 NTT 替代
# -----------------------------
def ring_mul_xn1_ntt_numba(a: np.ndarray, b: np.ndarray, q: int, n: int) -> np.ndarray:
    """
    Multiply in R_q = Z_q[x]/(x^n+1) using length-2n NTT + folding.
    a,b: shape (n,)
    return: shape (n,)
    """
    plan = _get_ntt_plan(n, q)
    q0 = np.int64(plan["q"])
    N = plan["N"]

    a = np.asarray(a, dtype=np.int64) % q0
    b = np.asarray(b, dtype=np.int64) % q0
    if a.ndim != 1 or b.ndim != 1 or a.shape[0] != n or b.shape[0] != n:
        raise ValueError(f"a,b must be 1D with shape ({n},).")

    A2 = np.zeros(N, dtype=np.int64)
    B2 = np.zeros(N, dtype=np.int64)
    A2[:n] = a
    B2[:n] = b

    _ntt_inplace_1d(A2, plan["q"], plan["rev"], plan["tw_fwd"], plan["off"])
    _ntt_inplace_1d(B2, plan["q"], plan["rev"], plan["tw_fwd"], plan["off"])
    _pointwise_mul_inplace(A2, B2, plan["q"])
    _intt_inplace_1d(A2, plan["q"], plan["rev"], plan["tw_inv"], plan["off"], plan["invN"])

    # fold to mod (x^n+1): res[k] = C2[k] - C2[k+n]
    res = (A2[:n] - A2[n:]) % q0
    return res.astype(np.int64)

# -----------------------------
# 4) 批量：A 是 (n,m)，同一个 a 乘所有列（并行）
# -----------------------------
@njit(cache=True, parallel=True)
def _ring_vec_mul_scalar_ntt_kernel(A, a, out, q, n, N, rev, tw_fwd, tw_inv, off, invN):
    """
    out[:,j] = negacyclic_mul(A[:,j], a) for all j, parallel over columns.
    A: (n,m), a:(n,), out:(n,m)
    """
    m = A.shape[1]

    # 预计算 Fa = NTT([a,0]) 一次
    a2 = np.zeros(N, dtype=np.int64)
    for i in range(n):
        a2[i] = a[i] % q
    _ntt_inplace_1d(a2, q, rev, tw_fwd, off)

    for j in prange(m):
        x = np.zeros(N, dtype=np.int64)
        # pad
        for i in range(n):
            x[i] = A[i, j] % q

        _ntt_inplace_1d(x, q, rev, tw_fwd, off)
        _pointwise_mul_inplace(x, a2, q)
        _intt_inplace_1d(x, q, rev, tw_inv, off, invN)

        for i in range(n):
            out[i, j] = (x[i] - x[i + n]) % q

def ring_vec_mul_scalar_ntt(A: np.ndarray, a: np.ndarray, q: int, n: int) -> np.ndarray:
    """
    A: (n,m) coefficient matrix (each column poly in R_q)
    a: (n,) poly in R_q
    return: (n,m) where each column is A[:,j] ⊛ a in R_q (mod x^n+1)
    """
    plan = _get_ntt_plan(n, q)
    q0 = np.int64(plan["q"])

    A = np.asarray(A, dtype=np.int64) % q0
    a = np.asarray(a, dtype=np.int64) % q0
    if A.ndim != 2 or A.shape[0] != n:
        raise ValueError(f"A must have shape (n,m) with n={n}. Got {A.shape}")
    if a.ndim != 1 or a.shape[0] != n:
        raise ValueError(f"a must have shape (n,) with n={n}. Got {a.shape}")

    out = np.empty_like(A, dtype=np.int64)

    _ring_vec_mul_scalar_ntt_kernel(
        A, a, out,
        plan["q"], plan["n"], plan["N"],
        plan["rev"], plan["tw_fwd"], plan["tw_inv"], plan["off"], plan["invN"]
    )
    return out
