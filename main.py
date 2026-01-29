from LSSS import num_to_all_and_policy_str, gen_LSSS_from_policy_str, get_omega_from_attr_list_fast
from ring_lwe_symmetric import msg_to_matrix, encrypt_msg_matrix, ring_mul_xn1, ring_vec_mul_scalar, decrypt_msg_matrix, \
    msg_matrix_to_msg_str
from ring_mul_ntt import ring_mul_xn1_ntt_numba, ring_vec_mul_scalar_ntt
from samplePre import get_secure_param_only_n, gen_trapdoor_G_trapdoor, ring_samplePre, verify_preimage, \
    get_secure_param_only_n_fast, get_secure_param_only_n_min
import numpy as np
import time

import secrets
def rand_Zq(q: int) -> int:
    """
    Sample a uniform random element from Z_q = {0,1,...,q-1}.
    Cryptographically secure.
    """
    return secrets.randbelow(q - 1) + 1

def rand_matrix_Zq(n: int, m: int, q: int) -> np.ndarray:
    """
    Generate a random matrix in Z_q^{n×m}, i.e., entries uniformly sampled from {0,...,q-1}.

    Args:
        n, m: matrix shape (n rows, m cols)
        q: modulus (q >= 2)

    Returns:
        matrix: np.ndarray of shape (n, m), dtype int64, values in [0, q-1]
    """
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive.")
    if q <= 1:
        raise ValueError("q must be >= 2.")

    rng = np.random.default_rng()
    return rng.integers(0, q, size=(n, m), dtype=np.int64)

def rand_vec_Zq_star(n, q):
    """
    Generate a vector v ∈ (Z_q^*)^n.
    Assumes q is prime.
    """
    return np.random.randint(1, q, size=n, dtype=np.int64)

def modinv(a: int, q: int) -> int:
    """
    Compute modular inverse of a modulo q.
    Requires gcd(a, q) == 1.
    """
    return pow(a, -1, q)

def split_U_by_attr(U, attr_num):
    """
    Split U ∈ Z_q^{n×eta} into attr_num blocks along columns.
    Each block has shape (n, eta // attr_num).
    """

    U = np.asarray(U, dtype=np.int64)
    n, eta = U.shape

    if eta % attr_num != 0:
        raise ValueError("eta must be divisible by attr_num")

    block_size = eta // attr_num
    U_list = [
        U[:, i*block_size:(i+1)*block_size]
        for i in range(attr_num)
    ]

    return U_list

def split_e_by_attr(e, attr_num):
    """
    Split e ∈ Z_q^{m×eta} into attr_num blocks along rows.
    Each block has shape (m // attr_num, eta).
    """

    e = np.asarray(e, dtype=np.int64)
    m, eta = e.shape

    if m % attr_num != 0:
        raise ValueError("number of rows of e must be divisible by attr_num")

    block_size = m // attr_num
    e_list = [
        e[i*block_size:(i+1)*block_size, :]
        for i in range(attr_num)
    ]

    return e_list

def setup(attr_num, q, n, eta, k, bar_m):
    # KGCSetup
    g_list=[]
    f_list=[]
    for i in range(attr_num):
        g_list.append(rand_Zq(q))
        f_list.append(rand_Zq(q))
    u = rand_matrix_Zq(n, eta, q)
    VK0 = rand_Zq(q)

    # TrapdoorSetup
    A_list = []
    T_list = []
    G_list = []
    s_k_list = []

    A, T, G, S_k = gen_trapdoor_G_trapdoor(n, q, k, bar_m, sigma=3.0)
    for i in range(attr_num):
        A_list.append(A)
        T_list.append(T)
        G_list.append(G)
        s_k_list.append(S_k)

        f_i = f_list[i]
        f_i_inv = modinv(f_i, q)
        A_i = A_list[i]

    U, U_R, G, S_k = gen_trapdoor_G_trapdoor(n, q, k, bar_m, sigma=3.0)
    # 将U拆分成attr_num，并装到U_list
    U_list = split_U_by_attr(U, attr_num)

    # TrapdoorSetup2
    r_list = []
    for i in range(attr_num):
        A_i = A_list[i]
        T_i = T_list[i]
        U_i = U_list[i]
        s_k_i = s_k_list[i]
        r_i = ring_samplePre(A_i, T_i, U_i, s_k_i, q)
        r_list.append(r_i)
    U_id = U
    T_U_id = U_R
    e = ring_samplePre(U_id, T_U_id, u, S_k, q)
    e_list =  split_e_by_attr(e, attr_num)

    # # 从A_list中获取A_i，r_list中获取r_i，e_list中获取e_i，计算A_ir_ie_i=u_i，将u_i累加得到u_1
    # u_1 = np.zeros_like(u, dtype=np.int64)
    # for i, (A_i, r_i, e_i) in enumerate(zip(A_list, r_list, e_list), start=1):
    #     A_i = (np.asarray(A_i, dtype=np.int64) % q)  # (n, m)
    #     r_i = (np.asarray(r_i, dtype=np.int64) % q)  # (m, m_blk)
    #     e_i = (np.asarray(e_i, dtype=np.int64) % q)  # (m_blk, eta)
    #
    #     # mid = r_i @ e_i : (m, eta)
    #     mid = (A_i @ r_i) % q
    #
    #     # contrib = A_i @ mid : (n, eta)
    #     contrib = (mid @ e_i) % q
    #
    #     u_1 = (u_1 + contrib) % q
    # # 判断u_1 和 u是否一样
    # same = np.array_equal(u_1 % q, u % q)
    # print(f"[check] u_1 == u (mod q)? {same}")

    h = rand_Zq(q)
    return h, g_list, r_list, f_list, e_list, VK0, u, A_list, T_list, G_list, s_k_list, U, U_R, G, S_k, U_list

def skGen(q, attr_num, h, g_list, r_list, f_list, e_list, VK0):
    t_id = rand_Zq(q)
    t_id_inv = modinv(t_id, q)

    # h_inv, 以及后续常用因子
    h_inv = modinv(h, q)

    # 预合并：sk1 的公共标量因子是 (h_inv * t_id)；sk2 的公共标量因子是 (h * VK0 * t_id_inv)
    sk1_common = (h_inv * (t_id % q)) % q
    sk2_common = (h * VK0 % q) * (t_id_inv % q) % q

    sk_1 = [None] * attr_num
    sk_2 = [None] * attr_num

    for i in range(attr_num):
        # 标量
        g_i = int(g_list[i]) % q
        g_i_inv = modinv(g_i, q)        # q 为素数时 g_i!=0 则必可逆
        f_i = int(f_list[i]) % q

        # 矩阵（先归一化，保证 dtype）
        r_i = np.asarray(r_list[i], dtype=np.int64) % q
        e_i = np.asarray(e_list[i], dtype=np.int64) % q

        # sk_1_i = (sk1_common * g_i) * r_i  (mod q)
        coef1 = (sk1_common * g_i) % q
        sk1_i = np.empty_like(r_i, dtype=np.int64)
        np.multiply(r_i, coef1, out=sk1_i)   # in-place into sk1_i
        sk1_i %= q

        # sk_2_i = (sk2_common * g_i_inv * f_i) * e_i (mod q)
        coef2 = (sk2_common * g_i_inv) % q
        coef2 = (coef2 * f_i) % q
        sk2_i = np.empty_like(e_i, dtype=np.int64)
        np.multiply(e_i, coef2, out=sk2_i)
        sk2_i %= q

        sk_1[i] = sk1_i
        sk_2[i] = sk2_i

    return sk_1, sk_2, t_id

def encrypt(n, msg, eta, leaf_node_v_list, s, s_index, attr_num, q, VK0, u, f_list, A_list, r_list, e_list):
    msg_matrix, length = msg_to_matrix(n, msg)

    if length > eta:
        raise ValueError(
            f"message length ({length}) exceeds eta ({eta}); "
            "cannot encrypt without truncation."
        )
    if length < eta:
        # 在右侧补 0 向量列
        pad_cols = eta - length
        zero_pad = np.zeros((n, pad_cols), dtype=msg_matrix.dtype)
        msg_matrix = np.concatenate([msg_matrix, zero_pad], axis=1)

    v = rand_vec_Zq_star(attr_num, q)
    v[s_index] = s
    # 将leaf_node_v_list转换成矩阵
    L = np.asarray(leaf_node_v_list, dtype=np.int64)
    # lambda = l*v
    lambda_vec = (L @ v) % q
    c_1 = lambda_vec

    a = rand_vec_Zq_star(n, q)
    # 计算tmp = VK0*u*a*s
    # ua = ring_vec_mul_scalar(u, a, q)
    ua = ring_vec_mul_scalar_ntt(u, a, q, n) # NTT加速
    VK0s = VK0*s
    c0_left = (ua*VK0s) % q
    c0_right = encrypt_msg_matrix(msg_matrix, q, alpha = 0.01)
    c0 = c0_left + c0_right

    c_2 = []
    for i in range(attr_num):
        f_i = f_list[i]
        f_i_inv = modinv(f_i, q)
        A_i = A_list[i]
        A_i_a = ring_vec_mul_scalar_ntt(A_i, a, q, n)
        c_2_i = f_i_inv*A_i_a
        c_2.append(c_2_i)

    # au = np.zeros_like(ua, dtype=np.int64)
    # for i in range(attr_num):
    #     A_i = A_list[i]
    #     A_i_a = ring_vec_mul_scalar_ntt(A_i, a, q, n)
    #     r_i = r_list[i]
    #     e_i = e_list[i]
    #     # au_i = A_i_a*r_i*e_i
    #     mid = (A_i_a @ r_i) % q
    #     au_i = (mid @ e_i) % q
    #     # au +=au_i
    #     au = (au + au_i) % q
    # # 判断au = ua 是否一致
    # same = np.array_equal(au % q, ua % q)
    # print(f"[check] au == ua (mod q)? {same}")

    return length, c0, c_1, c_2, c0_left, ua, a

def decrypt(length, attr_num, c0, c_1, c_2, sk_1, sk_2, omega_vector, n, eta, c0_left, ua, VK0):
    # 生成零矩阵ptc
    ptc = np.zeros((n, eta), dtype=np.int64)
    for i in range(attr_num):
        c_2_i = c_2[i]
        sk_1_i = sk_1[i]
        sk_2_i = sk_2[i]
        # ptc_i = c_2_i*sk_1_i*sk_2_i
        mid   = (c_2_i @ sk_1_i) % q
        ptc_i = (mid  @ sk_2_i) % q
        ptc += ptc_i
    ptc %= q

    # # 判断 ptc是否等于 ua*VK0
    # ua_VK0 = (ua * VK0) % q
    # same = np.array_equal(ptc, ua_VK0 % q)
    # print(f"[check] ptc == ua * VK0 (mod q)? {same}")


    s = int(np.dot(c_1, omega_vector) % q)
    # 计算 ptc_2 = ptc*s
    ptc_2 = (ptc * s) % q

    # # 验证 ptc_2 是否等于 c0_left
    # same = np.array_equal(ptc_2 % q, c0_left % q)
    # print(f"[check] ptc * s == c0_left (mod q)? {same}")

    c = c0-ptc_2
    msg_matrix = decrypt_msg_matrix(c, q)

    msg_matrix_length = msg_matrix[:, :length]
    msg = msg_matrix_to_msg_str(msg_matrix_length)

    return msg


def revocation(
        attr_name_list_revocation,
        leaf_node_name_list,
        q,
        g_list, r_list,
        h, t_id,
        f_list, e_list,
        VK0,
        sk_1, sk_2, c_2,
        A_list, a, n,
):
    q = int(q)

    # O(1) index lookup: avoid leaf_node_name_list.index(attr_name) in a loop
    name2idx = {name: idx for idx, name in enumerate(leaf_node_name_list)}

    # scalar normalization
    h   = int(h) % q
    VK0 = int(VK0) % q
    t_id = int(t_id) % q

    h_inv    = modinv(h, q)
    t_id_inv = modinv(t_id, q)

    # pre-combine common scalar factors
    kuk1_common = (h_inv * t_id) % q              # for KUK_1
    kuk2_common = (h * VK0 % q) * t_id_inv % q    # for KUK_2

    for attr_name in attr_name_list_revocation:
        idx = name2idx.get(attr_name, None)
        if idx is None:
            raise ValueError(f"Attribute '{attr_name}' not found in leaf_node_name_list.")

        # sample new scalars (ephemeral for update keys), but DO NOT store back to g_list/f_list
        g_new = rand_Zq(q)
        f_new = rand_Zq(q)
        f_new_inv = modinv(f_new, q)

        # old scalars
        g_old = int(g_list[idx]) % q
        f_old = int(f_list[idx]) % q
        f_old_inv = modinv(f_old, q)

        # matrices
        r_i = np.asarray(r_list[idx], dtype=np.int64) % q
        e_i = np.asarray(e_list[idx], dtype=np.int64) % q

        # KUK_1 = (g_new - g_old) * r_i * h_inv * t_id
        dg = (g_new - g_old) % q
        coef_kuk1 = (dg * kuk1_common) % q
        KUK_1 = (r_i * coef_kuk1) % q

        # KUK_2 = h * (g_new*f_new - g_old*f_old) * e_i * VK0 * t_id_inv
        diff_gf = (g_new * f_new - g_old * f_old) % q
        coef_kuk2 = (diff_gf * kuk2_common) % q
        KUK_2 = (e_i * coef_kuk2) % q

        # CUK = (f_new_inv - f_old_inv) * (A_i ⊛ a)
        A_i = np.asarray(A_list[idx], dtype=np.int64) % q
        A_i_a = ring_vec_mul_scalar_ntt(A_i, a, q, n)   # (n, m)
        df_inv = (f_new_inv - f_old_inv) % q
        CUK = (A_i_a * df_inv) % q

        # in-place updates (mod q)
        sk_1[idx] = (np.asarray(sk_1[idx], dtype=np.int64) + KUK_1) % q
        sk_2[idx] = (np.asarray(sk_2[idx], dtype=np.int64) + KUK_2) % q
        c_2[idx]  = (np.asarray(c_2[idx],  dtype=np.int64) + CUK)  % q

    return sk_1, sk_2, c_2

if __name__ == '__main__':
    # n = 256
    n=256

    n, k, q, w, bar_m, m = get_secure_param_only_n_min(n)

    # n, k, q, w, bar_m, m = get_secure_param_only_n(n)
    # attr_num = 256
    attr_num = 256
    eta = int(m/attr_num)

    policy_str = num_to_all_and_policy_str(attr_num)
    s = 50
    s_index = 1
    msg = "hello"

    start = time.time()
    h, g_list, r_list, f_list, e_list, VK0, u, A_list, T_list, G_list, s_k_list, U, U_R, G, S_k, U_list = setup(attr_num, q, n, eta, k, bar_m)
    end = time.time()
    elapsed_ms = (end - start) * 1000
    print(f"setup耗时: {elapsed_ms:.3f} ms")

    start = time.time()
    sk_1, sk_2, t_id = skGen(q, attr_num, h, g_list, r_list, f_list, e_list, VK0)
    end = time.time()
    elapsed_ms = (end - start) * 1000
    print(f"skGen耗时: {elapsed_ms:.3f} ms")

    # for i in range(attr_num):
    #     r_i = r_list[i]
    #     f_i = f_list[i]
    #     e_i = e_list[i]
    #     # left = r_i*f_i*e_i*VK0
    #     left = (r_i @ e_i) % q            # (m, eta)
    #     left = (left * f_i) % q
    #     left = (left * VK0) % q
    #
    #     sk_1_i = sk_1[i]
    #     sk_2_i = sk_2[i]
    #     sk_i = (sk_1_i @ sk_2_i) % q
    #
    #     # 判断 left 与 sk_i是否一致
    #     same = np.array_equal(left % q, sk_i % q)
    #     print(f"[check attr {i}] r_i*f_i*e_i*VK0 == sk_1_i@sk_2_i (mod q)? {same}")

    start = time.time()
    leaf_node_name_list, leaf_node_v_list = gen_LSSS_from_policy_str(policy_str)
    length, c0, c_1, c_2, c0_left, ua, a = encrypt(n, msg, eta, leaf_node_v_list, s, s_index, attr_num, q, VK0, u, f_list, A_list, r_list, e_list)
    end = time.time()
    elapsed_ms = (end - start) * 1000
    print(f"encrypt耗时: {elapsed_ms:.3f} ms")

    # start = time.time()
    user_attr_list = leaf_node_name_list
    omega_vector = get_omega_from_attr_list_fast(leaf_node_name_list,leaf_node_v_list, user_attr_list,s_index)
    start = time.time()
    msg_decrypt = decrypt(length, attr_num, c0, c_1, c_2, sk_1, sk_2, omega_vector, n, eta, c0_left, ua, VK0)
    end = time.time()
    elapsed_ms = (end - start) * 1000
    print(f"decrypt耗时: {elapsed_ms:.3f} ms")
    print(msg_decrypt)

    start = time.time()
    attr_name_list_revocation = ["a1","a2"]
    sk_1, sk_2, c_2 = revocation(attr_name_list_revocation,leaf_node_name_list,q,g_list, r_list,h, t_id,f_list, e_list,VK0,sk_1, sk_2, c_2,A_list, a, n,)
    end = time.time()
    elapsed_ms = (end - start) * 1000
    print(f"revocation耗时: {elapsed_ms:.3f} ms")