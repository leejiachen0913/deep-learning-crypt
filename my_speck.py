# L_{i+1}:=((L_{i}>>alpha)+(mod) R_{i})+k, R_{i+1}:=(R_{i} << beta)+L_{i+1}
import numpy as np
from os import urandom
import torch


def WORD_SIZE():
    return 16


def ALPHA():
    return 7


def BETA():
    return 2


# 二进制全为1
MASK_VAL = 2 ** WORD_SIZE() - 1


def shuffle_together(l):
    state = np.random.get_state()
    for x in l:
        np.random.set_state(state)
        np.random.shuffle(x)


# 左移
def rol(x, k):
    return ((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k))


# 右移
def ror(x, k):
    return (x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL)


def dec_one_round(c, k):
    c0, c1 = c[0], c[1]
    # print("c[0] shape",c[0].shape)
    # print("c[1] shape",c[1].shape)
    c1 = c1 ^ c0
    c1 = ror(c1, BETA())
    c0 = c0 ^ k
    c0 = (c0 - c1) & MASK_VAL
    c0 = rol(c0, ALPHA())
    # print(c1.shape)
    return c0, c1


def expand_key(k, t):
    ks = [0 for i in range(t)]
    ks[0] = k[len(k) - 1]
    l = list(reversed(k[:len(k) - 1]))
    for i in range(t - 1):
        l[i % 3], ks[i + 1] = enc_one_round((l[i % 3], ks[i]), i)
    return ks


def enc_one_round(p, k):
    c0, c1 = p[0], p[1]
    # print(c0.shape)
    c0 = ror(c0, ALPHA())
    # & MASK_VAL 模加操作
    c0 = (c0 + c1) & MASK_VAL
    c0 = c0 ^ k
    c1 = rol(c1, BETA())
    c1 = c1 ^ c0
    return c0, c1


def encrypt(p, ks):
    x, y = p[0], p[1]
    # print(x.shape)
    for k in ks:
        x, y = enc_one_round((x, y), k)
    return x, y


def decrypt(c, ks):
    x, y = c[0], c[1]
    for k in reversed(ks):
        x, y = dec_one_round((x, y), k)
    return x, y


def check_testvector():
    key = (0x1918, 0x1110, 0x0908, 0x0100)
    pt = (0x6574, 0x694c)
    ks = expand_key(key, 22)
    ct = encrypt(pt, ks)
    if ct == (0xa868, 0x42f2):
        print("Testvector verified.")
        return True
    else:
        print("Testvector not verified.")
        return False


# convert_to_binary takes as input an array of ciphertext pairs
# where the first row of the array contains the lefthand side of the ciphertexts,
# the second row contains the righthand side of the ciphertexts,
# the third row contains the lefthand side of the second ciphertexts,
# and so on
# it returns an array of bit vectors containing the same data
# def convert_to_binary(arr):
#     X = np.zeros((4 * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
#     for i in range(4 * WORD_SIZE()):
#         index = i // WORD_SIZE();
#         offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
#         X[i] = (arr[index] >> offset) & 1;
#     X = X.transpose();
#     return(X);

# def convert_to_binary(arr):
#     # print(arr.shape)
#     X = np.empty((6 * WORD_SIZE(),len(arr[0])),dtype=np.bool);
#     # print(arr[0])
#     for i in range(6 * WORD_SIZE()):
#         index = i // WORD_SIZE();
#         offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
#         X[i] = (arr[index] >> offset) & 1;
#     X = X.transpose();
#     return(X);
def convert_to_binary(l):
    n_words = len(l)
    n_samples = len(l[0])
    word_size = 16

    X = np.zeros((n_words, n_samples, word_size), dtype=np.uint8)

    for i in range(n_words):
        for j in range(word_size):
            # ✅ MSB-first（Gohr 用的）
            X[i, :, j] = (l[i] >> (15 - j)) & 1

    return X  # (6, n_total, 16)



# real differences data generator
def real_differences_data(n, nr, pairs=2, diff=(0x0040, 0)):
    # generate labels
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    Y = np.tile(Y, pairs)
    # generate keys
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    keys = np.tile(keys, pairs)
    # generate plaintexts
    plain0l = np.frombuffer(urandom(2 * n * pairs), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2 * n * pairs), dtype=np.uint16)
    # apply input difference
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    num_rand_samples = np.sum(Y == 0)
    # expand keys and encrypt
    ks = expand_key(keys, nr)
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
    # generate blinding values
    # 加入噪声
    k0 = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
    k1 = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
    # apply blinding to the samples labelled as random
    ctdata0l[Y == 0] = ctdata0l[Y == 0] ^ k0
    ctdata0r[Y == 0] = ctdata0r[Y == 0] ^ k1
    ctdata1l[Y == 0] = ctdata1l[Y == 0] ^ k0
    ctdata1r[Y == 0] = ctdata1r[Y == 0] ^ k1
    # convert to input data for neural networks
    # R0 = ror(ctdata0l^ctdata0r,BETA())
    # R1 = ror(ctdata1l^ctdata1r,BETA())
    # X = convert_to_binary([R0,R1,ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
    # X = X.reshape(pairs,n,16*6).transpose((1,0,2))
    # X = X.reshape(n,1,-1)
    # X = np.squeeze(X)

    ctdata0l = ctdata0l.reshape(pairs, n).transpose().flatten()
    ctdata0r = ctdata0r.reshape(pairs, n).transpose().flatten()
    ctdata1l = ctdata1l.reshape(pairs, n).transpose().flatten()
    ctdata1r = ctdata1r.reshape(pairs, n).transpose().flatten()
    R0 = ror(ctdata0l ^ ctdata0r, BETA())
    R1 = ror(ctdata1l ^ ctdata1r, BETA())

    # -------------------------------
    # 7. 转换为二进制并【物理对齐】
    # -------------------------------
    X = convert_to_binary([R0, R1, ctdata0l, ctdata0l, ctdata1l, ctdata1r])  # 返回 (6, n_total, 16)

    # 变换为 (n_total, 6, 16)
    X = X.transpose((1, 0, 2))

    # 变换为 (n, pairs, 6, 16)
    n_total = n * pairs
    X = X.reshape(n, pairs, 6, 16)

    # 展平为模型需要的 (n, 192)
    X = X.reshape(n, -1)

    # --- 最终验证打印 ---
    # 只有这里显示为一个 One-hot 向量，模型才能学到东西
    sample_pos = X[Y == 1]
    if len(sample_pos) > 0:
        # 验证第一个正样本的 c0l (0:16) 和 c1l (32:48) 的差异
        w0 = sample_pos[0, 0:16]
        w2 = sample_pos[0, 32:48]
        print(f"DEBUG: Final Diff Check: {(w0 != w2).astype(int)}")

    return X, Y


class SpeckCipher:
    """
    SPECK-like ARX cipher wrapper for neural cryptanalysis
    """

    def __init__(self):
        self.block_bits = 32
        self.word_bits = 16
        self.name = "speck"

    def state_bits(self) -> int:
        return self.block_bits

    def diff_words(self) -> int:
        """
        返回差分向量包含的 word 数量
        SPECK-32: 2 words
        """
        return 2

    def diff_bits_to_matrix(self, diff_bits):
        """
        diff_bits: (64,) 0/1 tensor or numpy array
        return: (2,) uint16 array，用于 XOR
        """
        # 如果是 torch.Tensor 且在 GPU 上，需要先移动到 CPU
        if isinstance(diff_bits, torch.Tensor):
            diff_bits = diff_bits.detach().cpu().numpy()

        diff_bits = np.array(diff_bits, dtype=np.uint8).flatten()

        if len(diff_bits) != 32:  # SPECK-32/64, 每 word 16 bit, 2 words
            raise ValueError("diff_bits length must be 32")

        diff_uint16 = np.zeros(2, dtype=np.uint16)
        for i in range(2):
            bits = diff_bits[i * 16:(i + 1) * 16]
            val = 0
            for j in range(16):
                val |= (int(bits[j]) << j)  # LSB-first
            diff_uint16[i] = val

        return diff_uint16

    def num_nibbles(self) -> int:
        return self.block_bits // 4  # 8 个 nibble

    def max_active_nibbles(self) -> int:
        return 2

    def word_size(self) -> int:
        return self.word_bits

    def input_dim(self):
        """
        返回神经网络的输入维度（不含 batch）
        你现在的 X 最终是 (n, pairs*16*6) 或 (n, 96) 等
        """
        return None  # 由数据生成阶段动态决定

    # ========= 核心接口：训练数据 =========

    def make_train_data(self, n, nr, pairs=2, diff=(0x0040, 0)):
        Y = np.random.randint(0, 2, n).astype(np.uint8)

        # 保持 (n, pairs) 的形状
        p0l = np.frombuffer(urandom(2 * n * pairs), dtype=np.uint16).reshape(n, pairs)
        p0r = np.frombuffer(urandom(2 * n * pairs), dtype=np.uint16).reshape(n, pairs)
        p1l = np.copy(p0l)
        p1r = np.copy(p0r)

        for i in range(n):
            if Y[i] == 1:
                # 🚨 修正：直接操作第 i 行（包含该样本的所有 pairs）
                p1l[i, :] ^= diff[0]
                p1r[i, :] ^= diff[1]
            else:
                # 负样本：随机生成，不使用 XOR
                p1l[i, :] = np.frombuffer(urandom(2 * pairs), dtype=np.uint16)
                p1r[i, :] = np.frombuffer(urandom(2 * pairs), dtype=np.uint16)

        # 加密前再展平
        p0l_flat, p0r_flat = p0l.flatten(), p0r.flatten()
        p1l_flat, p1r_flat = p1l.flatten(), p1r.flatten()

        # 5. 密钥生成：每组 sample 的 pairs 共享同一个 key
        keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, n)
        keys_ext = np.repeat(keys, pairs, axis=1)
        ks = expand_key(keys_ext, nr)

        c0l, c0r = encrypt((p0l_flat, p0r_flat), ks)
        c1l, c1r = encrypt((p1l_flat, p1r_flat), ks)

        # 6. 计算增强输入 y_{r-1} (Speck 轮函数回退)
        # y_{r-1} = (y_r >>> 2) ^ x_r
        r0 = ror(c0r, 2) ^ c0l
        r1 = ror(c1r, 2) ^ c1l

        # -------------------------------
        # 6. 转换为二进制并强制对齐
        # -------------------------------
        # 这里的 convert_to_binary 应该返回 6 个长度为 n*pairs 的 bit 数组
        raw = convert_to_binary([c0l, c0r, c1l, c1r, r0, r1])

        # 将列表显式转为 numpy 数组，形状应为 (6, n_total, 16)
        X = np.array(raw)

        # 🚨 解决报错的关键：如果维度只有2维，手动 reshape 回去
        n_total = n * pairs
        if X.ndim == 2:
            # 假设 raw 里面是 [6][n_total*16]，将其还原
            X = X.reshape(6, n_total, 16)
        elif X.ndim == 3 and X.shape[0] != 6:
            # 防止某些实现返回的是 (n_total, 6, 16)
            pass

            # 执行转置：(6, n_total, 16) -> (n_total, 6, 16)
        # 这样每个样本的 6 个字就连续排列在了一起
        X = X.transpose((1, 0, 2))

        # 进一步拆分出 pairs：(n, pairs, 6, 16)
        X = X.reshape(n, pairs, 6, 16)

        # 最后展平为模型需要的 (n, 192)
        # 192 = 2_pairs * 6_words * 16_bits

        # 验证第一个样本的第一个对的差分
        # 如果是对的，test_diff 应该只在 0x0040 对应的位置是 1
        sample_0 = X[Y == 1][0].reshape(pairs, 6, 16)
        # 检查 c0l 和 c1l 的异或（第 0 个字和第 2 个字）
        test_diff = (sample_0[0, 0, :] != sample_0[0, 2, :]).astype(int)
        print(f"DEBUG: Fixed Diff Check for Sample 0: {test_diff}")

        return X, Y

    def make_real_differences_data(
        self,
        n: int,
        nr: int,
        pairs: int = 2,
        diff=(0x0040, 0),
    ):
        """
        用于测试泛化能力（real differences）
        """
        return real_differences_data(
            n=n,
            nr=nr,
            pairs=pairs,
            diff=diff
        )

    # ========= 加密接口（调试 / 攻击阶段常用） =========

    def encrypt(self, plaintext_pair, round_keys):
        return encrypt(plaintext_pair, round_keys)

    def decrypt(self, ciphertext_pair, round_keys):
        return decrypt(ciphertext_pair, round_keys)

    def expand_key(self, key_words, nr):
        return expand_key(key_words, nr)


if __name__ == "__main__":
    num_rounds = 5
    X, Y = make_train_data(4, num_rounds)