import numpy as np
import os
from typing import Tuple
import torch

BIT_PER_ROW = 16
STATE_ROW = 4
KEY_ROW = 5

# RECTANGLE中明密文结构统一都是4*16bit
state = np.zeros((4, 16), dtype=np.uint8)
# 80bit的密钥结构
key_80 = np.zeros((5, 16), dtype=np.uint8)

# RECTANGLE的S盒
S_BOX = np.array([
    0x6, 0x5, 0xC, 0xA, 0x1, 0xE, 0x7, 0x9,
    0xB, 0x0, 0x3, 0xD, 0x8, 0xF, 0x4, 0x2
], dtype=np.uint8)

ROUND_CONSTANTS = np.array([
    0x01, 0x02, 0x04, 0x09, 0x12, 0x05, 0x0B, 0x16,
    0x0C, 0x19, 0x13, 0x07, 0x0F, 0x1F, 0x1E, 0x1C,
    0x18, 0x11, 0x03, 0x06, 0x0D, 0x1B, 0x17, 0x0E,
    0x1D
], dtype=np.uint8)


def bits_to_nibble_5bit(bits: np.ndarray) -> int:
    """将 5 个比特的数组（Row0 LSB, Row4 MSB）打包成一个整数。"""
    # 假设 bits[i] 对应权重 2^i
    key_int_5 = 0
    for i in range(5):
        key_int_5 += bits[4 - i] * (1 << i)  # bits[0] = κ0,4 → MSB，bits[4] = κ0,0 → LSB
    return int(key_int_5)


# S盒替换步骤
def rectangle_sbox_substitution(state: np.ndarray) -> np.ndarray:
    """
    对一个 4x16 的 NumPy 数组（存储比特）进行 RECTANGLE S 盒替换。
    严格按照论文 SubColumn 定义，每列单独替换。
    state[0, j] 是 a0,j (MSB)，state[3, j] 是 a3,j (LSB)
    """
    if state.shape != (4, 16) or state.dtype != np.uint8:
        raise ValueError("Input state must be a (4, 16) numpy array of dtype np.uint8.")

    state_out = np.zeros_like(state, dtype=np.uint8)

    for j in range(16):
        # 取每列的 4-bit，按照论文顺序 a3||a2||a1||a0
        col_bits = state[:, j]  # state[0..3, j]

        # 按论文定义打包：MSB = a3 (state[3]), LSB = a0 (state[0])
        nibble_in = (col_bits[3] << 3) | (col_bits[2] << 2) | (col_bits[1] << 1) | col_bits[0]

        # S-box 替换
        nibble_out = S_BOX[nibble_in]

        # 解包回 4-bit，严格对应 b3||b2||b1||b0
        state_out[3, j] = (nibble_out >> 3) & 1  # b3
        state_out[2, j] = (nibble_out >> 2) & 1  # b2
        state_out[1, j] = (nibble_out >> 1) & 1  # b1
        state_out[0, j] = nibble_out & 1  # b0

    return state_out


def add_round_key(state: np.ndarray, key_80: np.ndarray) -> np.ndarray:
    if state.shape != (4, 16) or key_80.shape != (5, 16):
        raise ValueError("State must be (4, 16) and key_80 must be (5, 16).")
    round_key_Ki = key_80[0:4, :]
    new_state = state ^ round_key_Ki

    return new_state


def shift_row(state: np.ndarray) -> np.ndarray:
    if state.shape != (4, 16):
        raise ValueError("Input state must be a (4, 16) numpy array.")

    new_state = state.copy()

    shifts = [0, 1, 12, 13]

    ROW_SIZE = 16

    for i in range(4):
        offset = shifts[i]
        row = state[i, :]

        # 循环左移实现
        new_state[i, :] = np.concatenate((row[offset:], row[:offset]))

    return new_state


def inverse_shift_row(state: np.ndarray) -> np.ndarray:
    if state.shape != (4, 16):
        raise ValueError("Input state must be a (4, 16) numpy array.")

    new_state = state.copy()

    shifts = [0, 1, 12, 13]   # 原 SR 的左移量
    ROW_SIZE = 16

    for i in range(4):
        offset = shifts[i]
        row = state[i, :]

        # 左移 offset 的逆操作 = 右移 offset = 左移 (16 - offset)
        inv_offset = (ROW_SIZE - offset) % ROW_SIZE

        new_state[i, :] = np.concatenate((row[inv_offset:], row[:inv_offset]))

    return new_state


def rotate_left_16bit(row, offset):
    offset = offset % 16
    if offset == 0:
        rotated = row.copy()  # 如果偏移为0，也可以复制一份
    else:
        rotated = np.concatenate((row[offset:], row[:offset]))
    return rotated


def key_schedule_update(key_80: np.ndarray, round_i: int) -> np.ndarray:
    """
    更新 RECTANGLE 密钥寄存器：
    1. 对右上角 4x4 比特块应用 S-box（每列替换）；
    2. 进行 1-round 广义 Feistel 变换；
    3. 5-bit 轮常数异或。
    """
    if key_80.shape != (5, 16) or not 0 <= round_i <= 24:
        raise ValueError("Invalid input dimensions or round index.")

    key_next_round = key_80.copy()

    # =====================================================
    # 1. S-box 应用：右上角 4x4 块 (Row0-3, Col12-15)
    # =====================================================
    for j in range(12, 16):  # 右上角 4 列
        col_bits = key_next_round[0:4, j]  # Row0..3
        # 按论文定义打包 a3||a2||a1||a0
        nibble_in = (col_bits[3] << 3) | (col_bits[2] << 2) | (col_bits[1] << 1) | col_bits[0]
        # S-box 替换
        nibble_out = S_BOX[nibble_in]
        # 解包回 4-bit，b3||b2||b1||b0
        key_next_round[3, j] = (nibble_out >> 3) & 1
        key_next_round[2, j] = (nibble_out >> 2) & 1
        key_next_round[1, j] = (nibble_out >> 1) & 1
        key_next_round[0, j] = nibble_out & 1

    # =====================================================
    # 2. 1-round 广义 Feistel 变换
    # =====================================================
    Row0 = key_next_round[0, :].copy()
    Row1 = key_next_round[1, :].copy()
    Row2 = key_next_round[2, :].copy()
    Row3 = key_next_round[3, :].copy()
    Row4 = key_next_round[4, :].copy()

    temp_next = np.zeros_like(key_next_round, dtype=np.uint8)
    temp_next[0, :] = rotate_left_16bit(Row0, 8) ^ Row1
    temp_next[1, :] = Row2
    temp_next[2, :] = Row3
    temp_next[3, :] = rotate_left_16bit(Row3, 12) ^ Row4
    temp_next[4, :] = Row0

    key_next_round = temp_next

    # =====================================================
    # 3. 5-bit 轮常数异或 (Row0, Col0-4)
    # =====================================================
    rc_i = ROUND_CONSTANTS[round_i]

    key_bits_5 = key_next_round[0, -5:].copy()  # Col11~15 或根据你列索引调整
    key_int_5 = bits_to_nibble_5bit(key_bits_5)
    result_int = key_int_5 ^ rc_i

    key_next_round[0, -5] = (result_int >> 4) & 1  # κ0,4
    key_next_round[0, -4] = (result_int >> 3) & 1  # κ0,3
    key_next_round[0, -3] = (result_int >> 2) & 1  # κ0,2
    key_next_round[0, -2] = (result_int >> 1) & 1  # κ0,1
    key_next_round[0, -1] = result_int & 1  # κ0,0

    return key_next_round


def rectangle_encrypt(plaintext_state: np.ndarray, master_key_80: np.ndarray, nr) -> np.ndarray:
    # 初始检查
    if plaintext_state.shape != (4, 16) or master_key_80.shape != (5, 16):
        raise ValueError("State must be (4, 16) and Key must be (5, 16).")

    STATE = plaintext_state.copy()
    key_register = master_key_80.copy()

    # GenerateRoundKeys() 是一个概念性的操作，它在每次迭代中更新 key_register

    # 循环 25 轮 (i = 0 到 24)
    for i in range(nr):
        # 1. AddRoundKey(STATE, Ki)
        # 提取 Ki (key_register 的前 4 行) 并进行异或
        STATE = add_round_key(STATE, key_register)

        # 2. SubColumn(STATE)
        STATE = rectangle_sbox_substitution(STATE)

        # 3. ShiftRow(STATE)
        STATE = shift_row(STATE)

        # 4. GenerateRoundKeys() - 更新密钥寄存器以供下一轮使用
        # 这一步生成 K_{i+1}
        # 注意：对于 i=24 这一轮，我们生成 K_25
        key_register = key_schedule_update(key_register, i)

    STATE = add_round_key(STATE, key_register)

    return STATE


def bin_to_matrix(bin_str: str) -> np.ndarray:
    """
    将二进制字符串每 16 个比特作为一行，转换为二维 NumPy 数组。
    """
    if len(bin_str) % 16 != 0:
        raise ValueError("Binary string length must be a multiple of 16.")

    rows = len(bin_str) // 16
    cols = 16

    matrix = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        block = bin_str[i * 16:(i + 1) * 16]  # 当前 16-bit 分块
        for j in range(16):
            matrix[i, j] = int(block[j])

    return matrix


def matrix_to_bin(matrix: np.ndarray) -> str:
    """
    修正后的 I/O 转换：将 NumPy 比特矩阵转换为二进制字符串。
    (输出顺序与 bin_to_matrix 的输入格式一致)
    """
    rows, cols = matrix.shape
    bin_parts = []

    # 按照 Row rows-1 (MS Row) 到 Row 0 (LS Row) 的顺序提取
    for i in range(rows):
        # 矩阵行索引: i=0 对应最高行 (Row 3/4)
        matrix_row_index = rows - 1 - i

        # 提取 Row i
        row = matrix[matrix_row_index, :]

        # 按照 Col 15 到 Col 0 的顺序提取
        row_bits = []
        for j in range(cols):
            # j=0 对应 Col 15
            matrix_col_index = cols - 1 - j
            bit = str(row[matrix_col_index])
            row_bits.append(bit)

        bin_parts.append("".join(row_bits))

    return "".join(bin_parts)


def diff_bits_to_matrix(self, diff_bits: torch.Tensor) -> np.ndarray:
    """
    将 64-bit 的差分向量转换为 4x16 的矩阵
    diff_bits: torch.Tensor, shape=(64,)
    return: np.ndarray, shape=(4,16), dtype=np.uint8
    """
    # 确保是 numpy 数组
    diff_bits = diff_bits.detach().cpu().numpy().astype(np.uint8)

    # reshape 成 4 行 16 列
    diff_matrix = diff_bits.reshape(self.state_rows, self.bit_per_row)
    return diff_matrix


def random_bitstring(num_bits):
    """生成 num_bits 位的随机二进制字符串,对应于随机生成明文和密钥"""
    num_bytes = (num_bits + 7) // 8  # 向上取整为字节数
    random_bytes = os.urandom(num_bytes)
    random_int = int.from_bytes(random_bytes, "big")
    bitstring = bin(random_int)[2:].zfill(num_bits)  # 转为 bit 字符串并补齐位数
    return bitstring[:num_bits]  # 多余位截掉（避免 num_bits 不为 8 的倍数）


def make_train_data(n, nr, pairs, diff):
    Y = np.frombuffer(os.urandom(n), dtype=np.uint8)
    Y = Y & 1
    # Y1 = np.tile(Y, pairs)
    # 最终数据集: X[i] = 第 i 条样本 = (pairs, 10, 16)
    # 每条样本是两条密文以及中间差分状态拼接后的结果，所以是12
    X = np.zeros((n, pairs, 12, 16), dtype=np.uint8)
    # diff = bin_to_matrix(diff)
    # 按照样本数量循环
    for i in range(n):
        for p in range(pairs):

            label = Y[i]  # 当前样本的标签

            # ===============================
            # 负样本：随机明文 → 随机密文
            # ===============================
            if label == 0:
                plain0 = random_bitstring(STATE_ROW * BIT_PER_ROW)
                plain0 = bin_to_matrix(plain0)
                master_key0 = random_bitstring(KEY_ROW * BIT_PER_ROW)
                master_key0 = bin_to_matrix(master_key0)

                plain1 = random_bitstring(STATE_ROW * BIT_PER_ROW)
                plain1 = bin_to_matrix(plain1)
                # TODO 考虑相关密钥差分问题
                # master_key1 = random_bitstring(KEY_ROW * BIT_PER_ROW)
                # master_key1 = bin_to_matrix(master_key1)

                cipher0 = rectangle_encrypt(plain0, master_key0, nr)
                cipher1 = rectangle_encrypt(plain1, master_key0, nr)
            #    添加中间解密状态（两密文异或抵消轮密钥的影响，然后再做一轮逆SR）
                cipher_diff = cipher0 ^ cipher1
                cipher_diff = inverse_shift_row(cipher_diff)

            # ===============================
            # 正样本：明文差分 → 密文两条相关
            # ===============================
            else:
                plain0 = random_bitstring(STATE_ROW * BIT_PER_ROW)
                plain0 = bin_to_matrix(plain0)
                master_key0 = random_bitstring(KEY_ROW * BIT_PER_ROW)
                master_key0 = bin_to_matrix(master_key0)

                plain1 = plain0 ^ diff
                master_key1 = random_bitstring(KEY_ROW * BIT_PER_ROW)
                master_key1 = bin_to_matrix(master_key1)

                cipher0 = rectangle_encrypt(plain0, master_key0, nr)
                cipher1 = rectangle_encrypt(plain1, master_key1, nr)

                cipher_diff = cipher0 ^ cipher1
                cipher_diff = inverse_shift_row(cipher_diff)

            # ===============================
            # 拼接 C0 和 C1 → 10 × 16
            # ===============================
            merged = np.concatenate([cipher_diff, cipher0, cipher1], axis=0)  # (10,16)
            X[i, p] = merged

    return X, Y


def state_shape():
    """
    返回神经网络期望的单样本 shape（不含 batch）
    """
    # 你现在的 X[i] = (pairs, 12, 16)
    return 12, 16


class RectangleCipher:
    """
    RECTANGLE-80 cipher wrapper for neural cryptanalysis
    """

    def __init__(self):
        self.state_rows = STATE_ROW
        self.bit_per_row = BIT_PER_ROW
        self.key_rows = KEY_ROW
        self.block_bits = STATE_ROW * BIT_PER_ROW  # 64
        self.key_bits = KEY_ROW * BIT_PER_ROW      # 80
        self.name = "rectangle"

    # ===== 网络 & 数据相关的统一接口 =====

    def name(self):
        return "rectangle"

        # ---- 实现 Cipher 接口 ----

    def state_bits(self) -> int:
        return self.block_bits

    def diff_words(self) -> int:
        """
        RECTANGLE-64: 4 words
        """
        return 4

    def diff_bits_to_matrix(self, diff_bits: torch.Tensor) -> np.ndarray:
        # diff_bits shape: (64,)
        return diff_bits.cpu().numpy().reshape(self.state_rows, self.bit_per_row)

    def num_nibbles(self) -> int:
        return self.state_rows * (self.bit_per_row // 4)  # 16 个 nibble

    def max_active_nibbles(self) -> int:
        return 4  # 可以自定义

    def word_size(self) -> int:
        return self.bit_per_row  # 16 bits

    def block_size(self):
        return self.block_bits

    def make_train_data(
        self,
        n: int,
        nr: int,
        pairs: int,
        diff: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对外统一接口，内部直接复用你写好的函数
        """
        return make_train_data(n, nr, pairs, diff)

    # ===== 加密接口（如果之后做验证/攻击会很有用） =====

    def encrypt(self, plaintext_state: np.ndarray, master_key: np.ndarray, nr: int):
        return rectangle_encrypt(plaintext_state, master_key, nr)


def testCryptography():
    # REC-80 Test Vector 1 (全零)
    KEY_1_BIN = "00000000000000000000000000000000000000000000000000000000000000000000000000000000"  # 80 bits
    PT_1_BIN = "0000000000000000000000000000000000000000000000000000000000000000"  # 64 bits
    CT_1_BIN = "0010110110010110111000110101010011101000101100010000100001110100"  # 64 bits

    # REC-80 Test Vector 2 (全一)
    KEY_2_BIN = "11111111111111111111111111111111111111111111111111111111111111111111111111111111"  # 80 bits
    PT_2_BIN = "1111111111111111111111111111111111111111111111111111111111111111"  # 64 bits
    CT_2_BIN = "1001100101000101101010100011010010101110001111010000000100010010"  # 64 bits

    """
        使用官方测试向量验证 RECTANGLE-80 加密算法。
        """
    print("--- 启动 RECTANGLE 官方测试向量验证 ---")

    test_vectors = [
        {"key": KEY_1_BIN, "pt": PT_1_BIN, "ct": CT_1_BIN, "name": "全零测试"},
        {"key": KEY_2_BIN, "pt": PT_2_BIN, "ct": CT_2_BIN, "name": "全一测试"},
    ]

    all_passed = True

    for idx, tv in enumerate(test_vectors):
        print(f"\n--- 运行测试向量 #{idx + 1}: {tv['name']} ---")

        # 1. 转换输入
        master_key_80 = bin_to_matrix(tv['key'])
        plaintext_state = bin_to_matrix(tv['pt'])
        expected_ciphertext_matrix = bin_to_matrix(tv['ct'])

        # 2. 运行算法
        try:
            actual_ciphertext_matrix = rectangle_encrypt(plaintext_state, master_key_80, 25)

            # 3. 转换输出为字符串进行比较和显示
            actual_ciphertext_bin = matrix_to_bin(actual_ciphertext_matrix)

            # 4. 验证结果
            is_correct = np.array_equal(actual_ciphertext_matrix, expected_ciphertext_matrix)

            if is_correct:
                print(f"✅ 测试通过: {tv['name']}")
            else:
                all_passed = False
                print(f"❌ 测试失败: {tv['name']}")
                print(f"预期密文 (BIN): {tv['ct']}")
                print(f"实际密文 (BIN): {actual_ciphertext_bin}")
                # print(f"预期密文 (Matrix):\n{expected_ciphertext_matrix}")
                # print(f"实际密文 (Matrix):\n{actual_ciphertext_matrix}")

        except Exception as e:
            all_passed = False
            print(f"❌ 算法执行出错: {e}")

    print("\n-------------------------------------------")
    if all_passed:
        print("🎉 **恭喜! 所有官方测试向量验证通过。**")
    else:
        print("🐛 **注意: 某些测试失败。请检查密钥调度和轮函数实现逻辑。**")


if __name__ == "__main__":
    diff = random_bitstring(STATE_ROW * BIT_PER_ROW)
    make_train_data(4, 3, 2, diff)
