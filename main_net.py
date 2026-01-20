import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from rectangle import *
from torch.distributions import Bernoulli

batch_size = 1000
wdir = './good_trained_nets/'
DATA_SIZE = 10 ** 6
TEST_DATA_SIZE = 10 ** 5
RL_DATA_SIZE = 10 ** 4
NUM_ROUNDS = 10
RL_NUM_ROUNDS = 20
NUM_EPOCHS = 10
PAIRS = 2
DEPTH = 5
E_rl = 300  # 一个折中值，通常比较稳妥
ACTIVE_NIBBLE_DIFF = torch.tensor([1, 1, 1, 1])  # 0b1111

RL_CONFIG = {
    "speck": {
        "episodes": 50,
        "max_active_nibbles": 1,  # SPECK: 1 nibble ≈ 4 bits
        "lambda_sparse": 0.8,  # 强烈惩罚 HW
        "sparsity_level": "bit",
    },
    "rectangle": {
        "episodes": 120,
        "max_active_nibbles": 2,  # RECTANGLE: 1~2 S-box
        "lambda_sparse": 0.3,
        "sparsity_level": "nibble",
    }
}


def cyclic_lr(num_epochs, high_lr, low_lr):
    def res(i): return low_lr + ((num_epochs - 1) - i %
                                 num_epochs) / (num_epochs - 1) * (high_lr - low_lr)

    return res


class ResNetCrypt(nn.Module):
    def __init__(self, pairs=2, num_blocks=6, num_filters=32, num_outputs=1,
                 d1=64, d2=64, word_size=16, ks=3, depth=5, reg_param=0.0001,
                 final_activation='sigmoid'):
        super().__init__()

        self.final_activation = final_activation
        self.pairs = pairs
        self.num_blocks = num_blocks
        self.word_size = word_size

        in_channels = pairs * 6  # 和 reshape 对齐
        self.conv1 = nn.Conv1d(in_channels, num_filters, kernel_size=1, padding="same")
        self.conv3 = nn.Conv1d(in_channels, num_filters, kernel_size=3, padding="same")
        self.conv5 = nn.Conv1d(in_channels, num_filters, kernel_size=5, padding="same")
        self.conv7 = nn.Conv1d(in_channels, num_filters, kernel_size=7, padding="same")

        self.bn0 = nn.BatchNorm1d(num_filters * 4)
        # ----------------------------
        # Residual blocks
        # ----------------------------
        self.res_blocks = nn.ModuleList()
        k = ks
        for i in range(depth):
            block = nn.ModuleDict({
                "conv1": nn.Conv1d(num_filters * 4, num_filters * 4, kernel_size=k, padding="same"),
                "bn1": nn.BatchNorm1d(num_filters * 4),
                "conv2": nn.Conv1d(num_filters * 4, num_filters * 4, kernel_size=k, padding="same"),
                "bn2": nn.BatchNorm1d(num_filters * 4)
            })
            self.res_blocks.append(block)
            k += 2

        # ----------------------------
        # MLP head
        # ----------------------------
        flatten_dim = (num_filters * 4) * word_size  # 128 * 16 = 2048
        self.fc1 = nn.Linear(flatten_dim, 512)
        self.fc2 = nn.Linear(512, d1)
        self.fc3 = nn.Linear(d1, d2)
        self.fc4 = nn.Linear(d2, 1)

        self.dropout = nn.Dropout(0.8)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(d1)
        self.bn3 = nn.BatchNorm1d(d2)

    def forward(self, x):
        # x: (N, P, W, B) = (batch, pairs, num_blocks, word_size)
        N, P, W, B = x.shape

        # (N, P, W, B) → (N, P, B, W)
        x = x.permute(0, 1, 3, 2)

        # 👉 合并 pairs 和 words 作为 channel（关键）
        # (N, P, B, W) → (N, P*B, W)
        x = x.reshape(N, P * W, B)

        # Inception
        c1 = self.conv1(x)
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        c7 = self.conv7(x)
        x = torch.cat([c1, c3, c5, c7], dim=1)
        x = F.relu(self.bn0(x))

        if not hasattr(self, "_debug_done"):
            print("After inception:", x.shape)
            self._debug_done = True

        # Residual blocks（不变）
        for block in self.res_blocks:
            identity = x
            out = F.relu(block["bn1"](block["conv1"](x)))
            out = F.relu(block["bn2"](block["conv2"](out)))
            x = identity + out

        # flatten（和 Keras 一样）
        x = x.flatten(start_dim=1)

        # MLP
        x = self.dropout(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)

        return x


class NibblePolicy(nn.Module):
    """
    Policy: 输出 16 个 nibble 是否激活
    """

    def __init__(self, num_nibbles=16, max_active_nibbles=4):
        super().__init__()
        self.num_nibbles = num_nibbles
        self.max_active_nibbles = max_active_nibbles
        self.logits = nn.Parameter(torch.zeros(num_nibbles))  # 初始 p=0.5

    def sample(self):
        probs = torch.sigmoid(self.logits)
        dist = torch.distributions.Bernoulli(probs)
        mask = dist.sample().float()  # (16,)
        # Top-k 限制
        if mask.sum() > self.max_active_nibbles:
            topk_idx = torch.topk(probs, self.max_active_nibbles).indices
            mask.zero_()
            mask[topk_idx] = 1.0
        log_prob = dist.log_prob(mask).sum()
        return mask, log_prob

    def get_prob(self):
        return torch.sigmoid(self.logits)


class BitPolicy(nn.Module):
    """
    Bit-level policy for SPECK with fixed Hamming weight = 1
    """

    def __init__(self, num_bits=32):
        super().__init__()
        self.num_bits = num_bits
        self.logits = nn.Parameter(torch.zeros(num_bits))

    def sample(self):
        # 使用 softmax 保证只选一个 bit
        probs = torch.softmax(self.logits, dim=0)  # (num_bits,)
        dist = torch.distributions.Categorical(probs)
        idx = dist.sample()  # 输出索引 (单个整数)

        mask = torch.zeros(self.num_bits, device=self.logits.device)
        mask[idx] = 1.0  # HW=1，只有这个位置为1

        log_prob = dist.log_prob(idx)  # policy gradient

        return mask, log_prob

    def get_prob(self):
        return torch.softmax(self.logits, dim=0)


# ----------------------------
# mask → diff 通用函数
# ----------------------------
def mask_to_bits(nibble_mask, cipher):
    """
    将 RL agent 输出的 nibble mask 转成对应 cipher 的 bit 差分
    """
    num_bits = cipher.state_bits()  # cipher 自己提供状态总比特数
    diff_bits = torch.zeros(num_bits, device=nibble_mask.device)

    bits_per_nibble = num_bits // len(nibble_mask)
    for i in range(len(nibble_mask)):
        if nibble_mask[i] > 0:
            diff_bits[i * bits_per_nibble:(i + 1) * bits_per_nibble] = 1  # 或自定义 ACTIVE_NIBBLE_DIFF

    return diff_bits


def nibble_mask_to_diff_matrix(cipher, bits):
    if isinstance(bits, torch.Tensor):
        bits = bits.detach().cpu().numpy()

    bits = bits.astype(np.uint8).flatten()  # flatten 确保是 1D

    num_words = cipher.diff_words()  # 2 for SPECK-32
    word_size = cipher.word_size()  # 16
    nibbles_per_word = word_size // 4  # 4

    if len(bits) != num_words * nibbles_per_word:
        raise ValueError(f"bits length {len(bits)} != expected {num_words * nibbles_per_word}")

    # 每个 nibble 扩展成 4 位
    diff_bits = np.repeat(bits, 4)  # 每个 nibble 4 位
    diff_matrix = diff_bits.reshape(num_words, word_size)
    return diff_matrix


def train_distinguisher(cipher, num_epochs, num_rounds=9, depth=1, pairs=2, num_blocks=2):
    print("pairs = ", pairs)
    print("num_rounds = ", num_rounds)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # ---------------------------- RL 获取差分 ----------------------------
    # policy = optimize_diff_with_rl(cipher=cipher, episodes=RL_NUM_ROUNDS, num_blocks=num_blocks)
    #
    # if isinstance(policy, BitPolicy):
    #     # 获取 softmax 后的概率分布
    #     probs = policy.get_prob()
    #     # 只取概率最大的那 1 个 bit
    #     top_idx = torch.argmax(probs)
    #
    #     final_diff_mask = torch.zeros_like(probs)
    #     final_diff_mask[top_idx] = 1.0
    #
    #     # 转换为 cipher 需要的格式（如果是 SPECK，通常需要转为矩阵/数组）
    #     diff = cipher.diff_bits_to_matrix(final_diff_mask)
    # else:
    #     # RECTANGLE 的逻辑保持不变
    #     final_probs = torch.sigmoid(policy.logits)
    #     topk_idx = torch.topk(final_probs, RL_CONFIG["rectangle"]["max_active_nibbles"]).indices
    #     final_nibble_mask = torch.zeros_like(final_probs)
    #     final_nibble_mask[topk_idx] = 1.0
    #     diff = nibble_mask_to_diff_matrix(cipher, final_nibble_mask)

    diff = (0x0040, 0)
    print("Difference (HW=1):", diff)

    # ---------------------------- 生成训练数据 ----------------------------
    print("Generating dataset...")
    X, Y = cipher.make_train_data(DATA_SIZE, num_rounds, pairs=pairs, diff=diff)

    X_eval, Y_eval = cipher.make_train_data(TEST_DATA_SIZE, num_rounds, pairs=pairs, diff=diff)
    print("Generating finished")

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
    X_eval = torch.tensor(X_eval, dtype=torch.float32)
    Y_eval = torch.tensor(Y_eval, dtype=torch.float32).unsqueeze(1)

    print("X min:", X.min().item(),
          "X max:", X.max().item(),
          "X mean:", X.mean().item())

    # 取 1000 个正样本 & 负样本
    pos = X[Y.squeeze() == 1][:1000]
    neg = X[Y.squeeze() == 0][:1000]

    print("pos mean:", pos.mean().item())
    print("neg mean:", neg.mean().item())
    print("mean abs diff:", (pos.mean(dim=0) - neg.mean(dim=0)).abs().mean().item())

    train_loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_eval, Y_eval), batch_size=batch_size)

    # ---------------------------- 创建模型 ----------------------------
    model = ResNetCrypt(num_blocks=num_blocks, pairs=pairs, depth=depth).to(device)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.002)
    print("weight_decay =", optimizer.param_groups[0].get("weight_decay", 0))

    lr_func = cyclic_lr(num_epochs=10, high_lr=0.001, low_lr=0.0001)
    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
    best_val_acc = 0
    best_model_path = wdir + f"best{num_rounds}r_depth{depth}_epochs{num_epochs}_pairs{pairs}.pt"

    # ---------------------------- Training loop ----------------------------
    for epoch in range(num_epochs):
        # update LR
        lr = lr_func(epoch)
        for g in optimizer.param_groups:
            g['lr'] = lr

        model.train()
        train_loss, correct, total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            l2_lambda = 1e-5
            l2_loss = torch.tensor(0., device=device)

            for name, p in model.named_parameters():
                if 'weight' in name and 'bn' not in name:
                    l2_loss += torch.sum(p ** 2)

            loss = loss + l2_lambda * l2_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            prob = torch.sigmoid(pred)
            correct += ((prob > 0.5).float() == yb).sum().item()
            total += len(xb)
        train_acc = correct / total
        train_loss /= total

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * len(xb)
                prob = torch.sigmoid(pred)
                correct += ((prob > 0.5).float() == yb).sum().item()
                total += len(xb)
        val_acc = correct / total
        val_loss /= total

        history["loss"].append(train_loss)
        history["acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch + 1}/{num_epochs} LR={lr:.6f} "
              f"loss={train_loss:.6f} acc={train_acc:.4f} "
              f"val_loss={val_loss:.6f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    final_path = wdir + f"model_{num_rounds}r_depth{depth}_epochs{num_epochs}_pairs{pairs}.pt"
    torch.save(model.state_dict(), final_path)

    with open(wdir + f"hist{num_rounds}r_depth{depth}_epochs{num_epochs}_pairs{pairs}.p", "wb") as f:
        pickle.dump(history, f)

    print("Best validation accuracy:", best_val_acc)
    return model, history


def train_distinguisher_for_rl(
        cipher,
        diff_bits,
        model,
        optimizer,
        device,
        epochs=5,
        batch_size=512
):
    """
    Train a lightweight distinguisher and return validation accuracy as reward
    """

    model.train()

    # bit → cipher-specific matrix
    diff_matrix = cipher.diff_bits_to_matrix(diff_bits)

    # generate fresh data EVERY episode
    X, Y = cipher.make_train_data(
        n=RL_DATA_SIZE,
        nr=NUM_ROUNDS,
        pairs=2,
        diff=diff_matrix
    )

    X = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)

    # shuffle labels sanity (防止泄漏)
    perm = torch.randperm(X.size(0))
    X, Y = X[perm], Y[perm]

    for _ in range(epochs):
        perm = torch.randperm(X.size(0))
        for i in range(0, X.size(0), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X[idx], Y[idx]

            optimizer.zero_grad()
            out = model(xb).squeeze()
            loss = nn.BCELoss()(out, yb)
            loss.backward()
            optimizer.step()

    # ===== validation-style reward =====
    model.eval()
    with torch.no_grad():
        pred = model(X).squeeze()
        acc = ((pred > 0.5) == Y).float().mean().item()

    # ⚠️ 关键：防止 reward 饱和
    acc = min(acc, 0.95)

    return acc


def optimize_diff_with_rl(cipher, episodes=50, lr_policy=5e-2, device='cuda', num_blocks=6):
    """
    RL to find optimal difference, HW=1 for SPECK
    """
    num_bits = cipher.state_bits()  # total bits in cipher state
    policy = BitPolicy(num_bits=num_bits).to(device)
    policy_optim = optim.Adam(policy.parameters(), lr=lr_policy)

    # fixed small distinguisher
    model = ResNetCrypt(pairs=2, num_blocks=num_blocks, word_size=cipher.word_size()).to(device)
    model_optim = optim.Adam(model.parameters(), lr=1e-3)

    baseline = 0.0
    lambda_sparse = 0.0  # HW already fixed, no additional penalty

    for episode in range(episodes):
        # 1️⃣ sample mask, HW=1
        mask, log_prob = policy.sample()
        diff_bits = mask  # already bit-level

        # 2️⃣ train small distinguisher and get reward
        reward = train_distinguisher_for_rl(
            cipher=cipher,
            diff_bits=diff_bits,
            model=model,
            optimizer=model_optim,
            device=device
        )

        # 3️⃣ policy gradient update
        advantage = reward - baseline
        baseline = 0.9 * baseline + 0.1 * reward

        loss = -log_prob * advantage
        policy_optim.zero_grad()
        loss.backward()
        policy_optim.step()

        if episode % 10 == 0:
            print(f"[{episode:03d}] reward={reward:.4f}, mean_p={policy.get_prob().mean().item():.3f}")

    return policy
