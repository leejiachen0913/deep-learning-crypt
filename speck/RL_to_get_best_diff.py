import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import speck as sp
from Main_Model_multiple_parallel_convolutional_layers_net import *


# ============================================================
# PPO Policy：Bernoulli Bit Vector
# ============================================================
class BitPolicyPPO(tf.keras.Model):
    def __init__(self, num_bits=32):
        super().__init__()
        self.num_bits = num_bits
        self.logits = self.add_weight(
            name="logits",
            shape=(num_bits,),
            initializer=tf.keras.initializers.RandomNormal(0.0, 0.5),
            trainable=True
        )

    def dist(self):
        probs = tf.nn.sigmoid(self.logits)
        return tfp.distributions.Bernoulli(probs=probs)

    def sample(self):
        d = self.dist()
        action = d.sample()
        log_prob = tf.reduce_sum(d.log_prob(action))
        return action, log_prob

    def log_prob(self, action):
        d = self.dist()
        return tf.reduce_sum(d.log_prob(action))

    def entropy(self):
        p = tf.nn.sigmoid(self.logits)
        return -tf.reduce_sum(
            p * tf.math.log(p + 1e-8) +
            (1 - p) * tf.math.log(1 - p + 1e-8)
        )


# ============================================================
# Utils
# ============================================================
def bits_to_uint16(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def get_diff_hex_from_probs(probs):
    bits = (probs > 0.5).astype(np.uint8)
    l_val, r_val = 0, 0
    for i in range(16):
        l_val = (l_val << 1) | int(bits[i])
        r_val = (r_val << 1) | int(bits[i + 16])
    return l_val, r_val


def popcount16(x: int) -> int:
    return int(bin(x & 0xFFFF).count("1"))


# reward cache（同一 diff 多次评估直接复用）
reward_cache = {}


# ============================================================
# Reward A: Train-input-space z-score (train-aligned, no NN)
# ============================================================
def compute_reward_linear_acc(policy_diff_bits, num_rounds=6, pairs=2,
                              n_train=65536, n_eval=16384,
                              steps=300, batch_size=1024,
                              repeats=2,
                              l2=1e-4,
                              hw_lambda=0.003,
                              seed_base=2026):
    diff_l = bits_to_uint16(policy_diff_bits[:16])
    diff_r = bits_to_uint16(policy_diff_bits[16:])
    if diff_l == 0 and diff_r == 0:
        return -5.0

    hw = popcount16(diff_l) + popcount16(diff_r)

    # 缓存：同一 diff 直接复用
    key = ("linacc", int(num_rounds), int(diff_l), int(diff_r), int(pairs),
           int(n_train), int(n_eval), int(steps), int(batch_size), int(repeats))
    if key in reward_cache:
        return reward_cache[key]

    # 注意：make_train_data 内部用 urandom，无法完全固定随机性
    # 所以用 repeats 做平均降低噪声
    accs = []
    for rep in range(repeats):
        tf.keras.backend.clear_session()

        X, Y   = sp.make_train_data(n_train, num_rounds, diff=(diff_l, diff_r), pairs=pairs)
        Xv, Yv = sp.make_train_data(n_eval,  num_rounds, diff=(diff_l, diff_r), pairs=pairs)

        # 线性 probe：单层
        inp = tf.keras.Input(shape=(X.shape[1],))
        out = tf.keras.layers.Dense(
            1, activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.l2(l2),
            dtype="float32"
        )(inp)
        model = tf.keras.Model(inp, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="binary_crossentropy",
                      metrics=["accuracy"])

        train_ds = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(8192).batch(batch_size).repeat()
        val_ds   = tf.data.Dataset.from_tensor_slices((Xv, Yv)).batch(batch_size)

        h = model.fit(train_ds, steps_per_epoch=steps, epochs=1, validation_data=val_ds, verbose=0)
        accs.append(float(h.history["val_accuracy"][-1]))

    acc = float(np.mean(accs))

    # reward：线性 probe 的 acc，才是你最终训练最关心的东西
    reward = (acc - 0.5) * 4000.0 - hw_lambda * (hw ** 2)
    reward_cache[key] = float(reward)
    return float(reward)


# ============================================================
# Reward B: Short-train val_accuracy (most aligned, slowest)
# ============================================================
def compute_reward_train_acc(policy_diff_bits, num_rounds=6, pairs=2,
                             n_train=20000, n_eval=5000,
                             steps=200, batch_size=512,
                             repeats=2,
                             hw_lambda=0.002):
    """
    直接用 train 模式网络做一个“短训练”得到 val_accuracy 当 reward。
    最对齐最终目标，但会慢很多。
    """
    diff_l = bits_to_uint16(policy_diff_bits[:16])
    diff_r = bits_to_uint16(policy_diff_bits[16:])
    if diff_l == 0 and diff_r == 0:
        return -5.0

    hw = popcount16(diff_l) + popcount16(diff_r)
    accs = []

    for _ in range(repeats):
        tf.keras.backend.clear_session()

        X, Y = sp.make_train_data(n_train, num_rounds, diff=(diff_l, diff_r), pairs=pairs)
        Xv, Yv = sp.make_train_data(n_eval,  num_rounds, diff=(diff_l, diff_r), pairs=pairs)

        net = make_resnet(mode='train', pairs=pairs, depth=2, num_blocks=3, reg_param=1e-5)
        net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        train_ds = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(4096).batch(batch_size).repeat()
        val_ds = tf.data.Dataset.from_tensor_slices((Xv, Yv)).batch(batch_size)

        h = net.fit(train_ds, steps_per_epoch=steps, epochs=1, validation_data=val_ds, verbose=0)
        accs.append(float(h.history["val_accuracy"][-1]))

    acc = float(np.mean(accs))
    reward = float((acc - 0.5) * 2000.0 - hw_lambda * (hw ** 2))
    return reward


# ============================================================
# Rollout Collection
# ============================================================
def collect_rollout(policy, num_samples, num_rounds, pairs, reward_mode="trainX"):
    actions = []
    log_probs = []
    raw_rewards = []

    for _ in range(num_samples):
        a, logp = policy.sample()
        a_np = a.numpy().astype(int)


        r = compute_reward_train_acc(a_np, num_rounds=num_rounds, pairs=pairs)

        actions.append(a)
        log_probs.append(logp)
        raw_rewards.append(r)

    raw_rewards = np.array(raw_rewards, dtype=np.float32)
    rewards = (raw_rewards - raw_rewards.mean()) / (raw_rewards.std() + 1e-8)
    return actions, log_probs, rewards, raw_rewards


# ============================================================
# PPO Update Step
# ============================================================
def ppo_update(policy, optimizer,
               actions, old_log_probs, rewards,
               clip_ratio=0.1,
               entropy_beta=0.015,
               train_iters=3):
    old_log_probs = tf.stack(old_log_probs)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)

    for _ in range(train_iters):
        with tf.GradientTape() as tape:
            new_log_probs = tf.stack([policy.log_prob(a) for a in actions])

            ratio = tf.exp(new_log_probs - old_log_probs)
            clipped = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
            surrogate = tf.minimum(ratio * rewards, clipped * rewards)

            policy_loss = -tf.reduce_mean(surrogate)
            entropy_loss = -entropy_beta * policy.entropy()
            loss = policy_loss + entropy_loss

        grads = tape.gradient(loss, policy.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, policy.trainable_variables))


# ============================================================
# Strong Eval (train-mode, final check)
# ============================================================
def strong_eval_train(diff_l, diff_r, num_rounds=6, pairs=2, repeats=3,
                      epochs=15, depth=5, num_blocks=3):
    accs = []
    for _ in range(repeats):
        n_train = 20000
        n_eval = 4000
        X, Y = sp.make_train_data(n_train, num_rounds, diff=(diff_l, diff_r), pairs=pairs)
        Xv, Yv = sp.make_train_data(n_eval, num_rounds, diff=(diff_l, diff_r), pairs=pairs)

        net = make_resnet(mode='train', pairs=pairs, depth=depth, num_blocks=num_blocks, reg_param=1e-5)
        net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        h = net.fit(X, Y, epochs=epochs, batch_size=512, validation_data=(Xv, Yv), verbose=0)
        accs.append(float(np.max(h.history["val_accuracy"])))
    return accs, float(np.mean(accs)), float(np.std(accs))


# ============================================================
# Training Loop
# ============================================================
if __name__ == "__main__":
    NUM_BITS = 32
    NUM_ROUNDS = 6
    PAIRS = 2

    EPISODES = 200

    # trainX 很快，可以 8~16；acc 很慢，建议 2~6
    SAMPLES_PER_EP = 12

    # "trainX"：推荐（对齐训练且更快）
    # "acc"：最对齐但慢
    REWARD_MODE = "trainX"

    policy = BitPolicyPPO(num_bits=NUM_BITS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)

    best_prob_std = 0.0
    curr_l = 0
    curr_r = 0

    print("=" * 70)
    print(f"🚀 PPO RewardMode={REWARD_MODE} | Speck32/{NUM_ROUNDS}r | pairs={PAIRS}")
    print("=" * 70)

    for ep in range(EPISODES):
        actions, logps, rewards, raw_rewards = collect_rollout(
            policy, SAMPLES_PER_EP, NUM_ROUNDS, PAIRS, reward_mode=REWARD_MODE
        )
        raw_mean = float(raw_rewards.mean())
        raw_max = float(raw_rewards.max())

        entropy_beta = max(0.015 * (0.995 ** ep), 0.005)

        ppo_update(
            policy,
            optimizer,
            actions,
            logps,
            rewards,
            clip_ratio=0.1,
            entropy_beta=entropy_beta,
            train_iters=3
        )

        probs = tf.nn.sigmoid(policy.logits).numpy()
        prob_std = float(probs.std())
        curr_l, curr_r = get_diff_hex_from_probs(probs)

        top = np.argsort(probs)[-4:][::-1]
        top_info = ", ".join(f"b{i}:{probs[i]:.2f}" for i in top)

        bar = "".join(
            "█" if p > 0.6 else
            "▒" if p > 0.4 else
            "░" for p in probs[:16]
        ) + "|" + "".join(
            "█" if p > 0.6 else
            "▒" if p > 0.4 else
            "░" for p in probs[16:]
        )

        best_prob_std = max(best_prob_std, prob_std)

        print(f"Ep {ep:03d} | Rμ:{raw_mean:8.2f} Rmax:{raw_max:8.2f} | σp:{prob_std:.3f} | {top_info} | {bar} |({hex(curr_l)}, {hex(curr_r)})")

    print("\n🎯 PPO 训练完成")
    print(f"最大 bit 概率方差 σp = {best_prob_std:.4f}")
    print(f"最终差分: ΔL={hex(curr_l)}, ΔR={hex(curr_r)}")

    # 强评估：先评估 PPO 找到的
    accs, m, s = strong_eval_train(curr_l, curr_r, num_rounds=NUM_ROUNDS, pairs=PAIRS, repeats=3,
                                   epochs=15, depth=5, num_blocks=3)
    print("[Final Train-eval] accs =", accs, "mean =", m, "std =", s)


