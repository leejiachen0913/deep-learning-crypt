#test if rapid training runs using neural networks can be used to find good initial differences in Speck32/64
#idea: try all differences up to a certain weight and keep track of the performance level reached

import train_nets as tn
import speck as sp
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from keras.models import Model
from sklearn.linear_model import Ridge

from random import sample, randint
from collections import defaultdict
from math import log2

linear_model = Ridge(alpha=0.01);

#first, we train a network to distinguish 3-round Speck with a randomly chosen input difference
#then, we use the penultimate layer output of that network to preprocess output data for other differences
#The preprocessed output for 1000 examples each is fed as training data to a single-layer perceptron
#That single-layer perceptron is then evaluated using 1000 validation samples
#The validation accuracy of the perceptron is taken as an indication of how much the differential distribution deviates from uniformity for that input difference

def train_preprocessor(n, nr, epochs):
  net = tn.make_resnet(depth=1);
  net.compile(optimizer='adam',loss='mse',metrics=['acc']);
  #create a random input difference
  diff_in = (randint(0,2**16), randint(0,2**16));
  X,Y = sp.make_train_data(n, nr, diff=diff_in);
  net.fit(X,Y,epochs=epochs, batch_size=5000,validation_split=0.1);
  net_pp = Model(inputs=net.layers[0].input, outputs=net.layers[-2].output);
  return(net_pp);

def evaluate_diff(diff, net_pp, nr=3, n=1000):
  if (diff == 0): return(0.0);
  d = (diff >> 16, diff & 0xffff);
  X,Y = sp.make_train_data(2*n, nr,diff=d);
  Z = net_pp.predict(X,batch_size=5000);
  #perceptron.fit(Z[0:n],Y[0:n]);
  linear_model.fit(Z[0:n],Y[0:n]);
  #val_acc = perceptron.score(Z[n:],Y[n:]);
  Y2 = linear_model.predict(Z[n:]);
  Y2bin = (Y2 > 0.5);
  val_acc = float(np.sum(Y2bin == Y[n:])) / n;
  return(val_acc);

# ----------------------------
# Utils
# ----------------------------
def int_to_bits32(x: int) -> np.ndarray:
    """uint32 -> {0,1} float32 bits, little-endian or big-endian都可以，只要一致"""
    # 这里用 bit0 在 index0（LSB-first）
    bits = np.array([(x >> i) & 1 for i in range(32)], dtype=np.float32)
    return bits

def safe_nonzero_uint32(rng: np.random.Generator) -> int:
    x = int(rng.integers(0, 2**32, dtype=np.uint64))
    if x == 0:
        x = 1
    return x

# ----------------------------
# Env
# ----------------------------
@dataclass
class EnvConfig:
    num_bits: int = 32
    nr: int = 3
    n_eval: int = 1000
    episode_len: int = 32
    reward_scale: float = 100.0
    repeat_action_penalty: float = 0.0  # 可设 0.1 防止来回翻同一位
    k_repeat: int = 1  # 评估重复次数 K（训练早期=1；稳定后=3）
    use_score_cache: bool = True

class BitFlipDiffEnv:
    """
    状态：bits32 + 当前score (可选)
    动作：0..31 选择翻哪一位
    reward：Δscore * reward_scale - repeat_penalty(可选)
    """
    def __init__(self, net_pp, evaluate_fn, cfg: EnvConfig, seed: int = 0, include_score_in_state: bool = True):
        self.net_pp = net_pp
        self.evaluate_fn = evaluate_fn
        self.cfg = cfg
        self.include_score = include_score_in_state

        self.rng = np.random.default_rng(seed)
        self.diff: int = 1
        self.score: float = 0.5
        self.t: int = 0
        self.prev_action: Optional[int] = None

        self._cache: Dict[int, Tuple[float, int]] = {}  # diff -> (score_est, hits)

    def _score_once(self, diff: int) -> float:
        return float(self.evaluate_fn(diff, self.net_pp, nr=self.cfg.nr, n=self.cfg.n_eval))

    def score_diff(self, diff: int) -> float:
        """K 次平均 + 缓存"""
        if diff == 0:
            return 0.0

        if self.cfg.use_score_cache and diff in self._cache:
            return self._cache[diff][0]

        K = max(1, int(self.cfg.k_repeat))
        vals = [self._score_once(diff) for _ in range(K)]
        sc = float(np.mean(vals))

        if self.cfg.use_score_cache:
            self._cache[diff] = (sc, 1)
        return sc

    def reset(self) -> np.ndarray:
        self.t = 0
        self.prev_action = None
        self.diff = safe_nonzero_uint32(self.rng)
        self.score = self.score_diff(self.diff)
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        bits = int_to_bits32(self.diff)
        if self.include_score:
            return np.concatenate([bits, np.array([self.score], dtype=np.float32)], axis=0)
        return bits

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert 0 <= action < self.cfg.num_bits
        self.t += 1

        old_score = self.score
        new_diff = self.diff ^ (1 << action)
        new_score = self.score_diff(new_diff)

        # reward: 改进幅度
        r = (new_score - old_score) * self.cfg.reward_scale

        # 可选：惩罚连续重复同一 action（防抖动）
        if self.cfg.repeat_action_penalty > 0.0 and self.prev_action is not None:
            if action == self.prev_action:
                r -= self.cfg.repeat_action_penalty

        self.diff = new_diff
        self.score = new_score
        self.prev_action = action

        done = (self.t >= self.cfg.episode_len)

        info = {
            "diff": self.diff,
            "score": self.score,
            "old_score": old_score,
            "new_score": new_score
        }
        return self._get_state(), float(r), bool(done), info

# ----------------------------
# PPO Networks (Keras)
# ----------------------------
class ActorCritic(tf.keras.Model):
    def __init__(self, state_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(hidden, activation="tanh")
        self.fc2 = tf.keras.layers.Dense(hidden, activation="tanh")
        self.logits = tf.keras.layers.Dense(num_actions, activation=None)
        self.value = tf.keras.layers.Dense(1, activation=None)

        # build once
        dummy = tf.zeros([1, state_dim], dtype=tf.float32)
        _ = self(dummy)

    def call(self, x):
        z = self.fc1(x)
        z = self.fc2(z)
        return self.logits(z), self.value(z)

# ----------------------------
# GAE / Returns
# ----------------------------
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: [T]
    values:  [T+1] (最后一个是 bootstrap)
    dones:   [T]  终止标记
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
    returns = adv + values[:-1]
    return adv.astype(np.float32), returns.astype(np.float32)

# ----------------------------
# PPO Trainer
# ----------------------------
@dataclass
class PPOConfig:
    num_actions: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    train_epochs: int = 5          # 每批数据重复更新次数
    minibatch_size: int = 256
    max_grad_norm: float = 0.5

class PPOTrainer:
    def __init__(self, model: ActorCritic, cfg: PPOConfig):
        self.model = model
        self.cfg = cfg
        self.opt = tf.keras.optimizers.Adam(learning_rate=cfg.lr)

    @tf.function
    def _train_step(self, states, actions, old_logp, advantages, returns):
        with tf.GradientTape() as tape:
            logits, values = self.model(states)
            values = tf.squeeze(values, axis=-1)

            dist = tfp.distributions.Categorical(logits=logits)
            logp = dist.log_prob(actions)
            entropy = dist.entropy()

            ratio = tf.exp(logp - old_logp)
            clip_adv = tf.clip_by_value(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clip_adv))

            value_loss = tf.reduce_mean((returns - values) ** 2)

            ent_loss = -tf.reduce_mean(entropy)

            loss = policy_loss + self.cfg.vf_coef * value_loss + self.cfg.ent_coef * ent_loss

        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.cfg.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.cfg.max_grad_norm)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, policy_loss, value_loss, tf.reduce_mean(entropy)

    def update(self, batch):
        # batch: dict of numpy arrays
        states = tf.convert_to_tensor(batch["states"], tf.float32)
        actions = tf.convert_to_tensor(batch["actions"], tf.int32)
        old_logp = tf.convert_to_tensor(batch["logp"], tf.float32)
        adv = tf.convert_to_tensor(batch["adv"], tf.float32)
        ret = tf.convert_to_tensor(batch["ret"], tf.float32)

        # normalize advantages
        adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-8)

        N = states.shape[0]
        idx = np.arange(N)

        for _ in range(self.cfg.train_epochs):
            np.random.shuffle(idx)
            for start in range(0, N, self.cfg.minibatch_size):
                mb = idx[start:start+self.cfg.minibatch_size]
                self._train_step(tf.gather(states, mb),
                                 tf.gather(actions, mb),
                                 tf.gather(old_logp, mb),
                                 tf.gather(adv, mb),
                                 tf.gather(ret, mb))

# ----------------------------
# Rollout collection
# ----------------------------
def sample_action(model: ActorCritic, state: np.ndarray) -> Tuple[int, float, float]:
    """
    返回 action, logp, value
    """
    s = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
    logits, value = model(s)
    dist = tfp.distributions.Categorical(logits=logits)
    a = dist.sample()[0]
    logp = dist.log_prob(a)[0]
    return int(a.numpy()), float(logp.numpy()), float(value[0, 0].numpy())

def collect_rollouts(env: BitFlipDiffEnv, model: ActorCritic, num_episodes: int,
                     gamma: float, lam: float) -> dict:
    states, actions, logps, rewards, dones, values = [], [], [], [], [], []

    episode_infos = []

    for _ in range(num_episodes):
        s = env.reset()
        ep_scores = []
        ep_rewards = 0.0

        # rollout episode
        for t in range(env.cfg.episode_len):
            a, logp, v = sample_action(model, s)
            s2, r, done, info = env.step(a)

            states.append(s)
            actions.append(a)
            logps.append(logp)
            rewards.append(r)
            dones.append(done)
            values.append(v)

            ep_rewards += r
            ep_scores.append(info["score"])

            s = s2
            if done:
                break

        # bootstrap value for last state
        _, _, v_last = sample_action(model, s)
        values.append(v_last)  # 注意：每个 episode 会多 append 一次，需要对齐处理

        episode_infos.append({
            "final_diff": env.diff,
            "final_score": env.score,
            "mean_score": float(np.mean(ep_scores)) if ep_scores else float(env.score),
            "sum_reward": float(ep_rewards),
        })

    # 现在 values 比 rewards 多了 num_episodes 个末尾值，需要按 episode 分段计算 GAE
    # 简化做法：重新按 episode_len 切分
    T = env.cfg.episode_len
    adv_all, ret_all = [], []

    ptr_r = 0
    ptr_v = 0
    for _ in range(num_episodes):
        # 每个 episode 最多 T 步，但可能提前 done；我们按实际 done 来切
        # 找到这一段的 done=True 位置（如果没有提前 done，就取 T 步）
        seg_rewards = []
        seg_dones = []
        seg_values = []

        for _t in range(T):
            seg_rewards.append(rewards[ptr_r])
            seg_dones.append(dones[ptr_r])
            seg_values.append(values[ptr_v])
            ptr_r += 1
            ptr_v += 1
            if seg_dones[-1]:
                break

        # bootstrap
        seg_values.append(values[ptr_v])  # v_last
        ptr_v += 1

        adv, ret = compute_gae(seg_rewards, seg_values, seg_dones, gamma=gamma, lam=lam)
        adv_all.append(adv)
        ret_all.append(ret)

    adv_all = np.concatenate(adv_all, axis=0).astype(np.float32)
    ret_all = np.concatenate(ret_all, axis=0).astype(np.float32)

    batch = {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int32),
        "logp": np.asarray(logps, dtype=np.float32),
        "adv": adv_all,
        "ret": ret_all,
        "episode_infos": episode_infos
    }
    return batch

# ----------------------------
# Main training loop
# ----------------------------
def train_ppo_bitflip(
    net_pp,
    evaluate_diff_fn,
    total_iters: int = 200,
    episodes_per_iter: int = 8,
    env_cfg: EnvConfig = EnvConfig(),
    ppo_cfg: PPOConfig = PPOConfig(),
    seed: int = 0,
):
    # env
    env = BitFlipDiffEnv(net_pp, evaluate_diff_fn, env_cfg, seed=seed, include_score_in_state=True)
    state_dim = 32 + 1  # bits + score
    model = ActorCritic(state_dim=state_dim, num_actions=ppo_cfg.num_actions, hidden=128)
    trainer = PPOTrainer(model, ppo_cfg)

    best_diff = None
    best_score = -1.0

    for it in range(total_iters):
        batch = collect_rollouts(env, model, num_episodes=episodes_per_iter,
                                 gamma=ppo_cfg.gamma, lam=ppo_cfg.gae_lambda)
        trainer.update(batch)

        # logging
        infos = batch["episode_infos"]
        mean_final_score = float(np.mean([x["final_score"] for x in infos]))
        max_final_score = float(np.max([x["final_score"] for x in infos]))
        mean_sum_reward = float(np.mean([x["sum_reward"] for x in infos]))

        # track best
        best_ep = max(infos, key=lambda x: x["final_score"])
        if best_ep["final_score"] > best_score:
            best_score = best_ep["final_score"]
            best_diff = best_ep["final_diff"]

        print(f"[Iter {it:04d}] mean_final_score={mean_final_score:.4f} "
              f"max_final_score={max_final_score:.4f} mean_sum_reward={mean_sum_reward:.2f} "
              f"best_so_far_score={best_score:.4f} best_diff=0x{best_diff:08x}")

    return model, best_diff, best_score


#for a given difference, derive a guess how many rounds may be attackable
def extend_attack(diff, net_pp, nr, val_acc):
  print("Estimates of attack accuracy:");
  while(val_acc > 0.52):
    print(str(nr) + " rounds:" + str(val_acc));
    nr = nr + 1;
    val_acc = evaluate_diff(diff, net_pp, nr=nr, n=1000);

def greedy_optimizer_with_exploration(guess, f, n=2000, alpha=0.01, num_bits=32):
  best_guess = guess;
  best_val = f(guess); val = best_val;
  d = defaultdict(int)
  for i in range(n):
    d[guess] = d[guess] + 1;
    r = randint(0, num_bits-1);
    guess_neu = guess ^ (1 << r);
    val_neu = f(guess_neu);
    if (val_neu > best_val):
      best_val = val_neu; best_guess = guess_neu;
      print(hex(best_guess), best_val);
    if (val_neu - alpha*log2(d[guess_neu]+1) > val - alpha*log2(d[guess]+1)):
      val = val_neu; guess = guess_neu;
  return(best_guess, best_val);

net_pp = train_preprocessor(10**7,3,1);
for i in range(10):
  print("Run ",i,": ");
  diff, val_acc = greedy_optimizer_with_exploration(randint(0,2**32-1), lambda x: evaluate_diff(x, net_pp, 3));
  extend_attack(diff, net_pp, 3, val_acc);
