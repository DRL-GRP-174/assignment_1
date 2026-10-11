"""
PPO (clip) re-implementation - DRL Assignment Problem V, Group 174
Paper: Schulman et al., "Proximal Policy Optimization Algorithms", 2017 (arXiv:1707.06347)

We follow Algorithm 1 of the paper (actor-critic style):
  1. N parallel actors run pi_old for T steps
  2. compute GAE advantages
  3. optimise L^{CLIP+VF+S} for K epochs with minibatches of size M
  4. pi_old <- pi, repeat

Usage:
  python ppo.py --env CartPole-v1 --clip 0.2 --seed 1
  python ppo.py --env CartPole-v1 --clip 0     # 0 = no clipping (plain surrogate, row 1 of Table 1)
"""
import argparse
import csv
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=150_000)
    p.add_argument("--num-envs", type=int, default=4)        # N actors
    p.add_argument("--num-steps", type=int, default=128)     # T horizon
    p.add_argument("--epochs", type=int, default=10)         # K
    p.add_argument("--minibatches", type=int, default=4)     # NT / M
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)        # GAE lambda
    p.add_argument("--clip", type=float, default=0.2)        # epsilon, 0 disables clipping
    p.add_argument("--vf-coef", type=float, default=0.5)     # c1
    p.add_argument("--ent-coef", type=float, default=0.01)   # c2
    p.add_argument("--out", default="results")
    return p.parse_args()


class ActorCritic(nn.Module):
    # separate actor / critic MLPs, 64-64 tanh (same sizes the paper uses for MuJoCo)
    def __init__(self, obs_dim, n_actions):
        super().__init__()

        def layer(i, o, std=np.sqrt(2)):
            l = nn.Linear(i, o)
            nn.init.orthogonal_(l.weight, std)
            nn.init.constant_(l.bias, 0.0)
            return l

        self.critic = nn.Sequential(layer(obs_dim, 64), nn.Tanh(), layer(64, 64), nn.Tanh(), layer(64, 1, 1.0))
        self.actor = nn.Sequential(layer(obs_dim, 64), nn.Tanh(), layer(64, 64), nn.Tanh(), layer(64, n_actions, 0.01))

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def act(self, x, action=None):
        dist = Categorical(logits=self.actor(x))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.value(x)


def compute_gae(rewards, values, dones, last_value, last_done, gamma, lam):
    # truncated GAE, eq. (11)-(12) in the paper
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - last_done
            next_value = last_value
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        gae = delta + gamma * lam * next_nonterminal * gae
        adv[t] = gae
    return adv, adv + values


def ppo_loss(new_logp, old_logp, adv, eps):
    """L^CLIP from eq. (7). Returns the loss to *minimise* (so negated)."""
    ratio = torch.exp(new_logp - old_logp)               # r_t(theta)
    surr1 = ratio * adv
    if eps > 0:
        surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
        return -torch.min(surr1, surr2).mean(), ratio
    return -surr1.mean(), ratio                          # no clipping baseline


def main():
    args = get_args()
    run_name = f"{args.env}_clip{args.clip}_s{args.seed}"
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)  # small MLP - 1 thread is faster, and lets us run seeds in parallel

    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(args.env)) for _ in range(args.num_envs)]
    )
    obs_dim = envs.single_observation_space.shape[0]
    n_act = envs.single_action_space.n
    agent = ActorCritic(obs_dim, n_act)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    N, T = args.num_envs, args.num_steps
    batch = N * T
    mb_size = batch // args.minibatches
    n_updates = args.total_steps // batch

    obs_buf = torch.zeros((T, N, obs_dim))
    act_buf = torch.zeros((T, N), dtype=torch.long)
    logp_buf = torch.zeros((T, N))
    rew_buf = torch.zeros((T, N))
    done_buf = torch.zeros((T, N))
    val_buf = torch.zeros((T, N))

    obs, _ = envs.reset(seed=args.seed)
    obs = torch.tensor(obs, dtype=torch.float32)
    done = torch.zeros(N)
    step = 0
    recent = []
    log_rows = []
    t0 = time.time()

    for update in range(1, n_updates + 1):
        # linear lr annealing, as in the paper's Atari setup
        opt.param_groups[0]["lr"] = args.lr * (1 - (update - 1) / n_updates)

        # ---- 1. collect N*T timesteps with pi_old ----
        for t in range(T):
            step += N
            obs_buf[t], done_buf[t] = obs, done
            with torch.no_grad():
                a, logp, _, v = agent.act(obs)
            act_buf[t], logp_buf[t], val_buf[t] = a, logp, v
            obs, r, term, trunc, info = envs.step(a.numpy())
            rew_buf[t] = torch.tensor(r, dtype=torch.float32)
            obs = torch.tensor(obs, dtype=torch.float32)
            done = torch.tensor(np.logical_or(term, trunc), dtype=torch.float32)
            if "episode" in info:
                for i, finished in enumerate(info["_episode"]):
                    if finished:
                        recent.append(float(info["episode"]["r"][i]))

        # ---- 2. advantages with GAE ----
        with torch.no_grad():
            last_v = agent.value(obs)
        adv, returns = compute_gae(rew_buf, val_buf, done_buf, last_v, done, args.gamma, args.lam)

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act = act_buf.reshape(-1)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = returns.reshape(-1)

        # ---- 3. K epochs of minibatch Adam on L^{CLIP+VF+S} ----
        clipfracs, kls = [], []
        idx = np.arange(batch)
        for epoch in range(args.epochs):
            np.random.shuffle(idx)
            for s in range(0, batch, mb_size):
                mb = idx[s:s + mb_size]
                _, new_logp, ent, new_v = agent.act(b_obs[mb], b_act[mb])
                mb_adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)

                pg_loss, ratio = ppo_loss(new_logp, b_logp[mb], mb_adv, args.clip)
                v_loss = ((new_v - b_ret[mb]) ** 2).mean()
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent.mean()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                opt.step()

                with torch.no_grad():
                    kls.append(((ratio - 1) - torch.log(ratio)).mean().item())
                    clipfracs.append(((ratio - 1).abs() > max(args.clip, 0.2)).float().mean().item())

        # ---- 4. logging (pi_old <- pi happens implicitly next rollout) ----
        mean_ret = np.mean(recent[-20:]) if recent else 0.0
        log_rows.append([step, mean_ret, np.mean(kls), np.mean(clipfracs), v_loss.item()])
        if update % 10 == 0 or update == n_updates:
            print(f"[{run_name}] step {step:7d} | avg return {mean_ret:6.1f} | "
                  f"approx KL {np.mean(kls):.4f} | clipfrac {np.mean(clipfracs):.3f} | "
                  f"{time.time() - t0:5.1f}s")

    with open(os.path.join(args.out, run_name + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "avg_return", "approx_kl", "clip_frac", "value_loss"])
        w.writerows(log_rows)
    envs.close()


if __name__ == "__main__":
    main()
