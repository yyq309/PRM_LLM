from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import random
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web_attack_sim import WebAttackSimEnv  # noqa: E402
from web_attack_sim.reward import DEFAULT_REWARDS  # noqa: E402
from web_attack_sim.tasks import bundled_task_paths  # noqa: E402

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required to train DQN") from exc


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, num_actions: int, seed: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.next_masks = np.ones((capacity, num_actions), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_action_mask: list[int],
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.next_masks[self.ptr] = np.asarray(next_action_mask, dtype=np.float32)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        idxs = self.rng.choice(self.size, size=batch_size, replace=False)
        return (
            torch.as_tensor(self.obs[idxs], device=device),
            torch.as_tensor(self.actions[idxs], device=device),
            torch.as_tensor(self.rewards[idxs], device=device),
            torch.as_tensor(self.next_obs[idxs], device=device),
            torch.as_tensor(self.dones[idxs], device=device),
            torch.as_tensor(self.next_masks[idxs], device=device),
        )


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = obs_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.ReLU())
            last_dim = hidden
        layers.append(nn.Linear(last_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class EpisodeResult:
    task_id: str
    reward: float
    steps: int
    goal: bool


class WebDQNAgent:
    def __init__(
        self,
        *,
        seed: int,
        lr: float,
        gamma: float,
        batch_size: int,
        replay_size: int,
        hidden_sizes: list[int],
        target_update_freq: int,
        device: str,
        mask_invalid_actions: bool,
        task_paths: list[Path] | None = None,
        rewards: dict[str, float] | None = None,
        decay_recon_reward: bool = False,
    ):
        self.rng = random.Random(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.env = WebAttackSimEnv(rewards=rewards, decay_recon_reward=decay_recon_reward)
        self.task_paths = list(task_paths or bundled_task_paths())
        if not self.task_paths:
            raise ValueError("at least one training task is required")

        sample_task = self.task_paths[0]
        sample_obs, _ = self.env.reset(sample_task)
        sample_vec = np.asarray(self.env.encode_observation(sample_obs), dtype=np.float32)

        self.obs_dim = int(sample_vec.shape[0])
        self.num_actions = len(self.env.actions)
        self.device = torch.device(device)

        self.q_net = QNetwork(self.obs_dim, self.num_actions, hidden_sizes).to(self.device)
        self.target_net = QNetwork(self.obs_dim, self.num_actions, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(replay_size, self.obs_dim, self.num_actions, seed)

        self.gamma = gamma
        self.batch_size = batch_size
        self.hidden_sizes = list(hidden_sizes)
        self.mask_invalid_actions = mask_invalid_actions
        self.target_update_freq = target_update_freq
        self.steps_done = 0

    def select_action(self, obs_vec: np.ndarray, epsilon: float, action_mask: list[int] | None = None) -> int:
        valid_actions = self._valid_action_ids(action_mask)
        if self.rng.random() < epsilon:
            return self.rng.choice(valid_actions)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(obs_t).squeeze(0)
            if action_mask is not None:
                mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
                if bool(mask_t.any()):
                    q_values = q_values.masked_fill(~mask_t, -1e9)
            return int(torch.argmax(q_values).item())

    def optimize(self) -> float | None:
        if self.replay.size < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones, next_masks = self.replay.sample(self.batch_size, self.device)
        q_selected = self.q_net(obs).gather(1, actions)
        with torch.no_grad():
            next_q_all = self.target_net(next_obs)
            if self.mask_invalid_actions:
                valid = next_masks > 0
                masked_next_q = next_q_all.masked_fill(~valid, -1e9)
                next_q = masked_next_q.max(dim=1, keepdim=True).values
                no_valid = ~valid.any(dim=1, keepdim=True)
                next_q = torch.where(no_valid, torch.zeros_like(next_q), next_q)
            else:
                next_q = next_q_all.max(dim=1, keepdim=True).values
            target = rewards + self.gamma * (1.0 - dones) * next_q
        loss = F.smooth_l1_loss(q_selected, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())

    def train_episode(self, epsilon: float) -> EpisodeResult:
        task_path = self.rng.choice(self.task_paths)
        obs, info = self.env.reset(task_path)
        task_id = str(info["task_id"])
        obs_vec = np.asarray(self.env.encode_observation(obs), dtype=np.float32)
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            action_mask = self.current_action_mask()
            action = self.select_action(obs_vec, epsilon, action_mask)
            next_obs, reward, done, _truncated, _step_info = self.env.step(action)
            next_obs_vec = np.asarray(self.env.encode_observation(next_obs), dtype=np.float32)
            next_action_mask = self.current_action_mask()
            self.replay.add(obs_vec, action, reward, next_obs_vec, done, next_action_mask)
            self.steps_done += 1
            self.optimize()

            obs_vec = next_obs_vec
            episode_reward += reward
            steps += 1

        return EpisodeResult(task_id=task_id, reward=episode_reward, steps=steps, goal=self.goal_reached())

    def eval_episode(self, task_path: Path | None = None, epsilon: float = 0.0) -> EpisodeResult:
        if task_path is None:
            task_path = self.rng.choice(self.task_paths)
        obs, info = self.env.reset(task_path)
        task_id = str(info["task_id"])
        obs_vec = np.asarray(self.env.encode_observation(obs), dtype=np.float32)
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            action = self.select_action(obs_vec, epsilon, self.current_action_mask())
            next_obs, reward, done, _truncated, _step_info = self.env.step(action)
            obs_vec = np.asarray(self.env.encode_observation(next_obs), dtype=np.float32)
            episode_reward += reward
            steps += 1
        return EpisodeResult(task_id=task_id, reward=episode_reward, steps=steps, goal=self.goal_reached())

    def current_action_mask(self) -> list[int]:
        return self.env.action_mask(permissive=not self.mask_invalid_actions)

    def _valid_action_ids(self, action_mask: list[int] | None) -> list[int]:
        if action_mask is None:
            return list(range(self.num_actions))
        valid = [idx for idx, allowed in enumerate(action_mask) if allowed]
        return valid or list(range(self.num_actions))

    def q_values(self, obs_vec: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.q_net(obs_t).squeeze(0).cpu().numpy()

    def score_action(self, obs_vec: np.ndarray, action_id: int, action_mask: list[int] | None = None) -> dict[str, Any]:
        q = self.q_values(obs_vec)
        if action_mask is not None and any(action_mask):
            masked_q = np.where(np.asarray(action_mask, dtype=bool), q, -1e9)
            greedy_action = int(np.argmax(masked_q))
        else:
            greedy_action = int(np.argmax(q))
        v = float(q[greedy_action])
        q_selected = float(q[action_id])
        return {
            "q_values": q.tolist(),
            "greedy_action": greedy_action,
            "greedy_action_type": self.env.actions[greedy_action].value,
            "v_web": v,
            "selected_action": int(action_id),
            "selected_action_type": self.env.actions[action_id].value,
            "selected_action_allowed": bool(action_mask[action_id]) if action_mask is not None else None,
            "q_selected": q_selected,
            "value_gap": float(v - q_selected) if action_mask is None or action_mask[action_id] else None,
        }

    def goal_reached(self) -> bool:
        return bool(self.env.state and self.env.state.done and self.env._goal_reached())

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": self.q_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "obs_dim": self.obs_dim,
            "num_actions": self.num_actions,
            "actions": [a.value for a in self.env.actions],
            "metadata": {
                "hidden_sizes": self.hidden_sizes,
                "mask_invalid_actions": self.mask_invalid_actions,
                **(metadata or {}),
            },
        }
        torch.save(payload, path)


def linear_epsilon(step: int, total_steps: int, start: float, final: float, fraction: float) -> float:
    decay_steps = max(1, int(total_steps * fraction))
    if step >= decay_steps:
        return final
    ratio = step / decay_steps
    return start + ratio * (final - start)


def resolve_task_paths(items: list[str] | None) -> list[Path]:
    if not items:
        return bundled_task_paths()

    resolved: list[Path] = []
    for item in items:
        pattern = item.replace("\\", "/")
        if any(char in pattern for char in "*?[]"):
            matches = sorted(ROOT.glob(pattern))
            if not matches:
                matches = sorted(Path.cwd().glob(pattern))
            resolved.extend(path for path in matches if path.suffix == ".json")
            continue

        path = Path(item)
        if not path.is_absolute():
            root_path = ROOT / path
            path = root_path if root_path.exists() else Path.cwd() / path
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.json")))
        else:
            resolved.append(path)

    unique = sorted({path.resolve() for path in resolved})
    if not unique:
        raise ValueError(f"no task configs matched: {items}")
    missing = [path for path in unique if not path.exists()]
    if missing:
        raise FileNotFoundError(f"task config not found: {missing[0]}")
    return unique


def build_reward_table(scale: float, jitter: float, seed: int) -> dict[str, float] | None:
    """Perturb the reward magnitudes for the §5.1 reward-sensitivity ablation.

    `scale` multiplies every reward event; `jitter` applies a deterministic per-event
    multiplicative noise in [1-jitter, 1+jitter]. Returns None when no perturbation is
    requested so the env keeps DEFAULT_REWARDS.
    """
    if scale == 1.0 and jitter == 0.0:
        return None
    # Seed the jitter on BOTH seed and scale so per-event multipliers differ across
    # configs: this perturbs the RELATIVE reward ratios (the hand-tuned constants), not
    # just a global multiplier (whose ranking-invariance is a trivial MDP property).
    rng = random.Random(seed * 7919 + 13 + int(round(scale * 1000)))
    table: dict[str, float] = {}
    for event, value in DEFAULT_REWARDS.items():
        factor = scale * (1.0 + rng.uniform(-jitter, jitter)) if jitter > 0 else scale
        table[event] = float(value) * factor
    return table


def train(args: argparse.Namespace) -> None:
    train_task_paths = resolve_task_paths(args.train_tasks)
    eval_task_paths = resolve_task_paths(args.eval_tasks) if args.eval_tasks else list(train_task_paths)
    rewards = build_reward_table(args.reward_scale, args.reward_jitter, args.seed)
    agent = WebDQNAgent(
        seed=args.seed,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        hidden_sizes=args.hidden_sizes,
        target_update_freq=args.target_update_freq,
        device=args.device,
        mask_invalid_actions=args.mask_invalid_actions,
        task_paths=train_task_paths,
        rewards=rewards,
        decay_recon_reward=args.decay_recon_reward,
    )

    # Curriculum (method §12.1): easy -> easy+medium -> all, by the task `difficulty` tag.
    curriculum_pools = None
    if args.curriculum:
        from web_attack_sim.tasks import load_task_config as _load

        by_diff: dict[str, list[Path]] = {"easy": [], "medium": [], "hard": []}
        for p in train_task_paths:
            by_diff.get(_load(p).get("difficulty", "hard"), by_diff["hard"]).append(p)
        easy = by_diff["easy"] or train_task_paths
        easy_med = (by_diff["easy"] + by_diff["medium"]) or train_task_paths
        curriculum_pools = [(0.25, easy), (0.5, easy_med), (1.0, list(train_task_paths))]
        print(f"curriculum: easy={len(easy)} easy+med={len(easy_med)} all={len(train_task_paths)}")

    results: list[dict[str, Any]] = []
    while agent.steps_done < args.training_steps:
        if curriculum_pools is not None:
            progress = agent.steps_done / max(args.training_steps, 1)
            agent.task_paths = next(pool for frac, pool in curriculum_pools if progress < frac)
        epsilon = linear_epsilon(agent.steps_done, args.training_steps, args.init_epsilon, args.final_epsilon, args.exploration_fraction)
        result = agent.train_episode(epsilon)
        results.append(result.__dict__)
        if len(results) % args.log_every == 0:
            recent = results[-args.log_every :]
            avg_reward = sum(r["reward"] for r in recent) / len(recent)
            goal_rate = sum(int(r["goal"]) for r in recent) / len(recent)
            avg_steps = sum(r["steps"] for r in recent) / len(recent)
            print(
                f"episodes={len(results):4d} steps={agent.steps_done:6d}/{args.training_steps} "
                f"eps={epsilon:.3f} avg_reward={avg_reward:6.2f} goal_rate={goal_rate:.2f} avg_steps={avg_steps:.1f}"
            )

    eval_results = [agent.eval_episode(task_path=path).__dict__ for path in eval_task_paths for _ in range(args.eval_episodes)]
    eval_goal_rate = sum(int(r["goal"]) for r in eval_results) / max(len(eval_results), 1)
    eval_avg_reward = sum(r["reward"] for r in eval_results) / max(len(eval_results), 1)
    print(f"eval_goal_rate={eval_goal_rate:.2f} eval_avg_reward={eval_avg_reward:.2f}")

    metadata = {
        "training_steps": args.training_steps,
        "seed": args.seed,
        "mask_invalid_actions": args.mask_invalid_actions,
        "reward_scale": args.reward_scale,
        "reward_jitter": args.reward_jitter,
        "train_tasks": [str(path) for path in train_task_paths],
        "eval_tasks": [str(path) for path in eval_task_paths],
        "eval_goal_rate": eval_goal_rate,
        "eval_avg_reward": eval_avg_reward,
        "eval_results": eval_results,
    }
    agent.save(args.output, metadata)

    report_path = args.output.with_suffix(".eval.json")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"saved checkpoint: {args.output}")
    print(f"saved eval report: {report_path}")

    obs, _info = agent.env.reset(agent.task_paths[0])
    obs_vec = np.asarray(agent.env.encode_observation(obs), dtype=np.float32)
    q_report = agent.score_action(obs_vec, action_id=2, action_mask=agent.current_action_mask())
    print("sample_q_report:", json.dumps(q_report, ensure_ascii=False)[:500])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-steps", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=20000)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--target-update-freq", type=int, default=250)
    parser.add_argument("--init-epsilon", type=float, default=1.0)
    parser.add_argument("--final-epsilon", type=float, default=0.05)
    parser.add_argument("--exploration-fraction", type=float, default=0.55)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train-tasks", nargs="*", default=None, help="Task JSON paths, dirs, or ROOT-relative globs.")
    parser.add_argument("--eval-tasks", nargs="*", default=None, help="Held-out eval task JSON paths, dirs, or ROOT-relative globs.")
    parser.set_defaults(mask_invalid_actions=True)
    parser.add_argument(
        "--no-action-mask",
        dest="mask_invalid_actions",
        action="store_false",
        help="Disable strict action masks during DQN selection and Bellman targets.",
    )
    parser.add_argument("--curriculum", action="store_true", help="Easy->medium->hard curriculum by task difficulty tag (method §12.1).")
    parser.add_argument("--reward-scale", type=float, default=1.0, help="Multiply all reward magnitudes (reward-sensitivity ablation).")
    parser.add_argument("--reward-jitter", type=float, default=0.0, help="Per-event multiplicative reward noise in [1-j, 1+j].")
    parser.add_argument("--decay-recon-reward", action="store_true",
                        help="G2/H reward fix: zero the recon info-discovery bonus (path/input/fingerprint/service "
                             "found) once a foothold (webshell/command_execution) exists, so the oracle stops "
                             "over-valuing post-foothold recon.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "web_dqn.pt")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
