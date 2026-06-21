from pathlib import Path
import argparse
import json
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web_attack_sim import WebAttackSimEnv  # noqa: E402
from web_attack_sim.tasks import bundled_task_paths  # noqa: E402


def export_rollouts(output: Path, episodes: int, seed: int) -> None:
    rng = random.Random(seed)
    task_paths = bundled_task_paths()
    env = WebAttackSimEnv()

    with output.open("w", encoding="utf-8") as f:
        for episode_id in range(episodes):
            task_path = rng.choice(task_paths)
            obs, info = env.reset(task_path)
            task_id = info["task_id"]
            done = False
            step_idx = 0

            while not done:
                obs_before = obs
                action_id = rng.randrange(len(env.actions))
                obs, reward, done, truncated, step_info = env.step(action_id)
                row = {
                    "episode_id": episode_id,
                    "task_id": task_id,
                    "step": step_idx,
                    "obs": obs_before.to_dict(),
                    "obs_vec": env.encode_observation(obs_before),
                    "action_id": action_id,
                    "action_type": env.actions[action_id].value,
                    "reward": reward,
                    "next_obs": obs.to_dict(),
                    "next_obs_vec": env.encode_observation(obs),
                    "done": done,
                    "truncated": truncated,
                    "feedback": step_info["feedback"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                step_idx += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "random_transitions.jsonl")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_rollouts(args.output, args.episodes, args.seed)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

