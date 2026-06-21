from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web_attack_sim import ActionType, WebAttackSimEnv  # noqa: E402
from web_attack_sim.tasks import load_task_config  # noqa: E402
from web_attack_sim.tasks import bundled_task_paths  # noqa: E402


PLANS = {
    "web_001_backup_leak": [
        ActionType.WEB_PATH_ENUMERATION,
        ActionType.WEB_PATH_ENUMERATION,
        ActionType.CONTENT_RETRIEVAL,
        ActionType.WEB_PATH_ENUMERATION,
        ActionType.INPUT_DISCOVERY,
        ActionType.CREDENTIAL_USE,
        ActionType.SENSITIVE_FILE_READ,
    ],
    "web_002_sqli_login": [
        ActionType.WEB_PATH_ENUMERATION,
        ActionType.WEB_PATH_ENUMERATION,
        ActionType.INPUT_DISCOVERY,
        ActionType.VULNERABILITY_CHECK,
        ActionType.EXPLOIT_ATTEMPT,
        ActionType.WEB_PATH_ENUMERATION,
        ActionType.INPUT_DISCOVERY,
        ActionType.CREDENTIAL_USE,
        ActionType.SENSITIVE_FILE_READ,
    ],
    "web_003_upload_shell": [
        ActionType.WEB_PATH_ENUMERATION,
        ActionType.WEB_PATH_ENUMERATION,
        ActionType.INPUT_DISCOVERY,
        ActionType.AUTH_ATTEMPT,
        ActionType.WEB_PATH_ENUMERATION,
        ActionType.INPUT_DISCOVERY,
        ActionType.FILE_UPLOAD_ATTEMPT,
        ActionType.COMMAND_EXECUTION,
        ActionType.SENSITIVE_FILE_READ,
    ],
}


def run_task(task_path: Path) -> None:
    env = WebAttackSimEnv()
    obs, info = env.reset(task_path)
    task_id = info["task_id"]
    task_config = load_task_config(task_path)
    plan = task_config.get("expert_plan") or [action.value for action in PLANS[task_id]]
    total_reward = 0.0
    print(f"\n== {task_id} ==")
    print("initial:", obs.to_dict())

    for action in plan:
        obs, reward, done, _truncated, info = env.step(action)
        total_reward += reward
        feedback = info["feedback"]
        action_type = info["action"]["action_type"]
        print(f"{action_type:24s} reward={reward:5.1f} event={feedback['progress_event']} error={feedback['error_type']}")
        if done:
            break

    print("final:", obs.to_dict())
    print("total_reward:", round(total_reward, 2))
    assert done, f"{task_id} did not terminate"
    assert env.state is not None and env.state.done


def main() -> None:
    for task_path in bundled_task_paths():
        run_task(task_path)
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
