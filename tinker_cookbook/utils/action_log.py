import json
import os


def log_action(log_path: str, action: dict) -> None:
    """Append one action record to {log_path}/actions.jsonl."""
    os.makedirs(log_path, exist_ok=True)
    out_path = os.path.join(log_path, "actions.jsonl")
    with open(out_path, "a") as f:
        f.write(json.dumps(action, default=str) + "\n")
