import os
import shutil
from pathlib import Path

LOCAL_BINARY = Path(__file__).resolve().parent / "tools/abc2midi"
if "ABC2MIDI_BIN" not in os.environ and LOCAL_BINARY.is_file():
    os.environ["ABC2MIDI_BIN"] = str(LOCAL_BINARY)

from vendor.xiaomi_music_scorer import compute_score  # noqa: E402


def music_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = (
            completion
            if isinstance(completion, str)
            else "".join(message["content"] for message in completion)
        )
        rewards.append(compute_score("music", text))
    return rewards


def check_scorer():
    binary = os.environ.get("ABC2MIDI_BIN", "abc2midi")
    if not shutil.which(binary):
        raise FileNotFoundError(f"Missing abc2midi: {binary}")
    example = "X:1\nM:4/4\nL:1/8\nQ:1/4=93\nK:Em\nE2 G2 B2 e2 | E8 |]"
    if music_reward([example])[0] <= 0:
        raise RuntimeError("Music scorer smoke check failed")


if __name__ == "__main__":
    import sys

    check_scorer()
    print(music_reward([Path(sys.argv[1]).read_text()])[0])
