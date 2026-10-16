"""Build a pinned abc2midi executable locally without a system installation."""

import io
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVISION = "333ca1566c9682df3362d3d0f9a7d2ab431a26fd"


def main():
    tools_dir = ROOT / "tools"
    tools_dir.mkdir(exist_ok=True)
    url = f"https://codeload.github.com/sshlien/abcmidi/tar.gz/{REVISION}"
    payload = urllib.request.urlopen(url, timeout=60).read()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        archive.extractall(tools_dir, filter="data")
    source = tools_dir / f"abcmidi-{REVISION}"
    subprocess.run(["make", "-f", "Makefile", "abc2midi"], cwd=source, check=True)
    shutil.copy2(source / "abc2midi", tools_dir / "abc2midi")
    print(tools_dir / "abc2midi")


if __name__ == "__main__":
    main()
