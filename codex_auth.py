"""Interactive device login with an explicitly mounted persistent auth volume."""

import os
import subprocess

import modal

app = modal.App("mimo-music-codex-login")
AUTH_VOLUME = os.environ.get("MUSIC_CODEX_AUTH_VOLUME", "mimo-music-codex-auth")
auth = modal.Volume.from_name(AUTH_VOLUME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("nodejs", "npm")
    .run_commands("npm install -g @openai/codex@0.160.0")
    .env({"MUSIC_CODEX_AUTH_VOLUME": AUTH_VOLUME})
)


@app.function(image=image, cpu=0.25, memory=512, timeout=1200, volumes={"/root/.codex": auth})
def login():
    subprocess.run(
        ["codex", "-c", 'cli_auth_credentials_store="file"', "login", "--device-auth"],
        check=True,
        timeout=1000,
    )
    subprocess.run(["codex", "login", "status"], check=True)
    auth.commit()
    print("New Codex login saved to the mounted auth volume.", flush=True)


@app.function(image=image, cpu=0.25, memory=512, timeout=60, volumes={"/root/.codex": auth})
def check():
    subprocess.run(["codex", "login", "status"], check=True, timeout=30)
    print(f"Codex login available in {AUTH_VOLUME}.", flush=True)


@app.local_entrypoint()
def main(check_only: bool = False):
    if check_only:
        check.remote()
    else:
        login.remote()
