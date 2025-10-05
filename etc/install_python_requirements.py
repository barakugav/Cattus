import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()


def main() -> None:
    subprocess.check_call([sys.executable, "-m", "ensurepip"], cwd=ROOT_DIR)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"], cwd=ROOT_DIR)
    subprocess.check_call(
        [
            sys.executable,
            "-muv",
            "pip",
            "install",
            "-r",
            ROOT_DIR / "etc" / "requirements.txt",
            "--torch-backend=cpu",
        ],
        cwd=ROOT_DIR,
    )


if __name__ == "__main__":
    main()
