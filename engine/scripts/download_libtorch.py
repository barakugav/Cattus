import argparse
import platform
import shutil
import tempfile
import urllib
import zipfile
from pathlib import Path

CRATE_DIR = Path(__file__).parent.parent.resolve()
PYTORCH_DIR = CRATE_DIR / "third-party" / "libtorch"


def main():
    parser = argparse.ArgumentParser("Download libtorch")
    parser.add_argument("--version", type=str, default="2.9.1")
    parser.add_argument(
        "--cuda", type=str, default="none", choices=["none", "cu118", "cu121", "cu129"]
    )
    args = parser.parse_args()

    if PYTORCH_DIR.exists():
        print(f"Removing existing directory {PYTORCH_DIR}...")
        shutil.rmtree(PYTORCH_DIR)

    match platform.system():
        case "Linux":
            match args.cuda:
                case "none":
                    url = f"cpu/libtorch-shared-with-deps-{args.version}%2Bcpu.zip"
                case "cu118" | "cu121" | "cu129":
                    url = f"{args.cuda}/libtorch-shared-with-deps-{args.version}%2B{args.cuda}.zip"
                case _:
                    raise ValueError(f"Unsupported CUDA version: {args.cuda}")
        case "Darwin":  # macOS
            if args.cuda != "none":
                raise ValueError("CUDA is not supported on macOS")
            url = f"cpu/libtorch-macos-arm64-{args.version}.zip"
        case "Windows":
            match args.cuda:
                case "none":
                    url = f"cpu/libtorch-win-shared-with-deps-{args.version}%2Bcpu.zip"
                case "cu118" | "cu121" | "cu129":
                    url = f"{args.cuda}/libtorch-win-shared-with-deps-{args.version}%2B{args.cuda}.zip"
                case _:
                    raise ValueError(f"Unsupported CUDA version: {args.cuda}")
        case unknown_system:
            raise ValueError(f"Unsupported platform: {unknown_system}")
    url = f"https://download.pytorch.org/libtorch/{url}"

    download_and_extract_zip(url, PYTORCH_DIR)


def download_and_extract_zip(url: str, extract_to: Path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        zip_path = Path(tmpdirname) / "temp.zip"
        print(f"Downloading from {url}...")
        with urllib.request.urlopen(url) as response, open(zip_path, "wb") as f:
            shutil.copyfileobj(response, f)

        print(f"Extracting to {extract_to}...")
        extracted_dir = Path(tmpdirname) / "extracted"
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extracted_dir)

        shutil.move(extracted_dir / "libtorch", extract_to)


if __name__ == "__main__":
    main()
