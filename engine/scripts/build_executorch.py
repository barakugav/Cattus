import argparse
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CRATE_DIR = Path(__file__).parent.parent.resolve()
EXECUTORCH_DIR = CRATE_DIR / "third-party" / "executorch"


def main():
    parser = argparse.ArgumentParser("Build executorch from source")
    parser.add_argument(
        "--xnnpack",
        action="store_true",
        help="Enable XNNPACK support",
    )
    parser.add_argument(
        "--mps",
        action="store_true",
        help="Enable MPS (Metal Performance Shaders) support",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the existing executorch directory before cloning",
    )
    parser.add_argument(
        "--no-new-venv",
        action="store_true",
        help="Create a new virtual environment for the building process",
    )
    args = parser.parse_args()

    subprocess.check_call([sys.executable, "-m", "ensurepip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])

    if not args.no_new_venv:
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            if os.name == "nt":  # Windows
                bin_dir = tmpdir / ".venv" / "Scripts"
                uv_exe = "uv.exe"
            else:
                bin_dir = tmpdir / ".venv" / "bin"
                uv_exe = "uv"
            if not (bin_dir / uv_exe).exists():
                subprocess.check_call(
                    ["uv", "venv", "-p", "3.12", "--seed"], cwd=tmpdir
                )
                subprocess.check_call([bin_dir / "pip", "install", "uv"], cwd=tmpdir)
            subprocess.check_call(
                [
                    bin_dir / "python",
                    __file__,
                    *(["--xnnpack"] if args.xnnpack else []),
                    *(["--mps"] if args.mps else []),
                    *(["--clean"] if args.clean else []),
                    "--no-new-venv",
                ],
            )
            return

    clone_executorch()

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "uv",
            "pip",
            "install",
            "-r",
            EXECUTORCH_DIR / "requirements-dev.txt",
            "torch==2.9.1",
            "--torch-backend",
            "cpu",
        ]
    )
    build_executorch(
        xnnpack=args.xnnpack,
        mps=args.mps,
        clean=args.clean,
    )


def clone_executorch():
    if not EXECUTORCH_DIR.exists():
        EXECUTORCH_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "v1.0.1",
                "https://github.com/pytorch/executorch.git",
                ".",
            ],
            cwd=EXECUTORCH_DIR,
        )

        if platform.system() == "Darwin":
            # Clone coremltools repo
            # Required on apple when EXECUTORCH_BUILD_DEVTOOLS=ON
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    "9.0",
                    "https://github.com/apple/coremltools.git",
                ],
                cwd=EXECUTORCH_DIR / "backends" / "apple" / "coreml" / "scripts",
            )

    subprocess.check_call(
        ["git", "submodule", "update", "--init", "--recursive"], cwd=EXECUTORCH_DIR
    )
    subprocess.check_call(
        ["git", "submodule", "sync", "--recursive"], cwd=EXECUTORCH_DIR
    )


def build_executorch(xnnpack: bool, mps: bool, clean: bool):
    build_dir = EXECUTORCH_DIR / "build"
    if clean and build_dir.exists():
        shutil.rmtree(build_dir)
    if not build_dir.exists():
        build_dir.mkdir()

    option = {True: "ON", False: "OFF"}
    cmake_flags = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_FLAGS=-O3",
        "-DCMAKE_C_FLAGS=-O3",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
        "-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF",
        "-DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=OFF",
        "-DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON",
        "-DEXECUTORCH_ENABLE_LOGGING=ON",
        "-DEXECUTORCH_BUILD_PORTABLE_OPS=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_MODULE=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON",
        "-DEXECUTORCH_BUILD_MPS=" + option[mps],
        "-DEXECUTORCH_BUILD_XNNPACK=" + option[xnnpack],
        "-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=OFF",
        "-DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON",
        "-DEXECUTORCH_BUILD_KERNELS_CUSTOM=OFF",
        "-DEXECUTORCH_BUILD_DEVTOOLS=OFF",
        "-DEXECUTORCH_ENABLE_EVENT_TRACER=OFF",
        *(["-DCMAKE_OSX_DEPLOYMENT_TARGET=12.0"] if mps else []),
        # "-DCMAKE_C_COMPILER=/Library/Developer/CommandLineTools/usr/bin/gcc",
        # "-DCMAKE_CXX_COMPILER=/Library/Developer/CommandLineTools/usr/bin/g++",
    ]
    subprocess.check_call(
        ["cmake", *cmake_flags, ".."],
        cwd=build_dir,
        env={
            **os.environ,
            **({"MACOSX_DEPLOYMENT_TARGET": "12.0"} if mps else {}),
        },
    )

    subprocess.check_call(
        ["cmake", "--build", build_dir, "-j" + str(multiprocessing.cpu_count() + 1)],
        cwd=EXECUTORCH_DIR,
    )


if __name__ == "__main__":
    main()
