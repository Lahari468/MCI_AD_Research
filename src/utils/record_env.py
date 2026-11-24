#!/usr/bin/env python3
"""Utility to inspect system environment, hardware, CLI neuroimaging tools,
and Python packages, and record software versions to logs/env_versions.json.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    """Run a shell command and return returncode, stdout, stderr."""
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        return res.returncode, res.stdout.strip(), res.stderr.strip()
    except Exception as e:
        return -1, "", str(e)


def get_hardware_info() -> dict[str, str | int | float | None]:
    """Get Apple Silicon hardware metrics via sysctl."""
    info: dict[str, str | int | float | None] = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }

    # Brand string (e.g. Apple M4)
    code, out, _ = run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    if code == 0 and out:
        info["cpu_brand"] = out
    else:
        info["cpu_brand"] = platform.processor()

    # Total RAM in bytes
    code, out, _ = run_cmd(["sysctl", "-n", "hw.memsize"])
    if code == 0 and out.isdigit():
        bytes_ram = int(out)
        info["total_ram_bytes"] = bytes_ram
        info["total_ram_gb"] = round(bytes_ram / (1024**3), 2)
    else:
        info["total_ram_bytes"] = None
        info["total_ram_gb"] = None

    return info


def check_python_packages(package_list: list[str]) -> dict[str, dict[str, str | bool]]:
    """Check installed versions of required python packages."""
    results = {}
    for pkg in package_list:
        installed = False
        version = None
        try:
            version = importlib.metadata.version(pkg)
            installed = True
        except importlib.metadata.PackageNotFoundError:
            # Fallback import check
            try:
                mod = __import__(pkg)
                version = getattr(mod, "__version__", "unknown")
                installed = True
            except ImportError:
                installed = False
                version = None

        results[pkg] = {
            "installed": installed,
            "version": version if installed else "NOT INSTALLED",
        }
    return results


def check_pytorch_mps() -> dict[str, bool | str | None]:
    """Check PyTorch MPS build, availability, and tensor execution."""
    result: dict[str, bool | str | None] = {
        "torch_installed": False,
        "torch_version": None,
        "mps_built": False,
        "mps_available": False,
        "tensor_allocation_test": None,
    }
    try:
        import torch

        result["torch_installed"] = True
        result["torch_version"] = torch.__version__
        result["mps_built"] = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_built()
        result["mps_available"] = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

        if result["mps_available"]:
            try:
                t = torch.ones((2, 2), device="mps")
                result["tensor_allocation_test"] = f"SUCCESS: Allocated shape {list(t.shape)} on {t.device}"
            except Exception as e:
                result["tensor_allocation_test"] = f"FAILED: {e}"
        else:
            result["tensor_allocation_test"] = "SKIPPED: MPS is not available"
    except ImportError:
        result["tensor_allocation_test"] = "SKIPPED: torch is not installed"

    return result


def inspect_fsl() -> dict[str, str | bool | None]:
    """Check FSL installation, binaries, and version."""
    fsldir = os.environ.get("FSLDIR", "/usr/local/fsl")
    fsl_info: dict[str, str | bool | None] = {
        "fsldir": fsldir,
        "fsldir_exists": os.path.isdir(fsldir),
        "fsl_version": None,
        "bin_in_path": False,
    }

    # Check fslversion file
    version_file = Path(fsldir) / "etc" / "fslversion"
    if version_file.is_file():
        try:
            fsl_info["fsl_version"] = version_file.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    # Check whether fsl bin is in current PATH
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    fsl_bin = str(Path(fsldir) / "bin")
    fsl_info["bin_in_path"] = fsl_bin in path_dirs

    tools = ["flirt", "bet", "fast"]
    for tool in tools:
        # Check fsl_bin directly first, then standard PATH
        direct_bin = os.path.join(fsl_bin, tool)
        if os.path.isfile(direct_bin):
            tool_path = direct_bin
        else:
            tool_path = shutil.which(tool)

        is_runnable = False
        tool_version = None
        arch = None

        if tool_path and os.path.isfile(tool_path):
            code_f, out_f, _ = run_cmd(["file", tool_path])
            arch = out_f if code_f == 0 else "unknown"

            if tool == "flirt":
                code_r, out_r, _ = run_cmd([tool_path, "-version"])
                is_runnable = code_r == 0 or "FLIRT" in out_r
                tool_version = out_r.strip() if out_r else "FLIRT"
            elif tool == "bet":
                code_r, out_r, err_r = run_cmd([tool_path])
                output_r = out_r + " " + err_r
                is_runnable = "Usage:    bet" in output_r or "bet <input>" in output_r
                tool_version = "FSL BET2"
            elif tool == "fast":
                code_r, out_r, err_r = run_cmd([tool_path])
                output_r = out_r + " " + err_r
                is_runnable = "FAST" in output_r
                tool_version = "FSL FAST"

        fsl_info[f"{tool}_path"] = tool_path
        fsl_info[f"{tool}_installed"] = bool(tool_path)
        fsl_info[f"{tool}_runnable"] = is_runnable
        fsl_info[f"{tool}_version"] = tool_version
        fsl_info[f"{tool}_arch"] = arch

    return fsl_info


def inspect_cli_tools() -> dict[str, dict[str, str | bool | None]]:
    """Inspect git, dcm2niix, docker, synthstrip."""
    tools: dict[str, dict[str, str | bool | None]] = {}

    # Git
    git_path = shutil.which("git")
    if git_path:
        code, out, _ = run_cmd([git_path, "--version"])
        code_f, out_f, _ = run_cmd(["file", git_path])
        tools["git"] = {
            "installed": True,
            "path": git_path,
            "version": out if code == 0 else None,
            "arch": out_f if code_f == 0 else None,
            "runnable": code == 0,
        }
    else:
        tools["git"] = {"installed": False, "path": None, "runnable": False}

    # dcm2niix
    dcm_path = shutil.which("dcm2niix")
    if dcm_path:
        code, out, err = run_cmd([dcm_path, "-v"])
        output = out or err
        code_f, out_f, _ = run_cmd(["file", dcm_path])
        ver = None
        for line in output.splitlines():
            if "dcm2niiX version" in line or line.startswith("v1."):
                ver = line.strip()
                break
        tools["dcm2niix"] = {
            "installed": True,
            "path": dcm_path,
            "version": ver or (output.splitlines()[-1] if output else None),
            "arch": out_f if code_f == 0 else None,
            "runnable": bool(ver) or ("Chris Rorden" in output),
        }
    else:
        tools["dcm2niix"] = {"installed": False, "path": None, "runnable": False}

    # Docker
    docker_path = shutil.which("docker")
    if docker_path:
        code_v, out_v, _ = run_cmd([docker_path, "--version"])
        code_f, out_f, _ = run_cmd(["file", docker_path])
        code_i, _, _ = run_cmd([docker_path, "info"])
        tools["docker"] = {
            "installed": True,
            "path": docker_path,
            "version": out_v if code_v == 0 else None,
            "arch": out_f if code_f == 0 else None,
            "runnable": code_v == 0,
            "daemon_running": code_i == 0,
        }
    else:
        tools["docker"] = {"installed": False, "path": None, "runnable": False, "daemon_running": False}

    # SynthStrip (FreeSurfer or standalone)
    mri_synthstrip_path = shutil.which("mri_synthstrip") or shutil.which("synthstrip")
    tools["synthstrip"] = {
        "installed": bool(mri_synthstrip_path),
        "path": mri_synthstrip_path,
        "runnable": bool(mri_synthstrip_path),
    }

    return tools


def inspect_package_managers() -> dict[str, dict[str, str | bool | None]]:
    """Inspect brew, pip, conda, mamba, uv."""
    managers = ["brew", "pip", "pip3", "conda", "mamba", "uv", "poetry"]
    results = {}
    for mgr in managers:
        path = shutil.which(mgr)
        if path:
            code, out, _ = run_cmd([path, "--version"])
            results[mgr] = {
                "available": True,
                "path": path,
                "version": out if code == 0 else None,
            }
        else:
            results[mgr] = {"available": False, "path": None, "version": None}
    return results


def record_environment(output_path: Path | str = "logs/env_versions.json") -> dict:
    """Collect all environment information and write to json."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    required_packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "nibabel",
        "torch",
        "monai",
        "matplotlib",
        "seaborn",
        "shap",
        "pydantic",
        "pyyaml",
        "pytest",
        "pydicom",
        "SimpleITK",
    ]

    report = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "timestamp_local": datetime.datetime.now().astimezone().isoformat(),
        "os": {
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
        },
        "hardware": get_hardware_info(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "build": platform.python_build(),
            "compiler": platform.python_compiler(),
        },
        "package_managers": inspect_package_managers(),
        "cli_tools": inspect_cli_tools(),
        "fsl": inspect_fsl(),
        "pytorch_mps": check_pytorch_mps(),
        "python_packages": check_python_packages(required_packages),
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    out_target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/env_versions.json")
    print(f"Inspecting environment and generating {out_target}...")
    data = record_environment(out_target)
    print(f"Successfully recorded environment to {out_target}")
