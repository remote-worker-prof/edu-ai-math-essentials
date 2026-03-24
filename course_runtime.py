"""Shared runtime configuration for course notebooks.

This module keeps the student-facing notebook setup short and consistent across
local runs, Google Colab, and Kaggle.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

VALID_RUNTIME_MODES = {
    "auto",
    "local-cpu",
    "local-gpu",
    "colab-cpu",
    "colab-gpu",
    "kaggle-cpu",
    "kaggle-gpu",
}

PLACEHOLDER_COURSE_REPO_HTTPS_URL = "https://github.com/<org>/<repo>.git"
COURSE_REPO_DIRNAME = "students-AI_math_essentials"
MANAGED_CLOUD_PACKAGES = {
    "tensorflow",
    "jupyter",
    "ipykernel",
    "nbconvert",
    "nbformat",
}

_RUNTIME_STATE: RuntimeInfo | None = None


@dataclass(frozen=True)
class RuntimeInfo:
    requested_mode: str
    effective_mode: str
    platform: str
    repo_root: str
    visible_gpus: tuple[str, ...]
    compute_device: str
    cloud_bootstrap: bool
    dependencies_installed: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_notebook_platform() -> str:
    """Return local/colab/kaggle based on environment signals."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").exists():
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or "google.colab" in sys.modules:
        return "colab"
    return "local"


def cloud_repo_root(platform: str) -> Path:
    if platform == "colab":
        return Path("/content") / COURSE_REPO_DIRNAME
    if platform == "kaggle":
        return Path("/kaggle/working") / COURSE_REPO_DIRNAME
    raise ValueError(
        f"Cloud repo root is only defined for colab/kaggle, got {platform!r}."
    )


def setup_notebook_runtime(
    runtime_mode: str = "auto",
    course_repo_https_url: str = PLACEHOLDER_COURSE_REPO_HTTPS_URL,
    notebook_requirements: str = "",
) -> RuntimeInfo:
    """Configure repo root, optional cloud dependencies, and TensorFlow devices."""
    global _RUNTIME_STATE

    runtime_mode = os.environ.get("COURSE_RUNTIME_MODE", runtime_mode)
    course_repo_https_url = os.environ.get(
        "COURSE_REPO_HTTPS_URL", course_repo_https_url
    )
    runtime_mode = normalize_runtime_mode(runtime_mode)

    if _RUNTIME_STATE is not None:
        if runtime_mode == _RUNTIME_STATE.requested_mode:
            print(
                "Runtime is already configured in this kernel. "
                "Keeping the existing settings."
            )
            _print_runtime_summary(_RUNTIME_STATE)
            return _RUNTIME_STATE
        raise RuntimeError(
            "TensorFlow runtime is already configured in this kernel. "
            "To switch RUNTIME_MODE, use 'Restart & Run All'."
        )

    platform = detect_notebook_platform()
    _validate_platform_request(runtime_mode, platform)

    repo_root = _resolve_repo_root()
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    dependencies_installed = False
    if platform in {"colab", "kaggle"}:
        dependencies_installed = install_cloud_requirements(
            repo_root / notebook_requirements
        )

    if "tensorflow" in sys.modules:
        raise RuntimeError(
            "TensorFlow was imported before runtime setup. "
            "Move the runtime cell above any TensorFlow/Keras import and then "
            "use 'Restart & Run All'."
        )

    import tensorflow as tf

    if runtime_mode == "auto" or runtime_mode.endswith("-gpu"):
        physical_gpus = tuple(tf.config.list_physical_devices("GPU"))
    else:
        physical_gpus = ()
    effective_mode = resolve_effective_mode(runtime_mode, platform, physical_gpus)
    compute_device = configure_tensorflow_devices(effective_mode, physical_gpus)
    visible_gpus = tuple(
        device.name for device in tf.config.list_logical_devices("GPU")
    )

    _RUNTIME_STATE = RuntimeInfo(
        requested_mode=runtime_mode,
        effective_mode=effective_mode,
        platform=platform,
        repo_root=str(repo_root),
        visible_gpus=visible_gpus,
        compute_device=compute_device,
        cloud_bootstrap=platform in {"colab", "kaggle"},
        dependencies_installed=dependencies_installed,
    )
    _print_runtime_summary(_RUNTIME_STATE)
    return _RUNTIME_STATE


def normalize_runtime_mode(runtime_mode: str) -> str:
    mode = (runtime_mode or "auto").strip().lower()
    if mode not in VALID_RUNTIME_MODES:
        supported = ", ".join(sorted(VALID_RUNTIME_MODES))
        raise ValueError(
            f"Unsupported RUNTIME_MODE {runtime_mode!r}. Supported values: {supported}."
        )
    return mode


def resolve_effective_mode(
    runtime_mode: str, platform: str, physical_gpus: tuple[object, ...]
) -> str:
    if runtime_mode == "auto":
        device_kind = "gpu" if physical_gpus else "cpu"
        return f"{platform}-{device_kind}"

    requested_platform, _ = runtime_mode.split("-", 1)
    if requested_platform != platform:
        raise RuntimeError(
            f"RUNTIME_MODE={runtime_mode!r} expects platform {requested_platform!r}, "
            f"but the current environment looks like {platform!r}."
        )
    return runtime_mode


def configure_tensorflow_devices(
    effective_mode: str, physical_gpus: tuple[object, ...]
) -> str:
    import tensorflow as tf

    _, device_kind = effective_mode.split("-", 1)

    if device_kind == "cpu":
        tf.config.set_visible_devices([], "GPU")
        return "CPU"

    if not physical_gpus:
        raise RuntimeError(
            "GPU mode was requested, but TensorFlow cannot see any GPU.\n"
            "If you are on a local machine, switch to 'local-cpu' or finish the "
            "local GPU setup.\n"
            "If you are in Colab/Kaggle, enable a GPU accelerator in the runtime "
            "settings and restart the notebook."
        )

    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    return "GPU"


def install_cloud_requirements(requirements_path: Path) -> bool:
    packages = filtered_cloud_requirements(requirements_path)
    if not packages:
        print("Cloud runtime: no extra course packages need installation.")
        return False

    print("Cloud runtime: installing course packages without TensorFlow/Jupyter...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", *packages],
        check=True,
    )
    return True


def filtered_cloud_requirements(requirements_path: Path) -> list[str]:
    requirements_path = Path(requirements_path)
    if not requirements_path.exists():
        raise FileNotFoundError(
            f"Requirements file not found: {requirements_path}."
        )

    packages: list[str] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        package_name = _requirement_name(line)
        if package_name in MANAGED_CLOUD_PACKAGES:
            continue
        packages.append(line)
    return packages


def ensure_cloud_repo_root(
    platform: str, course_repo_https_url: str
) -> tuple[Path, bool]:
    repo_root = cloud_repo_root(platform)
    if _looks_like_repo_root(repo_root):
        return repo_root, False

    if _is_placeholder_repo_url(course_repo_https_url):
        raise RuntimeError(
            "Cloud auto-bootstrap needs a public HTTPS repository URL.\n"
            "Replace COURSE_REPO_HTTPS_URL with the public GitHub URL of the course "
            "repository before running this notebook in Colab or Kaggle."
        )

    repo_root.parent.mkdir(parents=True, exist_ok=True)
    if repo_root.exists() and any(repo_root.iterdir()):
        raise RuntimeError(
            f"Cloud bootstrap expected an empty target directory, but {repo_root} "
            "already exists and does not look like the course repository."
        )

    subprocess.run(
        ["git", "clone", course_repo_https_url, str(repo_root)],
        check=True,
    )
    if not _looks_like_repo_root(repo_root):
        raise RuntimeError(
            f"Cloned repository into {repo_root}, but the course root was not found."
        )
    return repo_root, True


def _resolve_repo_root() -> Path:
    repo_root = Path(__file__).resolve().parent
    if not _looks_like_repo_root(repo_root):
        raise RuntimeError(
            "course_runtime.py was imported from a directory that does not look like "
            "the course repository root."
        )
    return repo_root


def _validate_platform_request(runtime_mode: str, platform: str) -> None:
    if runtime_mode == "auto":
        return

    requested_platform, _ = runtime_mode.split("-", 1)
    if requested_platform != platform:
        raise RuntimeError(
            f"RUNTIME_MODE={runtime_mode!r} does not match the current environment "
            f"{platform!r}. Use one of the {platform}-* modes or leave 'auto'."
        )


def _requirement_name(requirement_line: str) -> str:
    match = re.match(r"^([A-Za-z0-9_.-]+)", requirement_line)
    if not match:
        raise ValueError(f"Cannot parse requirement line: {requirement_line!r}")
    return match.group(1).replace("_", "-").lower()


def _is_placeholder_repo_url(repo_url: str) -> bool:
    return repo_url.strip() == PLACEHOLDER_COURSE_REPO_HTTPS_URL


def _looks_like_repo_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "themes").is_dir()
        and (path / "course_runtime.py").is_file()
    )


def _print_runtime_summary(runtime_info: RuntimeInfo) -> None:
    print("Runtime summary:")
    print(f"- requested mode: {runtime_info.requested_mode}")
    print(f"- effective mode: {runtime_info.effective_mode}")
    print(f"- detected platform: {runtime_info.platform}")
    print(f"- repo root: {runtime_info.repo_root}")
    if runtime_info.visible_gpus:
        print(f"- visible GPUs: {list(runtime_info.visible_gpus)}")
    else:
        print("- visible GPUs: []")
    print(f"- compute device: {runtime_info.compute_device}")
    if runtime_info.platform in {"colab", "kaggle"}:
        install_status = (
            "installed filtered course dependencies"
            if runtime_info.dependencies_installed
            else "no extra dependency installation needed"
        )
        print(f"- cloud setup: {install_status}")
    print(
        "If you change RUNTIME_MODE, use 'Restart & Run All' before continuing."
    )
