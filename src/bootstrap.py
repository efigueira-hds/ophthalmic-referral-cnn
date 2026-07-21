"""Colab/local environment detection, path resolution and Drive mount.

No side effects at import time -- mounting Drive and touching sys.path only
happen inside setup(), when a notebook calls it explicitly.
"""
from pathlib import Path

REPO_NAME = "ophthalmic-referral-cnn"


def setup():
    """Detect the runtime, mount Drive (Colab) or register cwd (local).

    Returns True if running on Colab, False otherwise.
    """
    try:
        import google.colab  # noqa: F401
        from google.colab import drive
        drive.mount('/content/drive')
        import sys
        REPO = '/content/drive/MyDrive/ophthalmic-referral-cnn'
        sys.path.insert(0, REPO)
        in_colab = True
    except ImportError:
        import os, sys
        sys.path.insert(0, os.getcwd())
        in_colab = False
    return in_colab


def get_repo_dir(in_colab):
    """Root of the repo: on Drive in Colab, the git root when local."""
    if in_colab:
        return Path('/content/drive/MyDrive') / REPO_NAME
    repo_root = Path.cwd().resolve()
    while not (repo_root / '.git').exists() and repo_root != repo_root.parent:
        repo_root = repo_root.parent
    return repo_root


def get_data_dir(in_colab, repo_dir):
    """Root that contains the ocular/ dataset folder.

    In Colab the dataset lives directly under MyDrive, a level above the
    repo. Locally it is expected inside the repo root.
    """
    return Path('/content/drive/MyDrive') if in_colab else repo_dir


def get_artifact_dirs(repo_dir):
    """Paths under artifacts/. Callers create the ones they need."""
    artifacts_dir = Path(repo_dir) / "artifacts"
    return {
        "artifacts": artifacts_dir,
        "models": artifacts_dir / "models",
        "preds": artifacts_dir / "preds",
        "history": artifacts_dir / "history",
    }
