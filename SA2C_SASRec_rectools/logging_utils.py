from pathlib import Path
import logging
import shutil
import subprocess

import yaml


def configure_logging(run_dir: Path, debug: bool):
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(levelname)s: %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "train.log"),
        ],
        force=True,
    )


def _find_git_root(start: Path) -> Path | None:
    for p in [start, *start.parents]:
        if (p / ".git").exists():
            return p
    return None


def _get_git_repo_commit_hash(repo_root: Path) -> str | None:
    if shutil.which("git") is None:
        return None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None
    return out or None


def dump_config(cfg: dict, run_dir: Path):
    cfg_out = dict(cfg)
    repo_root = _find_git_root(Path(__file__).resolve())
    if repo_root is not None:
        commit_hash = _get_git_repo_commit_hash(repo_root)
        if commit_hash is not None:
            cfg_out["git_repo_commit_hash"] = commit_hash
    with open(run_dir / "config.yml", "w") as f:
        yaml.safe_dump(cfg_out, f, sort_keys=False)


__all__ = ["configure_logging", "dump_config"]

