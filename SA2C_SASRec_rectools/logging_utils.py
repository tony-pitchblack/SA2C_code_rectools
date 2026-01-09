from pathlib import Path
import logging

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


def dump_config(cfg: dict, run_dir: Path):
    with open(run_dir / "config.yml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


__all__ = ["configure_logging", "dump_config"]

