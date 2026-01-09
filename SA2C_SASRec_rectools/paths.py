from pathlib import Path


def make_run_dir(dataset_name: str, config_name: str) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    run_dir = repo_root / "logs" / "SA2C_SASRec_rectools" / dataset_name / config_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_dataset_root(dataset: str) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    if dataset == "yoochoose":
        return repo_root / "RC15"
    if dataset == "retailrocket":
        return repo_root / "Kaggle"
    raise ValueError("dataset must be one of: yoochoose | retailrocket")


__all__ = ["make_run_dir", "resolve_dataset_root"]

