"""
utils/paths.py
Cross-environment path resolver for local dev and Kaggle notebooks.

When running on Kaggle:
  - Datasets are mounted under /kaggle/input/<dataset-name>/
  - The working/output directory is /kaggle/working/

When running locally:
  - Paths are resolved relative to the repository root (parent of the
    script's directory).

For Kaggle auto-detection of splits to work, the uploaded dataset should
contain files named exactly:
  train.csv, test.csv, valid.csv (or train/, test/, valid/ folders).
"""

import os


def is_kaggle() -> bool:
    """Return True if the current environment is a Kaggle notebook."""
    return os.path.exists("/kaggle/input")


def _find_kaggle_data_dir():
    """Scan /kaggle/input for the dataset that contains our CSV files."""
    input_dir = "/kaggle/input"
    if not os.path.isdir(input_dir):
        return None
    for name in sorted(os.listdir(input_dir)):
        candidate = os.path.join(input_dir, name)
        if os.path.isdir(candidate):
            if any(
                os.path.exists(os.path.join(candidate, f))
                for f in ("veremi.csv", "split_A.csv", "split_B.csv", "split_C.csv")
            ):
                return candidate
    return None


def get_project_paths(current_file=None):
    """
    Return a dict with standard project paths.

    Usage inside a script:
        paths = get_project_paths(__file__)
        DATA_DIR = paths["DATA_DIR"]
    """
    if is_kaggle():
        kaggle_data = _find_kaggle_data_dir()
        if kaggle_data:
            DATA_DIR = kaggle_data
        else:
            DATA_DIR = os.path.join("/kaggle", "working", "data")

        PROJECT_ROOT = "/kaggle/working"
        TEACHERS_DIR = os.path.join(PROJECT_ROOT, "teachers")
        TRAINING_DIR = os.path.join(PROJECT_ROOT, "training")
        EVALUATION_DIR = os.path.join(PROJECT_ROOT, "evaluation")
        AGGREGATOR_DIR = os.path.join(PROJECT_ROOT, "aggregator")
        STUDENT_DIR = os.path.join(PROJECT_ROOT, "student")
    else:
        if current_file:
            script_dir = os.path.dirname(os.path.abspath(current_file))
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if os.path.basename(script_dir) == "utils":
                script_dir = os.path.dirname(script_dir)

        PROJECT_ROOT = script_dir
        DATA_DIR = os.path.join(PROJECT_ROOT, "data")
        TEACHERS_DIR = os.path.join(PROJECT_ROOT, "teachers")
        TRAINING_DIR = os.path.join(PROJECT_ROOT, "training")
        EVALUATION_DIR = os.path.join(PROJECT_ROOT, "evaluation")
        AGGREGATOR_DIR = os.path.join(PROJECT_ROOT, "aggregator")
        STUDENT_DIR = os.path.join(PROJECT_ROOT, "student")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TEACHERS_DIR, exist_ok=True)
    os.makedirs(TRAINING_DIR, exist_ok=True)
    os.makedirs(EVALUATION_DIR, exist_ok=True)

    return {
        "PROJECT_ROOT": PROJECT_ROOT,
        "DATA_DIR": DATA_DIR,
        "TEACHERS_DIR": TEACHERS_DIR,
        "TRAINING_DIR": TRAINING_DIR,
        "EVALUATION_DIR": EVALUATION_DIR,
        "AGGREGATOR_DIR": AGGREGATOR_DIR,
        "STUDENT_DIR": STUDENT_DIR,
    }


def resolve_dataset_path(filename: str, current_file=None) -> str:
    """Return the full path to a dataset file (works on Kaggle + local)."""
    paths = get_project_paths(current_file)
    return os.path.join(paths["DATA_DIR"], filename)
