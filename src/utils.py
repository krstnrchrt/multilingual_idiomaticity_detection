import os
import shutil
from pathlib import Path

import src.settings as sts

def solve_secrets_related_env(
        env_name: str,
        env_secrets_filepath: Path | None = None,
        default_value: str | None = None
) -> str:
    """
    (Prio-1) Defined in secrets file
    (Prio-2) Defined as the environment variable
    """
    # Try to read out environment variable
    env_value: str = os.environ.get(env_name)

    # Check if a secrets file is associated
    if env_secrets_filepath is None:

        # If not, a default value must be defined
        if default_value is None:
            # A default value must then be defined
            raise ValueError(
                f"Missing environment variable '{env_name}'! "
                f"If no secrets filepath is provided, a default value must be passed."
            )

        # Return the value of the environment variable, otherwise the passed default value
        return env_value if env_value is not None else default_value

    # Check if the file exists
    if not env_secrets_filepath.exists():
        raise FileNotFoundError(
            f"Missing environment variable '{env_name}! "
            f"The referenced secrets filepath '{env_secrets_filepath}' doesn't exist."
        )

    # Read out the content of the secrets file
    secrets_file_content: str = env_secrets_filepath.read_text().strip()

    # If the secrets file contains something, return this
    if (secrets_file_content is not None) and (secrets_file_content != ""):
        return secrets_file_content

    # If the secrets file was empty AND the environment variable - that's an error
    if env_value is None:
        raise KeyError(
            f"Missing environment variable '{env_name}'! "
            f"If the provided secrets file is empty, the environment variable must exist."
        )

    # Finally, return the value of the existing environment variable
    return env_value

def retrieve_artifact_path_from_nested_best_artifact_directory(dir_best_artifact: Path) -> Path:
    elements: list[Path] = list(dir_best_artifact.iterdir())
    if len(elements) != 1:
        raise FileNotFoundError("Only one artifact per best artifact folder allowed!")

    return dir_best_artifact / elements[0].name


def copy_dir_contents(src: Path, dst: Path):
    """Copy content of src directory into dst directory."""
    dst.mkdir(parents=True, exist_ok=True)
    for entry in os.scandir(src):
        source_path = Path(entry.path)
        destination_path = dst / source_path.name
        if entry.is_dir():
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, destination_path)


def define_experiment_directory_names(experiment_name: str):
    """Returns: dir_experiment, dir_artifacts, dir_ray, dir_best"""
    return (
        dir_experiment := sts.DIR_PATH_RESULTS_EXPERIMENTS / experiment_name,
        dir_experiment / sts.DIR_NAME_RESULTS_EXPERIMENTS_ARTIFACTS,
        dir_experiment / sts.DIR_NAME_RESULTS_EXPERIMENTS_RAY,
        dir_experiment / sts.DIR_NAME_RESULTS_BEST
    )
