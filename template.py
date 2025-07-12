import os
from pathlib import Path
import logging

# Correct logging setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "cnn classifier"  # change to your desired project name

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "test.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir = filepath.parent

    # Create directories if they don't exist
    if filedir != Path("."):
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir}")

    # Create the file if it doesn't exist or is empty
    if not filepath.exists() or filepath.stat().st_size == 0:
        filepath.touch()
        logging.info(f"Created empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")
