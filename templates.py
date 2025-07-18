import os
from pathlib import Path

project_name = "WeatherPredictor"

list_of_files = [
    ".github/workflows/.gitkeep",
    "artifacts/.gitkeep",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/pipelines/__init__.py",
    "src/logging/__init__.py",
    "src/utils/__init__.py",
    "src/exception.py",
    "requirements.txt",
    "setup.py",
    "Dockerfile",
    "notebooks/explore_dataset.ipynb"
]

for files in list_of_files:
    filepath = Path(files)
    folder, file = os.path.split(filepath)
    if folder:
        os.makedirs(folder, exist_ok=True)
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w"):
            pass
    else:
        print("file already exists")

