import os
from pathlib import Path


list_of_files = [
    "src/__init__.py",
    "src/utils.py",
    "src/prompt.py",
    "src/logger.py",
    ".env",
    "setup.py",
    "reseach/trials.ipynb",
    "app.py",
    "requirements.txt",
    "store_index.py",
    "static/.gitkeep",
    "templates/chat.html"

]

for file_path in list_of_files:
    file_path=Path(file_path)
    folder,file=os.path.split(file_path)
    if folder!="":
        os.makedirs(folder,exist_ok=True)
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path)==0):
        with open(file_path,'w') as f:
            pass