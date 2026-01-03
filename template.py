from pathlib import Path
import os

while True:
    project = input("Enter your Project name")
    if project!="":
        break

list_of_files = [
"project_name/__init__.py",
"project_name/Data_preparation.py", 

"project_name/Data_injestion.py",
"project_name/Data_generation.py",
"app.py","Fastapi.py","Readme.md","streamlit.py",".env",
"Requirements.txt","logger.py","exception.py","Dockerfile_backend",
"Dockerfile_frontend","gitignore","helper.ipynb","logger.py",
]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir,file_name = os.path.split(file_path)


    if file_dir!="":
        os.makedirs(file_dir,exist_ok=True)

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path)):
        with open(file_path,"w") as f:
            pass