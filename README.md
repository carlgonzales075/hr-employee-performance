# hr_employee_performance

This repository is created as a part of fulfilling the requirements for the course subject ML Operations under Professor Rey Tugade Jr.

This is an HR Employee Performance dataset wherein the main target for the trained models is the employee satisfaction score.

# Dependency
This project requires Docker to run.

# Dataset
The dataset can be downloaded here:
- https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data
  
Once downloaded create a folder in the root folder named: `dataset` and put the .csv file within.

# File Structure
```
hr_employee_performance/
|── dataset/
|   |── Extended_Employee_Performance_and_Productivity_Data.csv
│── fastapi/
│   │── app/
│   │   ├── api.py
│   ├── Dockerfile
│   ├── requirements.txt
│
│── mlflow/
│   ├── Dockerfile
│
│── notebooks/
│   │── commons/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── commons.py
│   │   ├── data_loading.py
│   ├── model_trainer.ipynb
│   ├── predictor.ipynb
│
├── .env
├── .gitignore
├── docker-compose.yaml
├── minio_init.sh
├── pyproject.toml
├── README.md
├── setup.cfg
├── setup.py
```
# Setup
Access the project root folder using bash and then run:
```
docker-compose up -d --build
```
This should create the necessary containers and will automatically create the S3 bucket via `minio_init.sh`.


# References
Mexwell. (2024, September 4). 👩🏽 💻 Employee Performance and Productivity Data. Kaggle. https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data
