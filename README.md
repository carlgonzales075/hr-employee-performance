# hr_employee_performance

This repository is created as a part of fulfilling the requirements for the course subject ML Operations under Professor Rey Tugade Jr.
This is an HR Employee Performance dataset wherein the main target for the trained models is the employee satisfaction score.

# Dependency
This project requires Docker to run. The following python packages are also included in this docker environment:
- `MLFlow` - for ML model registry and monitoring of performance.
- `FastAPI` - to serve the ML model as an API service.
- `MinIO` - serves as a local S3 bucket for storing the artifacts and models created by MLFlow.
- `PostgreSQL` - used for recording activity trails of the above services.
- `Optuna` - for hyperparameter tuning and model optimization.
These services are set to be installed automatically through the `docker-compose`. There is no need to manually install each services separately.

# Dataset
The dataset can be downloaded here:
- https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data
  
Once downloaded create a folder in the root folder named: `dataset` and put the .csv file within.

**Update: 

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
**Env Setup**
Rename `.env.example` to `.env`.

**Docker Setup**
Access the project root folder using bash and then run:
```
docker-compose up -d --build
```
This should create the necessary containers and will automatically create the S3 bucket via `minio_init.sh`.

**MinIO Setup**
Once complete, go to your preferred browser and access `http://localhost:9000`. This is the MinIO UI.

Login to MinIO using the following credentials:
- username: `minio_user`
- password: `minio_pwd`

Go to `Access Keys` and then create the access key. Store the `Access Key` and `Secret Access Key` in your new `.env` similar below:
```
MINIO_ACCESS_KEY=${PUT_ACCESS_KEY_HERE}
MINIO_SECRET_ACCESS_KEY=${PUT_SECRET_ACCESS_KEY_HERE}
```
These environment variables should already be available on the `.env` file and changing the values on the right should be sufficient.


# References
Mexwell. (2024, September 4). 👩🏽 💻 Employee Performance and Productivity Data. Kaggle. https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data
MLflow with Optuna: Hyperparameter Optimization and Tracking. (n.d.). https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs.html
Nolan, A. (2024, June 27). Optuna Hyperparameter Tuning - Ryan Nolan Data. Ryan Nolan Data - My WordPress Blog. https://ryannolandata.com/optuna-hyperparameter-tuning/
