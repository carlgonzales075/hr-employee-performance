from fastapi import FastAPI, HTTPException
import logging
import mlflow
import pandas as pd

app = FastAPI()

logging.basicConfig(level=logging.INFO)

mlflow.set_tracking_uri("http://tracking_server:5000")
MLFLOW_MODEL_URI = "models:/gboost_regressor@champion"

@app.post("/predict")
async def predict(data: list[dict]):
    """
    Receives JSON data, loads the MLflow model, makes predictions, 
    and returns predictions in JSON format.

    Parameters
    ----------
    data : list
        List of dictionaries representing input features.
    
    Returns
    -------
    dict
        JSON object with row numbers as keys and predicted values as lists.
    """
    logging.info("Received request. Starting prediction.")
    try:
        df = pd.DataFrame(data)
        logging.info("Converted request to pandas dataframe.")
        model = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)
        logging.info("Completed downloading the model."
                     f"\nmodel URI: {MLFLOW_MODEL_URI}")
        logging.info("Starting predictions...")
        predictions = model.predict(df)
        logging.info("Completed predictions. Returning response to client...")
        response = {
            "row_number": list(range(len(predictions))),
            "predicted_value": predictions.tolist(),
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
