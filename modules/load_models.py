import os
import pickle
from tensorflow.keras.models import load_model

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def save_forecast_assets(ticker, model, df_processed, actual_next, pred_next,residual_std):
    """ Save model and forecasting data locally """
    model_path = f"{MODEL_DIR}/{ticker}_model.keras"
    data_path  = f"{MODEL_DIR}/{ticker}_data.pkl"

    # save model
    model.save(model_path)

    # save df + data
    with open(data_path, "wb") as f:
        pickle.dump({
            "df_processed": df_processed,
            "actual_next": actual_next,
            "pred_next": pred_next,
            "residual_std": residual_std
        }, f)
        
def load_forecast_assets(ticker):
    """ Load model + data if exists, else return None """
    model_path = f"{MODEL_DIR}/{ticker}_model.keras"
    data_path  = f"{MODEL_DIR}/{ticker}_data.pkl"

    if not (os.path.exists(model_path) and os.path.exists(data_path)):
        return None, None, None, None, None

    model = load_model(model_path)

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    return (
        model,
        data["df_processed"],
        data["actual_next"],
        data["pred_next"],
        data["residual_std"]
    )
