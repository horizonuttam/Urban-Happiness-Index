from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel


ARTIFACTS_DIR = Path(__file__).parent / 'artifacts'
metadata = joblib.load(ARTIFACTS_DIR / 'preprocessing.joblib')
model = xgb.XGBRegressor()
model.load_model(ARTIFACTS_DIR / 'happiness_xgb.json')

app = FastAPI(title='Urban Happiness XGBoost API')


class HappinessInput(BaseModel):
    gdp_per_capita: float
    healthy_life_expectancy: float
    pm2_5: float
    intentional_homicide_rate: float
    degree_of_urbanization: float
    sdg_index_score: float


def normalize(value, minimum, maximum):
    if minimum == maximum:
        return 5.5
    return 1 + 9 * (value - minimum) / (maximum - minimum)


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(request: HappinessInput):
    values = request.model_dump()
    normalized = {
        f'{feature}_n': normalize(
            values[feature],
            metadata['min_values'][feature],
            metadata['max_values'][feature],
        )
        for feature in metadata['features']
    }
    prediction = float(model.predict(pd.DataFrame([normalized]))[0])
    return {
        'predicted_life_evaluation': prediction,
        'normalized_features': normalized,
    }