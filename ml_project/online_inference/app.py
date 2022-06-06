import os
from typing import List, Union
import pickle


from fastapi import FastAPI, Response, status
from pydantic import BaseModel, conlist
from ml_project.model.fit_predict import predict as mod_predict


app = FastAPI()


class HeartDisease(BaseModel):
    data: List[conlist(Union[int, float], min_items=13, max_items=13)]
    columns: conlist(str, min_items=13, max_items=13)


class Prob(BaseModel):
    prod: float


@app.on_event("startup")
async def load_model():
    global model
    path = os.getenv("MODEL_PATH")
    try:
        model = pickle.load(open(path, 'rb'))
    except:
        model = None
    return


@app.get("/")
def say_hi():
    return {"message": "Hi=)"}


@app.get("/predict", response_model=List[Prob], status_code=200)
def predict(request: HeartDisease):
    return [
        Prob(prod=x) for x in mod_predict(model, request.data, request.columns)
    ]


@app.get("/health",  status_code=200)
def check_health(response: Response):
    if model is None:
        response.status_code = status.HTTP_425_TOO_EARLY
        return {"message": 'Model not download!'}
    return {"message": "Everything fine"}
