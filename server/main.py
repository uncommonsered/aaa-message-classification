from fastapi import FastAPI
from pydantic import BaseModel

from model import MessageModel


app = FastAPI()


BERT_PATH = "../model/pretrained_models/bert_tiny_ens"
LOGREG_PATH = "../model/pretrained_models/logreg_pipeline.pkl"

ID_TO_LABEL = {
    0: "external",
    1: "harassment",
    2: "normal",
    3: "threat",
}

WEIGHTS = {
    0: 0.1,   # external
    1: 0.2,   # harassment
    2: 0,   # normal
    3: 0.6,   # threat
}


model = MessageModel(
    bert_path=BERT_PATH,
    logreg_path=LOGREG_PATH,
    weights=WEIGHTS,
    id_to_label=ID_TO_LABEL,
)


class TextData(BaseModel):
    text: str


@app.post("/predict")
def predict(data: TextData):
    label = model.predict(data.text)

    return {
        "label": label
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "bert_loaded": model.predict_bert is not None,
        "logreg_loaded": model.logreg is not None,
    }