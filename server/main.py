from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch
import joblib
import numpy as np

app = FastAPI()

MODEL_PATH = "../model/pretrained_models/bert_tiny_ens"

ID_TO_LABEL = {0: "harassment", 1: "normal", 2: "threat", 3: "external"}
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True).eval()

CLASS_WEIGHTS = {'external': 1, 'normal': 1.25, 'harassment' : 0.5, 'threat': 1}
LOG_REG = joblib.load('../model/pretrained_models/logreg_pipeline.pkl')


class TextData(BaseModel):
    text: str

def predict_with_weights(pipeline, X, class_weights):
    """
    pipeline: стандартная мультиклассовая модель
    class_weights: словарь весов для классов, например {'external': 1, 'normal': 1.25, 'harassment' : 0.5, 'threat': 1}
    Возвращает имя класса с учетом весового словаря
    """
    logits = pipeline.predict_proba(X) 
    classes = pipeline.classes_
    weights_vector = np.array([class_weights[cls] for cls in classes])
    weighted_logits = logits * weights_vector
    best_class_indices = np.argmax(weighted_logits, axis=1)
    return np.array([classes[idx] for idx in best_class_indices])

@app.post("/predict")
def predict(data: TextData):
    tokenaized_text = TOKENIZER(data.text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = MODEL(**tokenaized_text)

    predicted_class_id = torch.argmax(outputs.logits, dim=-1).item()
    predicted_label_bert = ID_TO_LABEL[predicted_class_id]

    # если берт отметил как нормальное сообщение, оставляем
    if predicted_label_bert == "normal":
        return {"label": "normal"}

    # иначе возвращает ответ логистической регрессии
    predicted_label_logreg = predict_with_weights(
        LOG_REG,
        [data.text],
        CLASS_WEIGHTS
    )[0]

    return {"label": predicted_label_logreg}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "bert_loaded": MODEL is not None,
        "logreg_loaded": LOG_REG is not None
    }