from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch
import joblib
import numpy as np


class MessageModel:
    def __init__(
        self, bert_path: str, logreg_path: str, weights: dict, id_to_label: dict
    ):
        self.weights = weights
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path, local_files_only=True)
        self.predict_bert = AutoModelForSequenceClassification.from_pretrained(
            bert_path, local_files_only=True
        ).eval()
        self.logreg = joblib.load(logreg_path)
        self.id_to_label = id_to_label

    def predict_proba_bert(self, text: str):
        tokenized_text = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.predict_bert(**tokenized_text)
        probs = torch.softmax(outputs.logits, dim=1).numpy()
        return probs

    def predict_proba_logreg(self, text: str):
        probs = self.logreg.predict_proba([text])
        return probs

    def _predict(self, bert_probs: np.ndarray, logreg_probs: np.ndarray):
        n_classes = bert_probs.shape[1]
        probs = np.zeros_like(bert_probs)
        for k in range(n_classes):
            w_b = self.weights[k]
            probs[:, k] = w_b * bert_probs[:, k] + (1 - w_b) * logreg_probs[:, k]

        return [self.id_to_label[x] for x in np.argmax(probs, axis=1)]

    def predict(self, text: str):
        bert_probs = self.predict_proba_bert(text)
        logreg_probs = self.predict_proba_logreg(text)
        return self._predict(bert_probs, logreg_probs)[0]
