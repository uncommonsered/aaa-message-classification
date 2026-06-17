import re
import joblib
import torch
from scipy.sparse import hstack
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel

# Загружаем всё один раз при старте процесса
BUNDLE = joblib.load("../model/pretrained_models/model.joblib")
CLASSES = BUNDLE["classes"]                 # порядок классов из обучения
C2I = {c: i for i, c in enumerate(CLASSES)}
THRESHOLDS = BUNDLE["thresholds"]           # сохранённые пороги для каждого класса

FT_DIR = '../model/pretrained_models/rubert_harassment_model'                  # путь к дообученному RuBERT
TOKENIZER = AutoTokenizer.from_pretrained(FT_DIR, local_files_only=True)
RUBERT = AutoModelForSequenceClassification.from_pretrained(FT_DIR, local_files_only=True).eval()

# Подготовка текста
# Часть пользователей заменяет кириллицу похожими латинскими буквами
LAT2CYR = str.maketrans({"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у",
                         "x": "х", "k": "к", "m": "м", "h": "н", "t": "т", "b": "в"})

def deobfuscate(text):
    """Немного чистим текст от частой маскировки и лишних повторов."""
    s = str(text).lower().replace("ё", "е").translate(LAT2CYR)
    s = re.sub(r"\b(?:[а-я]\s){2,}[а-я]\b", lambda m: m.group(0).replace(" ", ""), s)
    s = re.sub(r"(?<=[а-я])[\*\@\#\$\.\_\^]+(?=[а-я])", "", s)
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    return s

# Мат и распространённые оскорбительные слова
OBSCENE_RE = re.compile(
    r"(?:^|[^а-я])(?:бля(?:дь|ть)?|пизд\w*|хуй\w*|хуе\w*|хуё\w*|хуи\w*|нахуй\w*|хули\b|"
    r"мудак\w*|долбо[её]б\w*|гандон\w*|шлюх\w*|ебать\w*|ебал\w*|ебан\w*|ебуч\w*|заеб\w*|"
    r"наеб\w*|уеб\w*|выеб\w*|чмо\b|тварь|урод\w*|дебил\w*)(?=$|[^а-я])", re.I)
# Явные угрозы и насильственные формулировки
THREAT_RE = re.compile(
    r"(расстрел|пристрел|застрел|убью теб|зарежу|прирежу|сожгу теб|спалю теб|подожгу|"
    r"на кол|кастрир|изнасил|придушу|удавлю|переломаю|глотку перегрыз|башку (снес|пролом)|"
    r"череп пролом|выпущу кишки|шею сверну|голову оторв|на ремни|пристрелю)", re.I)
# Контакты, ссылки и попытки увести диалог за пределы площадки
CONTACT_RE = re.compile(
    r"(@[\w_]{5,}|t\.me/|telegram|телеграм|телега\b|whatsap|ватсап|ватцап|вотсап|"
    r"viber|вайбер|вконтакте|skype|(?:https?://|www\.))", re.I)

# Основная логика предсказания
def tfidf_proba(clean_text, clean_title):
    """Считаем вероятности классов через TF-IDF модель."""
    x = hstack([
        BUNDLE["char_vec"].transform([clean_text]),
        BUNDLE["word_vec"].transform([clean_text]),
        BUNDLE["title_vec"].transform([clean_title]) * 0.25,
    ], format="csr")
    order = [list(BUNDLE["tfidf_lr"].classes_).index(c) for c in CLASSES]

    return BUNDLE["tfidf_lr"].predict_proba(x)[0, order]

@torch.no_grad()
def rubert_harassment(clean_text):
    """Проверяем через RuBERT, насколько текст похож на оскорбление."""
    enc = TOKENIZER([clean_text], truncation=True, max_length=128,
                    padding=True, return_tensors="pt")
    p = torch.softmax(RUBERT(**enc).logits, dim=-1).numpy()[0]

    return float(p[C2I["harassment"]])

def classify(text, title=""):
    """Проверяем классы по очереди: от самых опасных к обычным."""
    clean = deobfuscate(text)
    clean_title = deobfuscate(title or "")
    p = tfidf_proba(clean, clean_title)

    # Сначала ловим угрозы: они важнее остальных классов
    if max(p[C2I["threat"]], float(bool(THREAT_RE.search(clean)))) >= THRESHOLDS["threat"]:
        return "threat"
    
    # Потом проверяем попытки перейти в мессенджеры или по ссылкам
    if max(p[C2I["external"]], float(bool(CONTACT_RE.search(str(text))))) >= THRESHOLDS["external"]:
        return "external"
    
    # Оскорбление засчитываем только если сработало правило и RuBERT согласен
    if OBSCENE_RE.search(clean) and rubert_harassment(clean) >= THRESHOLDS["harassment"]:
        return "harassment"
    
    return "normal"


app = FastAPI()

class Message(BaseModel):
    text: str
    item_title: str = ""

@app.post("/predict")
def predict(msg: Message):
    return {"label": classify(msg.text, msg.item_title)}

@app.get("/health")
def health():
    return {"status": "ok"}

