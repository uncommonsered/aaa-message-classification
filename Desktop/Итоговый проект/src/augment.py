"""
Модуль аугментации датасета из открытых источников.

Источники:
  1. AlexSham/Toxic_Russian_Comments  (248k, ok.ru, binary: 0=нейтральный, 1=токсичный)
     Разбиваем токсичные на threat / harassment через regex-эвристику.
  2. s-nlp/ru_paradetox               (11k пар toxic→neutral, соцсети)
     Используем ru_toxic_comment → harassment.

Ограничения:
  - ruSpamModels/russian-spam-detection — GATED, требует HF-токен → пропускаем
  - external-класс — нет публичного датасета для Авито-специфики → не аугментируем
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


# ── Regex-эвристика: прямые угрозы в адрес конкретного человека ──────────────
# Критерий: явное намерение причинить вред СОБЕСЕДНИКУ (не третьим лицам в целом)
DIRECT_THREAT_RE = re.compile(
    r"убью\b|убить тебя|убью тебя|я тебя убью|"
    r"пристрелю|застрелю|зарежу|пырну|"
    r"повешу тебя|утоплю тебя|придушу|задушу|"
    r"сдохни\b|сдохните\b|сдыхай\b|подохни\b|умри\b|"
    r"тебе\s+пизда\b|пизда\s+тебе\b|тебе\s+конец\b|"
    r"тебе\s+хана\b|тебе\s+капут\b|"
    r"найду\s+тебя|найду\s+вас|приду\s+за\s+тобой|"
    r"займусь\s+тобой|разберусь\s+с\s+тобой|"
    r"грохну\s+(?:тебя|вас)|замочу\s+(?:тебя|вас)|"
    r"прибью\s+(?:тебя|вас)|прикончу\s+(?:тебя|вас)|"
    r"башку\s+(?:снесу|проломлю|разобью)|"
    r"кости\s+переломаю|ноги\s+переломаю|"
    r"вырежу\s+кишки|выпущу\s+кишки|"
    r"найду\s+и\s+убью|найду\s+и\s+накажу",
    re.IGNORECASE,
)

# Длина текста, типичная для сообщений Авито (отфильтруем слишком длинные)
MIN_LEN = 5
MAX_LEN = 400


def _filter_by_length(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    lens = df[text_col].str.len()
    return df[(lens >= MIN_LEN) & (lens <= MAX_LEN)].copy()


def _classify_toxic(texts: pd.Series) -> pd.Series:
    """
    Возвращает серию меток 'threat' / 'harassment'.
    Прямые угрозы → 'threat', остальное → 'harassment'.
    """
    is_threat = texts.str.contains(DIRECT_THREAT_RE, regex=True, na=False)
    return is_threat.map({True: "threat", False: "harassment"})


# ── Источник 1: AlexSham/Toxic_Russian_Comments ───────────────────────────────
def load_alexsham(cap_harassment: int = 15_000,
                  cap_threat: Optional[int] = None,
                  random_state: int = 42) -> pd.DataFrame:
    """
    Загружает AlexSham/Toxic_Russian_Comments с HuggingFace.
    Берём только токсичные примеры (label=1), классифицируем через regex.

    Returns: DataFrame с колонками text, label, source='alexsham'
    """
    from datasets import load_dataset
    print("[AlexSham] Загрузка с HuggingFace...")
    ds = load_dataset("AlexSham/Toxic_Russian_Comments")

    # Объединяем train + test, берём только токсичные
    dfs = []
    for split in ["train", "test"]:
        part = ds[split].to_pandas()
        dfs.append(part[part["label"] == 1][["text"]])
    df = pd.concat(dfs, ignore_index=True)
    print(f"[AlexSham] Токсичных примеров до фильтрации: {len(df):,}")

    # Фильтр по длине
    df = _filter_by_length(df)
    print(f"[AlexSham] После фильтра длины ({MIN_LEN}–{MAX_LEN} симв.): {len(df):,}")

    # Классификация threat / harassment
    df["label"]  = _classify_toxic(df["text"])
    df["source"] = "alexsham"

    n_threat = (df["label"] == "threat").sum()
    n_hars   = (df["label"] == "harassment").sum()
    print(f"[AlexSham] → threat: {n_threat:,} | harassment: {n_hars:,}")

    # Применяем капы
    parts = []
    if cap_threat is not None:
        parts.append(df[df["label"] == "threat"].sample(
            min(cap_threat, n_threat), random_state=random_state))
    else:
        parts.append(df[df["label"] == "threat"])

    parts.append(df[df["label"] == "harassment"].sample(
        min(cap_harassment, n_hars), random_state=random_state))

    result = pd.concat(parts, ignore_index=True)
    print(f"[AlexSham] После капов → {len(result):,} строк")
    return result


# ── Источник 2: s-nlp/ru_paradetox ───────────────────────────────────────────
def load_paradetox(cap: int = 7_000, random_state: int = 42) -> pd.DataFrame:
    """
    Загружает s-nlp/ru_paradetox.
    Берём колонку ru_toxic_comment (уникальные), все → harassment.

    Returns: DataFrame с колонками text, label='harassment', source='paradetox'
    """
    from datasets import load_dataset
    print("[paradetox] Загрузка с HuggingFace...")
    ds = load_dataset("s-nlp/ru_paradetox")

    dfs = [ds[split].to_pandas()["ru_toxic_comment"] for split in ds]
    unique_toxic = pd.concat(dfs).drop_duplicates().reset_index(drop=True)
    print(f"[paradetox] Уникальных токсичных: {len(unique_toxic):,}")

    df = pd.DataFrame({"text": unique_toxic})
    df = _filter_by_length(df)
    df["label"]  = "harassment"
    df["source"] = "paradetox"
    print(f"[paradetox] После фильтра длины: {len(df):,}")

    # Кап
    if len(df) > cap:
        df = df.sample(cap, random_state=random_state).reset_index(drop=True)
    print(f"[paradetox] После капа → {len(df):,} строк")
    return df


# ── Сборка аугментированного датасета ─────────────────────────────────────────
def build_augmented(base_df: pd.DataFrame,
                    cap_harassment: int = 15_000,
                    cap_threat: Optional[int] = None,
                    random_state: int = 42) -> Tuple[pd.DataFrame, dict]:
    """
    Добавляет внешние данные к базовому датасету.

    base_df: существующий clean датасет (из data_prep.clean())
    Returns: (augmented_df, log_dict)
    """
    log = {}

    # Базовая статистика
    base_counts = base_df["label"].value_counts().to_dict()
    log["base"] = base_counts
    print("\n=== Базовый датасет ===")
    for cls, cnt in sorted(base_counts.items()):
        print(f"  {cls:<12}: {cnt:>7,}")

    # Загружаем внешние источники
    print("\n=== Загрузка AlexSham ===")
    alexsham = load_alexsham(cap_harassment=cap_harassment,
                             cap_threat=cap_threat,
                             random_state=random_state)
    alexsham["synthetic"] = 0

    print("\n=== Загрузка s-nlp/ru_paradetox ===")
    paradetox = load_paradetox(random_state=random_state)
    paradetox["synthetic"] = 0

    log["alexsham"]  = alexsham["label"].value_counts().to_dict()
    log["paradetox"] = paradetox["label"].value_counts().to_dict()

    # Объединяем с базой (только нужные колонки)
    base_slim = base_df[["text", "label", "source", "synthetic"]].copy()
    combined = pd.concat([base_slim, alexsham[["text", "label", "source", "synthetic"]],
                          paradetox[["text", "label", "source", "synthetic"]]],
                         ignore_index=True)

    # Финальная дедупликация
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    log["dedup_removed"] = before_dedup - len(combined)
    print(f"\nДедупликация: -{log['dedup_removed']:,} строк")

    aug_counts = combined["label"].value_counts().to_dict()
    log["augmented"] = aug_counts
    print("\n=== Аугментированный датасет ===")
    for cls in ["normal", "external", "spam", "harassment", "threat"]:
        b = base_counts.get(cls, 0)
        a = aug_counts.get(cls, 0)
        delta = a - b
        sign = "+" if delta >= 0 else ""
        print(f"  {cls:<12}: {b:>7,} → {a:>7,}  ({sign}{delta:,})")

    return combined, log


def augmentation_report(log: dict) -> pd.DataFrame:
    """Таблица: было / добавлено / стало по каждому классу."""
    classes = ["normal", "external", "spam", "harassment", "threat"]
    rows = []
    for cls in classes:
        base   = log["base"].get(cls, 0)
        alexsh = log["alexsham"].get(cls, 0)
        para   = log["paradetox"].get(cls, 0)
        total  = log["augmented"].get(cls, 0)
        rows.append({
            "Класс":          cls,
            "До (базовый)":   base,
            "AlexSham":       alexsh,
            "ru_paradetox":   para,
            "Итого":          total,
            "Прирост %":      round((total / base - 1) * 100, 1) if base > 0 else "∞",
        })
    return pd.DataFrame(rows).set_index("Класс")
