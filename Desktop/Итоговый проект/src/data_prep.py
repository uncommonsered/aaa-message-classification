"""
Модуль подготовки датасета для классификации сообщений.

Пайплайн:
  1. load_raw()         — загрузка исходных данных
  2. merge_with_labels() — объединение сырых данных с разметкой
  3. clean()            — очистка (NaN, дубли, пустые строки)
  4. add_text_features() — добавление признаков длины
  5. make_train_ready() — балансировка, источник (real/synthetic)
  6. split()            — разбивка train/val/test без утечек
  7. save_splits()      — сохранение на диск
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


# ── Константы ─────────────────────────────────────────────────────────────────
CLASS_ORDER = ["normal", "external", "spam", "harassment", "threat"]

# Целевые размеры класса normal при балансировке (примерно 5x от external)
NORMAL_CAP = 40_000


# ── 1. Загрузка ───────────────────────────────────────────────────────────────
def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"message_id", "text", "item_title"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Ожидались колонки {expected}, получены {set(df.columns)}")
    return df


def load_labeled(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"text", "sentiment", "synthetic"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Ожидались колонки {expected}, получены {set(df.columns)}")
    df = df.rename(columns={"sentiment": "label"})
    df["source"] = df["synthetic"].map({0: "real", 1: "synthetic"})
    return df


# ── 2. Объединение ────────────────────────────────────────────────────────────
def merge_with_labels(raw: pd.DataFrame, labeled: pd.DataFrame) -> pd.DataFrame:
    """
    Джойн raw (содержит message_id) с labeled (содержит label).
    Для реальных строк берём message_id из raw; синтетические добавляем отдельно.
    """
    real_labeled = labeled[labeled["source"] == "real"].copy()
    synthetic = labeled[labeled["source"] == "synthetic"].copy()

    # join по тексту — берём первое совпадение для дублей
    raw_deduped = raw.drop_duplicates(subset=["text"])
    merged = real_labeled.merge(
        raw_deduped[["message_id", "text", "item_title"]],
        on="text", how="left", suffixes=("_label", "")
    )
    # Если в labeled уже есть item_title — используем его
    if "item_title_label" in merged.columns:
        merged["item_title"] = merged["item_title"].fillna(merged["item_title_label"])
        merged.drop(columns=["item_title_label"], inplace=True)

    # Синтетические не имеют message_id
    synthetic["message_id"] = None

    df = pd.concat([merged, synthetic], ignore_index=True)
    return df


# ── 3. Очистка ────────────────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # Удалить строки без текста или метки
    df = df.dropna(subset=["text", "label"])

    # Убрать дубликаты (text + label); если один текст размечен по-разному — оставить первый
    df = df.drop_duplicates(subset=["text", "label"])

    # Убрать строки с пустым/сверхкоротким текстом (< 2 символов)
    df = df[df["text"].str.strip().str.len() >= 2]

    # Сбросить индекс
    df = df.reset_index(drop=True)
    print(f"Очистка: {before:,} → {len(df):,} строк (удалено {before - len(df):,})")
    return df


# ── 4. Текстовые признаки ────────────────────────────────────────────────────
def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_len"]   = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    return df


# ── 5. Подготовка к обучению ─────────────────────────────────────────────────
def make_train_ready(df: pd.DataFrame,
                     normal_cap: int = NORMAL_CAP,
                     random_state: int = 42) -> pd.DataFrame:
    """
    Балансировка: ограничиваем класс normal сверху, все остальные оставляем.
    Синтетические данные идут только в обучение (помечаются флагом).
    """
    rng = np.random.default_rng(random_state)
    parts = []

    for cls in CLASS_ORDER:
        subset = df[df["label"] == cls]
        if cls == "normal" and len(subset) > normal_cap:
            subset = subset.sample(normal_cap, random_state=random_state)
        parts.append(subset)

    result = pd.concat(parts, ignore_index=True)
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return result


# ── 6. Разбивка train / val / test ───────────────────────────────────────────
def split(df: pd.DataFrame,
          val_size: float = 0.10,
          test_size: float = 0.10,
          random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Стратифицированная разбивка.
    Правило: val и test содержат ТОЛЬКО оригинальные Авито-примеры (source=='real').
    Все остальные источники (synthetic, alexsham, paradetox, ...) — только в train.

    Returns: train, val, test
    """
    real      = df[df["source"] == "real"].copy()
    train_only = df[df["source"] != "real"].copy()   # synthetic + все внешние источники

    # Стратифицируем real по классам
    val_parts, test_parts, train_parts = [], [], []

    for cls in CLASS_ORDER:
        cls_df = real[real["label"] == cls].sample(frac=1, random_state=random_state)
        n = len(cls_df)

        n_val  = max(1, int(n * val_size))
        n_test = max(1, int(n * test_size))

        val_parts.append(cls_df.iloc[:n_val])
        test_parts.append(cls_df.iloc[n_val:n_val + n_test])
        train_parts.append(cls_df.iloc[n_val + n_test:])

    train = pd.concat(train_parts + [train_only], ignore_index=True)
    val   = pd.concat(val_parts,  ignore_index=True)
    test  = pd.concat(test_parts, ignore_index=True)

    # Перемешать train
    train = train.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return train, val, test


# ── 7. Сохранение ────────────────────────────────────────────────────────────
def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                out_dir: str | Path = "data/processed") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv",   index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    print(f"Сохранено в {out_dir}:")
    print(f"  train.csv — {len(train):,} строк")
    print(f"  val.csv   — {len(val):,} строк")
    print(f"  test.csv  — {len(test):,} строк")


# ── Утилиты ──────────────────────────────────────────────────────────────────
def split_stats(train: pd.DataFrame, val: pd.DataFrame,
                test: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Возвращает словарь {split: {class: count}} для визуализации."""
    result = {}
    for name, df in [("train", train), ("val", val), ("test", test)]:
        result[name] = df["label"].value_counts().to_dict()
    return result


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Таблица качества: пропуски, дубли, размер по классу."""
    rows = []
    for cls in CLASS_ORDER:
        sub = df[df["label"] == cls]
        real_cnt  = (sub["source"] == "real").sum()      if "source" in sub.columns else len(sub)
        synth_cnt = (sub["source"] == "synthetic").sum() if "source" in sub.columns else 0
        rows.append({
            "Класс":         cls,
            "Всего":         len(sub),
            "Реальных":      real_cnt,
            "Синтетических": synth_cnt,
            "% синтетики":   round(synth_cnt / len(sub) * 100, 1) if len(sub) else 0,
            "Ср. длина":     round(sub["text_len"].mean(), 1) if "text_len" in sub.columns else "-",
        })
    return pd.DataFrame(rows).set_index("Класс")
